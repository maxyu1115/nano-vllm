import atexit
from dataclasses import fields
from time import perf_counter
from typing import Callable, Optional
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch.nn as nn

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, INVALID_TOKEN_ID
from nanovllm.engine.scheduler import Scheduler, RunPhase
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model_loader: Optional[Callable[[], nn.Module]] = None, tokenizer_loader: Optional[Callable[[], AutoTokenizer]] = None, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(**config_kwargs)
        self.max_soft_mtp_tokens = config.max_soft_mtp_tokens
        self.soft_mtp_enabled = self.max_soft_mtp_tokens > 1
        self.bot_token_id = config.bot
        self.cot_pad_token_id = config.cot_pad_token
        if self.soft_mtp_enabled:
            assert self.bot_token_id >= 0
            assert self.cot_pad_token_id >= 0
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(model_loader, config, 0, self.events)
        if tokenizer_loader is not None:
            self.tokenizer = tokenizer_loader()
        else:
            assert config.model_path is not None
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
            if prompt[-1] != self.bot_token_id:
                prompt.append(self.bot_token_id)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, phase = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, phase)
        if phase.is_reasoning():
            self.scheduler.postprocess_reasoning(seqs, token_ids)
        else:
            self.scheduler.postprocess_ntp(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.uncompressed_completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if phase.is_prefill() else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens == 0:
                continue
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids, uncompressed_token_ids in output:
                token_ids = [self.cot_pad_token_id if token_id == INVALID_TOKEN_ID else token_id for token_id in token_ids]
                uncompressed_token_ids = [self.cot_pad_token_id if token_id == INVALID_TOKEN_ID else token_id for token_id in uncompressed_token_ids]
                outputs[seq_id] = (token_ids, uncompressed_token_ids)
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{
            "text": self.tokenizer.decode(token_ids),
            "cot_text": self.tokenizer.decode(uncompressed_token_ids),
            "token_ids": token_ids,
            "cot_token_ids": uncompressed_token_ids,
        } for token_ids, uncompressed_token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
