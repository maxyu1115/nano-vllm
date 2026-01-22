import pickle
from typing import Callable, Literal, Optional
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
import nanovllm.engine.sequence as sequence
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import RunPhase
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.attention import Attention
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.soft_mtp import AdaptiveDecodingThresholdPolicy
from nanovllm.utils.context import set_context, get_context, reset_context, DEFAULT_CONTEXT_KEY, MTP_MODULE_CONTEXT_KEY
from nanovllm.utils.loader import load_model


BATCH_SIZE_LIMIT = 512

INVALID_BLOCK_ID = -1

class ModelRunner:

    def __init__(self, model_loader: Optional[Callable[[], torch.nn.Module]], config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.soft_mtp_enabled = config.max_soft_mtp_tokens > 1
        self.max_soft_mtp_tokens = config.max_soft_mtp_tokens
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        sequence.COT_PAD_TOKEN_ID = config.cot_pad_token
        sequence.END_OF_THINK_TOKEN_ID = config.eot
        self.adapt_decode_type: Literal["threshold", "none"] = config.soft_mtp_adaptive_decoding_policy
        self.adapt_decode_policy = AdaptiveDecodingThresholdPolicy() if self.adapt_decode_type == "threshold" else None

        dist.init_process_group("nccl", config.dist_init_method, world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        if model_loader is not None:
            self.model = model_loader()
        else:
            self.model = Qwen3ForCausalLM(hf_config)
            load_model(self.model, config.model_path)
        self.sampler = Sampler()
        print("model loaded")
        self.warmup_model()
        print("warmup done")
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        warmup_phase = RunPhase.PREFILL_MTP if self.soft_mtp_enabled else RunPhase.PREFILL
        self.run(seqs, warmup_phase)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        num_layers = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                num_layers += 1
        block_bytes = 2 * num_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == num_layers

    def prepare_block_tables(self, seqs: list[Sequence]):
        # TODO: check if we need to special handle mtp
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [INVALID_BLOCK_ID] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:seqlen])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            # in the case of soft_mtp_enabled, the last block may be pre-allocated for the MTP module.
            # But seq.num_blocks doesn't include that additional block, so we're good
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != len(seq.block_table) - 1:
                    # assert self.max_soft_mtp_tokens <= 2, "This breaks with k>2"
                    # if not the last block, we can use the full block size
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        assert input_ids.numel() == positions.numel(), f"input_ids.numel()={input_ids.numel()} != positions.numel()={positions.numel()}"
        if block_tables is not None:
            assert input_ids.numel() == slot_mapping.numel(), f"input_ids.numel()={input_ids.numel()} != slot_mapping.numel()={slot_mapping.numel()}"
            assert slot_mapping.numel() == cu_seqlens_q[-1], f"slot_mapping.numel()={slot_mapping.numel()} != cu_seqlens_q[-1]={cu_seqlens_q[-1]}"

        set_context(
            key=DEFAULT_CONTEXT_KEY,
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )
        if self.soft_mtp_enabled:
            set_context(
                key=MTP_MODULE_CONTEXT_KEY,
                is_prefill=True,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
            )
        return input_ids, positions

    def prepare_soft_mtp_prefill_decode(self, seqs: list[Sequence]):
        mtp_positions = []
        mtp_slot_mapping = []
        mtp_context_lens = []
        is_warmup = False
        for seq in seqs:
            if not seq.block_table:    # warmup
                is_warmup = True
                mtp_positions.append(0)
                mtp_slot_mapping.append(0) # append a dummy values for warmup
                continue
            # MTP module positions are off by 1, since the ntp token isn't added to seq yet
            mtp_positions.append(len(seq))
            mtp_context_lens.append(len(seq) + 1)
            if seq.last_block_num_tokens == self.block_size:
                mtp_slot_mapping.append(seq.block_table[-1] * self.block_size)
            else:
                mtp_slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens)

        mtp_positions = torch.tensor(mtp_positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        mtp_slot_mapping = torch.tensor(mtp_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        mtp_context_lens = torch.tensor(mtp_context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        if is_warmup:
            B = len(seqs)
            cu_seqlens = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
            set_context(
                key=MTP_MODULE_CONTEXT_KEY,
                is_prefill=True,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=1,
                max_seqlen_k=1,
                slot_mapping=mtp_slot_mapping,
                block_tables=None,
            )
        else:
            block_tables = self.prepare_block_tables(seqs)
            set_context(key=MTP_MODULE_CONTEXT_KEY, is_prefill=False, slot_mapping=mtp_slot_mapping, context_lens=mtp_context_lens, block_tables=block_tables)
        return mtp_positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            if len(seq.block_table) != seq.num_blocks:
                slot_mapping.append(seq.block_table[-2] * self.block_size + seq.last_block_num_tokens  - 1)
            else:
                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            key=DEFAULT_CONTEXT_KEY,
            is_prefill=False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_soft_mtp_decode(self, seqs: list[Sequence]):
        mtp_input_ids = []
        positions = []
        mtp_positions = []
        slot_mapping = []
        mtp_slot_mapping = []
        context_lens = []
        for seq in seqs:
            # In soft mtp mode, we pass in multiple input tokens, but the transformer only sees 1 token
            # This is because the multiple input tokens are compressed into 1 token before fed into the transformer.
            assert len(seq.next_input_cot_ids) == self.max_soft_mtp_tokens
            mtp_input_ids.append(seq.next_input_cot_ids)
            positions.append(len(seq) - 1)
            mtp_positions.append(len(seq) - 1 + 1)
            context_lens.append(len(seq))
            if len(seq.block_table) != seq.num_blocks:
                # this is the case when we allocated an additional block for the MTP module.
                # Meaning the last tokens fall on the block boundary
                slot_mapping.append(seq.block_table[-2] * self.block_size + seq.last_block_num_tokens  - 1)
                mtp_slot_mapping.append(seq.block_table[-1] * self.block_size)
            else:
                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
                mtp_slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens)

        mtp_input_ids = torch.tensor(mtp_input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        mtp_positions = torch.tensor(mtp_positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        mtp_slot_mapping = torch.tensor(mtp_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        mtp_context_lens = context_lens + 1
        block_tables = self.prepare_block_tables(seqs)
        set_context(key=DEFAULT_CONTEXT_KEY, is_prefill=False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        # MTP module positions are off by 1
        set_context(key=MTP_MODULE_CONTEXT_KEY, is_prefill=False, slot_mapping=mtp_slot_mapping, context_lens=mtp_context_lens, block_tables=block_tables)
        return mtp_input_ids, positions, mtp_positions

    def prepare_sample(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        if not self.soft_mtp_enabled:
            return temperatures, None
        mtp_temperatures = []
        for seq in seqs:
            if seq.soft_mtp_params is None:
                mtp_temperatures.append(seq.temperature)
            else:
                mtp_temperatures.append(seq.soft_mtp_params.mtp_temperature)
        mtp_temperatures = torch.tensor(mtp_temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures, mtp_temperatures

    def _compute_slot_mapping(self, seq: Sequence, positions: list[int]) -> list[int]:
        """Compute slot mapping for given positions in a sequence."""
        return [seq.block_table[p // self.block_size] * self.block_size + p % self.block_size for p in positions]

    def _set_prefill_context_for_section(
        self,
        seqs: list[Sequence],
        positions_per_seq: list[list[int]],
        context_len_per_seq: list[int],
        context_key: str = DEFAULT_CONTEXT_KEY,
    ):
        """Set prefill context for a section of tokens across sequences."""
        all_positions = []
        slot_mapping = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq, positions, ctx_len in zip(seqs, positions_per_seq, context_len_per_seq):
            all_positions.extend(positions)
            slot_mapping.extend(self._compute_slot_mapping(seq, positions))
            seqlen_q = len(positions)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + ctx_len)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(ctx_len, max_seqlen_k)

        block_tables = self.prepare_block_tables(seqs) if cu_seqlens_k[-1] > cu_seqlens_q[-1] else None

        positions_t = torch.tensor(all_positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(
            key=context_key,
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q_t,
            cu_seqlens_k=cu_seqlens_k_t,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping_t,
            block_tables=block_tables,
        )
        return positions_t, cu_seqlens_q_t

    def _extract_restore_data(self, seqs: list[Sequence]) -> tuple[
        list[list[int]], list[list[int]],  # prompt_tokens, prompt_positions
        list[list[list[int]]], list[list[int]],  # cot_multi_ids, cot_positions
    ]:
        """Extract prompt and CoT data for restore prefill.
        
        Returns:
            prompt_tokens_per_seq: Single tokens for prompt section
            prompt_positions_per_seq: Positions for prompt tokens
            cot_multi_ids_per_seq: Multi-token IDs [ntp, mtp] for CoT section
            cot_positions_per_seq: Positions for CoT tokens
        """
        prompt_tokens_per_seq = []
        prompt_positions_per_seq = []
        cot_multi_ids_per_seq = []
        cot_positions_per_seq = []

        for seq in seqs:
            prompt_len = seq.num_prompt_tokens
            num_cot = seq.num_cot_tokens

            # Prompt section
            prompt_tokens_per_seq.append(seq.token_ids[:prompt_len])
            prompt_positions_per_seq.append(list(range(prompt_len)))

            # CoT section - extract multi_input_ids from uncompressed tokens
            if num_cot > 0:
                flat = []
                for block in seq.uncompressed_token_ids_by_block:
                    flat.extend(block)
                cot_multi = []
                for i in range(num_cot):
                    ntp_tok = flat[prompt_len + i * 2]
                    mtp_tok = flat[prompt_len + i * 2 + 1]
                    cot_multi.append([ntp_tok, mtp_tok])
                cot_multi_ids_per_seq.append(cot_multi)
                cot_positions_per_seq.append(list(range(prompt_len, prompt_len + num_cot)))
            else:
                cot_multi_ids_per_seq.append([])
                cot_positions_per_seq.append([])

        return prompt_tokens_per_seq, prompt_positions_per_seq, cot_multi_ids_per_seq, cot_positions_per_seq

    def _restore_prompt_and_cot(
        self,
        seqs: list[Sequence],
        prompt_tokens_per_seq: list[list[int]],
        prompt_positions_per_seq: list[list[int]],
        cot_multi_ids_per_seq: list[list[list[int]]],
        cot_positions_per_seq: list[list[int]],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Restore KV cache for prompt and CoT sections.
        
        Performs:
        1. Prompt section: ntp_prefill + mtp_prefill with single tokens
        2. CoT section: ntp_decode (in prefill mode) + mtp_decode for compressed multi-tokens
        
        Returns:
            cot_ntp_hidden_states: Hidden states from CoT NTP decode (or None if no CoT)
            cot_multi_input_embeds: Multi-token embeddings from CoT (or None if no CoT)
        """
        # Step 1: Prefill prompt section (NTP + MTP)
        all_prompt_tokens = [t for tokens in prompt_tokens_per_seq for t in tokens]
        if all_prompt_tokens:
            prompt_ctx_lens = [len(tokens) for tokens in prompt_tokens_per_seq]
            positions_t, _ = self._set_prefill_context_for_section(seqs, prompt_positions_per_seq, prompt_ctx_lens, DEFAULT_CONTEXT_KEY)
            self._set_prefill_context_for_section(seqs, prompt_positions_per_seq, prompt_ctx_lens, MTP_MODULE_CONTEXT_KEY)

            input_ids = torch.tensor(all_prompt_tokens, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            self.model.ntp_prefill(input_ids, positions_t)
            self.model.mtp_prefill(input_ids, positions_t)
            reset_context()

        # Step 2: Prefill CoT section
        all_cot_multi = [m for multi in cot_multi_ids_per_seq for m in multi]
        cot_ntp_hidden_states = None
        cot_multi_input_embeds = None
        if all_cot_multi:
            prompt_lens = [seq.num_prompt_tokens for seq in seqs]
            cot_ctx_lens = [prompt_lens[i] + len(cot_positions_per_seq[i]) for i in range(len(seqs))]
            positions_t, _ = self._set_prefill_context_for_section(seqs, cot_positions_per_seq, cot_ctx_lens, DEFAULT_CONTEXT_KEY)

            multi_input_ids = torch.tensor(all_cot_multi, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            cot_ntp_hidden_states, cot_multi_input_embeds = self.model.ntp_decode(multi_input_ids, positions_t)
            reset_context()

            # MTP prefill for CoT
            mtp_positions_per_seq = [[p + 1 for p in positions] for positions in cot_positions_per_seq]
            mtp_ctx_lens = [c + 1 for c in cot_ctx_lens]
            mtp_positions_t, _ = self._set_prefill_context_for_section(seqs, mtp_positions_per_seq, mtp_ctx_lens, MTP_MODULE_CONTEXT_KEY)

            all_ntp_output_ids = [m[0] for m in all_cot_multi]
            ntp_output_ids = torch.tensor(all_ntp_output_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

            self.model.mtp_decode(cot_multi_input_embeds, cot_ntp_hidden_states, ntp_output_ids, mtp_positions_t)
            reset_context()

        return cot_ntp_hidden_states, cot_multi_input_embeds

    @torch.inference_mode()
    def _restore_reasoning_phase(
        self,
        seqs: list[Sequence],
        temperatures: torch.Tensor,
        mtp_temperatures: torch.Tensor,
    ) -> list[list[int]] | None:
        """Restore KV cache for sequences in reasoning phase (prompt + CoT).
        
        Generates next [ntp_token, mtp_token] pair for continued CoT generation.
        """
        # Extract and restore prompt + CoT
        prompt_tokens, prompt_positions, cot_multi_ids, cot_positions = self._extract_restore_data(seqs)
        cot_ntp_hidden_states, cot_multi_input_embeds = self._restore_prompt_and_cot(
            seqs, prompt_tokens, prompt_positions, cot_multi_ids, cot_positions
        )

        # Update cached tokens
        for seq in seqs:
            seq.num_cached_tokens = len(seq)

        # Extract last hidden state per sequence from CoT
        cot_counts = [len(cot_positions[i]) for i in range(len(seqs))]
        last_indices = []
        offset = 0
        for count in cot_counts:
            last_indices.append(offset + count - 1)
            offset += count
        last_indices = torch.tensor(last_indices, dtype=torch.int64, device=cot_ntp_hidden_states.device)
        last_ntp_hidden = cot_ntp_hidden_states[last_indices].contiguous()

        # Compute NTP logits and sample
        ntp_logits = self.model.compute_logits(last_ntp_hidden)
        ntp_tokens = self.sampler(ntp_logits, temperatures)

        # MTP decode for next prediction
        mtp_positions = self.prepare_soft_mtp_prefill_decode(seqs)
        last_multi_embeds = cot_multi_input_embeds[last_indices].contiguous()
        mtp_hidden_states = self.model.mtp_decode(last_multi_embeds, last_ntp_hidden, ntp_tokens, mtp_positions)
        mtp_logits = self.model.compute_logits(mtp_hidden_states, MTP_MODULE_CONTEXT_KEY)

        if self.rank == 0:
            mtp_tokens = self._sample_mtp_logits(seqs, mtp_logits, ntp_logits, mtp_temperatures)
            return torch.stack([ntp_tokens, mtp_tokens], dim=1).tolist()
        return None

    @torch.inference_mode()
    def _restore_generation_phase(
        self,
        seqs: list[Sequence],
        temperatures: torch.Tensor,
    ) -> list[int] | None:
        """Restore KV cache for sequences in generation phase (prompt + CoT + answer).
        
        Generates next single token for answer generation (no MTP).
        """
        # Extract and restore prompt + CoT
        prompt_tokens, prompt_positions, cot_multi_ids, cot_positions = self._extract_restore_data(seqs)
        self._restore_prompt_and_cot(seqs, prompt_tokens, prompt_positions, cot_multi_ids, cot_positions)

        # Extract answer data
        ans_tokens_per_seq = []
        ans_positions_per_seq = []
        for seq in seqs:
            prompt_len = seq.num_prompt_tokens
            num_cot = seq.num_cot_tokens
            num_ans = seq.num_ans_tokens
            ans_start = prompt_len + num_cot
            ans_tokens_per_seq.append(seq.token_ids[ans_start:ans_start + num_ans])
            ans_positions_per_seq.append(list(range(ans_start, ans_start + num_ans)))

        # Step 3: Prefill answer section
        all_ans_tokens = [t for tokens in ans_tokens_per_seq for t in tokens]
        prompt_lens = [seq.num_prompt_tokens for seq in seqs]
        cot_lens = [seq.num_cot_tokens for seq in seqs]
        ans_ctx_lens = [prompt_lens[i] + cot_lens[i] + len(ans_tokens_per_seq[i]) for i in range(len(seqs))]
        positions_t, _ = self._set_prefill_context_for_section(seqs, ans_positions_per_seq, ans_ctx_lens, DEFAULT_CONTEXT_KEY)

        input_ids = torch.tensor(all_ans_tokens, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        ans_ntp_hidden_states = self.model.ntp_prefill(input_ids, positions_t)
        reset_context()

        # Update cached tokens
        for seq in seqs:
            seq.num_cached_tokens = len(seq)

        # Extract last hidden state per sequence from answer
        ans_counts = [len(ans_positions_per_seq[i]) for i in range(len(seqs))]
        last_indices = []
        offset = 0
        for count in ans_counts:
            last_indices.append(offset + count - 1)
            offset += count
        last_indices = torch.tensor(last_indices, dtype=torch.int64, device=ans_ntp_hidden_states.device)
        last_ntp_hidden = ans_ntp_hidden_states[last_indices].contiguous()

        ntp_logits = self.model.compute_logits(last_ntp_hidden)
        token_ids = self.sampler(ntp_logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids


    @torch.inference_mode()
    def run_model(self, seqs: list[Sequence], is_prefill: bool):
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        if is_prefill or self.enforce_eager or input_ids.size(0) > BATCH_SIZE_LIMIT:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context(DEFAULT_CONTEXT_KEY)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_model_mtp_prefill(self, seqs: list[Sequence], temperatures: torch.Tensor, mtp_temperatures: torch.Tensor) -> list[list[int]]:
        input_ids, positions = self.prepare_prefill(seqs)
        ntp_hidden_states = self.model.ntp_prefill(input_ids, positions)
        self.model.mtp_prefill(input_ids, positions)
        ntp_logits = self.model.compute_logits(ntp_hidden_states)

        # get the hidden states for the last token of each sequence
        # TODO: this is technically redundant, since compute_logits already does this select by index. But no good way to extract that output
        context = get_context(DEFAULT_CONTEXT_KEY)
        last_indices = context.cu_seqlens_q[1:] - 1
        ntp_hidden_states = ntp_hidden_states[last_indices].contiguous()

        # TODO: sync across all ranks
        ntp_tokens = self.sampler(ntp_logits, temperatures)

        reset_context()
        mtp_positions = self.prepare_soft_mtp_prefill_decode(seqs)
        # In this special case of the first MTP token, we don't have a second input token.
        # Technically we should pass in COT_PAD_TOKEN_ID, and then mask it out with 0.0, so
        # instead we just pass in zeros. (And note that the first tokens in multi_input_embeds is not used.)
        multi_input_embeds = torch.zeros(ntp_tokens.size(0), self.max_soft_mtp_tokens, self.model.config.hidden_size, dtype=torch.bfloat16, device=ntp_tokens.device)
        mtp_hidden_states = self.model.mtp_decode(multi_input_embeds, ntp_hidden_states, ntp_tokens, mtp_positions)
        mtp_logits = self.model.compute_logits(mtp_hidden_states, MTP_MODULE_CONTEXT_KEY)

        if self.rank == 0:
            mtp_tokens = self._sample_mtp_logits(seqs, mtp_logits, ntp_logits, mtp_temperatures)
            return torch.stack([ntp_tokens, mtp_tokens], dim=1).tolist()
        else:
            return None

    def _run_model_mtp_decode(
        self,
        mtp_input_ids: torch.Tensor,
        positions: torch.Tensor,
        mtp_positions: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ntp_hidden_states, multi_input_embeds = self.model.ntp_decode(mtp_input_ids, positions)
        ntp_logits = self.model.compute_logits(ntp_hidden_states)
        # TODO: sync across all ranks
        ntp_tokens = self.sampler(ntp_logits, temperatures)
        mtp_hidden_states = self.model.mtp_decode(multi_input_embeds, ntp_hidden_states, ntp_tokens, mtp_positions)
        return ntp_tokens, ntp_logits, mtp_hidden_states

    @torch.inference_mode()
    def run_model_mtp_decode(self, seqs: list[Sequence], temperatures: torch.Tensor, mtp_temperatures: torch.Tensor) -> list[list[int]] | None:
        mtp_input_ids, positions, mtp_positions = self.prepare_soft_mtp_decode(seqs)
        if self.enforce_eager or mtp_input_ids.size(0) > BATCH_SIZE_LIMIT:
            ntp_tokens, ntp_logits, mtp_hidden_states = self._run_model_mtp_decode(mtp_input_ids, positions, mtp_positions, temperatures)
        else:
            bs = mtp_input_ids.size(0)
            context = get_context(DEFAULT_CONTEXT_KEY)
            mtp_context = get_context(MTP_MODULE_CONTEXT_KEY)
            graph = self.mtp_graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.mtp_graph_vars
            graph_vars["multi_input_ids"][:bs] = mtp_input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["mtp_positions"][:bs] = mtp_positions
            graph_vars["temperatures"][:bs] = temperatures
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["mtp_slot_mapping"].fill_(-1)
            graph_vars["mtp_slot_mapping"][:bs] = mtp_context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["mtp_context_lens"].zero_()
            graph_vars["mtp_context_lens"][:bs] = mtp_context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            ntp_tokens = graph_vars["ntp_tokens"][:bs]
            ntp_logits = graph_vars["ntp_logits"][:bs]
            mtp_hidden_states = graph_vars["mtp_outputs"][:bs]

        mtp_logits = self.model.compute_logits(mtp_hidden_states, MTP_MODULE_CONTEXT_KEY)
        if self.rank == 0:
            mtp_tokens = self._sample_mtp_logits(seqs, mtp_logits, ntp_logits, mtp_temperatures)
            return torch.stack([ntp_tokens, mtp_tokens], dim=1).tolist()
        else:
            return None

    def _sample_mtp_logits(self, seqs: list[Sequence], mtp_logits: torch.Tensor, ntp_logits: torch.Tensor, mtp_temperatures: torch.Tensor) -> torch.Tensor:
        mtp_tokens = None
        if self.rank == 0:
            mtp_tokens = self.sampler(mtp_logits, mtp_temperatures)
            ignore_mtp_tokens = torch.zeros_like(mtp_tokens, dtype=torch.bool).cuda(non_blocking=True)
            if self.adapt_decode_type == "threshold":
                ntp_thresholds = torch.tensor(
                    [
                        seq.soft_mtp_params.adaptive_threshold[0]
                            if (seq.soft_mtp_params is not None and seq.soft_mtp_params.adaptive_threshold is not None)
                            else 0
                        for seq in seqs
                    ],
                    dtype=torch.float32,
                ).cuda(non_blocking=True)
                mtp_thresholds = torch.tensor(
                    [
                        seq.soft_mtp_params.adaptive_threshold[1]
                            if (seq.soft_mtp_params is not None and seq.soft_mtp_params.adaptive_threshold is not None)
                            else 0
                        for seq in seqs
                    ],
                    dtype=torch.float32,
                ).cuda(non_blocking=True)
                ignore_mtp_tokens = self.adapt_decode_policy(ntp_logits, mtp_logits, ntp_thresholds, mtp_thresholds)
            mtp_tokens[ignore_mtp_tokens] = sequence.COT_PAD_TOKEN_ID
        return mtp_tokens

    def run(self, seqs: list[Sequence], phase: RunPhase) -> list[int] | list[list[int]]:
        """Run model inference based on the specified phase.
        
        Args:
            seqs: Sequences to process
            phase: The run phase determining what operation to perform
            
        Returns:
            For reasoning phases (PREFILL_MTP, RESTORE_REASONING, DECODE_MTP): list[list[int]] of [ntp, mtp] pairs
            For generation phases (PREFILL, RESTORE_GENERATION, DECODE): list[int] of single tokens
        """
        if phase in (RunPhase.PREFILL, RunPhase.DECODE):
            is_prefill = (phase == RunPhase.PREFILL)
            if self.rank == 0:
                temperatures, _ = self.prepare_sample(seqs)
                logits = self.run_model(seqs, is_prefill=is_prefill)
                token_ids = self.sampler(logits, temperatures).tolist()
            else:
                token_ids = None
        else:
            temperatures, mtp_temperatures = self.prepare_sample(seqs)
            if phase == RunPhase.PREFILL_MTP:
                token_ids = self.run_model_mtp_prefill(seqs, temperatures, mtp_temperatures)
            elif phase == RunPhase.DECODE_MTP:
                token_ids = self.run_model_mtp_decode(seqs, temperatures, mtp_temperatures)
            elif phase == RunPhase.RESTORE_REASONING:
                token_ids = self._restore_reasoning_phase(seqs, temperatures, mtp_temperatures)
            elif phase == RunPhase.RESTORE_GENERATION:
                token_ids = self._restore_generation_phase(seqs, temperatures)
            else:
                raise ValueError(f"Unknown phase: {phase}")
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        max_soft_mtp_tokens = config.max_soft_mtp_tokens
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, BATCH_SIZE_LIMIT)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.mtp_graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                key=DEFAULT_CONTEXT_KEY,
                is_prefill=False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model.forward(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model.forward(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        if self.soft_mtp_enabled:
            multi_input_ids = torch.zeros(max_bs, max_soft_mtp_tokens, dtype=torch.int64)
            mtp_positions = torch.zeros(max_bs, dtype=torch.int64)
            mtp_slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
            mtp_context_lens = torch.zeros(max_bs, dtype=torch.int32)
            temperatures = torch.zeros(max_bs, dtype=torch.float32)
            ntp_tokens = torch.zeros(max_bs, dtype=torch.int64)
            ntp_logits = torch.zeros(max_bs, hf_config.vocab_size, dtype=torch.float32)
            mtp_outputs = torch.zeros(max_bs, hf_config.hidden_size)
            for bs in reversed(self.graph_bs):
                graph = torch.cuda.CUDAGraph()
                set_context(DEFAULT_CONTEXT_KEY, False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
                set_context(MTP_MODULE_CONTEXT_KEY, False, slot_mapping=mtp_slot_mapping[:bs], context_lens=mtp_context_lens[:bs], block_tables=block_tables[:bs])
                ntp_tokens[:bs], ntp_logits[:bs], mtp_outputs[:bs] = self._run_model_mtp_decode(multi_input_ids[:bs], positions[:bs], mtp_positions[:bs], temperatures[:bs])    # warmup
                # use the same graph pool as the ntp graph
                with torch.cuda.graph(graph, self.graph_pool):
                    ntp_tokens[:bs], ntp_logits[:bs], mtp_outputs[:bs] = self._run_model_mtp_decode(multi_input_ids[:bs], positions[:bs], mtp_positions[:bs], temperatures[:bs])    # capture
                self.mtp_graphs[bs] = graph
                torch.cuda.synchronize()
                reset_context()

            self.mtp_graph_vars = dict(
                multi_input_ids=multi_input_ids,
                positions=positions,
                mtp_positions=mtp_positions,
                temperatures=temperatures,
                slot_mapping=slot_mapping,
                mtp_slot_mapping=mtp_slot_mapping,
                context_lens=context_lens,
                mtp_context_lens=mtp_context_lens,
                block_tables=block_tables,
                ntp_tokens=ntp_tokens,
                ntp_logits=ntp_logits,
                mtp_outputs=mtp_outputs,
            )
