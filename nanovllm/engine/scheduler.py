from collections import deque
from enum import Enum, auto

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.soft_mtp_enabled = config.max_soft_mtp_tokens > 1
        # special tokens, -1 means not set
        self.eos = config.eos
        self.bot = config.bot
        self.eot = config.eot
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size, max_soft_mtp_tokens=config.max_soft_mtp_tokens)
        self.waiting: deque[Sequence] = deque()
        self.running_reasoning: deque[Sequence] = deque()
        self.running_generation: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running_reasoning and not self.running_generation

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool, bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.prefill_allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            if self.soft_mtp_enabled:
                self.running_reasoning.append(seq)
            else:
                self.running_generation.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True, self.soft_mtp_enabled

        # prioritize generation over reasoning, since it frees up resources
        # TODO: add heuristic to decide which to prioritize. E.g. prioritize generation if generation batch size is larger than B.
        # generation
        if self.running_generation:
            while self.running_generation and num_seqs < self.max_num_seqs:
                seq = self.running_generation.popleft()
                while not self.block_manager.can_append(seq):
                    if self.running_generation:
                        self.preempt(self.running_generation.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    num_seqs += 1
                    self.block_manager.may_append(seq)
                    scheduled_seqs.append(seq)
            assert scheduled_seqs
            self.running_generation.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False, False

        # reasoning
        while self.running_reasoning and num_seqs < self.max_num_seqs:
            seq = self.running_reasoning.popleft()
            while not self.block_manager.can_append(seq):
                if self.running_reasoning:
                    self.preempt(self.running_reasoning.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running_reasoning.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, True

    def preempt(self, seq: Sequence):
        if self.soft_mtp_enabled:
            # TODO: implement preemption for soft MTP
            raise NotImplementedError("Preemption not supported for soft MTP")
        # pause this sequence to free up resources
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess_ntp(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.sampling_params.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.sampling_params.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running_generation.remove(seq)

    def postprocess_reasoning(self, seqs: list[Sequence], mtp_ids: list[list[int]]):
        for seq, token_ids in zip(seqs, mtp_ids):
            seq.append_soft_mtp_tokens(token_ids)
            if seq.num_completion_tokens == seq.sampling_params.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running_reasoning.remove(seq)
            elif any(t == self.eot for t in token_ids):
                # move to generation queue (only if not finished)
                self.running_reasoning.remove(seq)
                self.running_generation.append(seq)
