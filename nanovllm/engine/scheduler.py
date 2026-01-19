from collections import deque
from enum import Enum, auto

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class RunPhase(Enum):
    """Describes what operation the model runner should perform."""
    # Prefill phases (processing waiting sequences)
    PREFILL = auto()              # Standard NTP prefill (non-MTP)
    PREFILL_MTP = auto()          # Fresh soft-MTP prefill
    RESTORE_REASONING = auto()    # Restore preempted sequence in reasoning phase (CoT)
    RESTORE_GENERATION = auto()   # Restore preempted sequence in generation phase (answer)
    
    # Decode phases (continuing running sequences)
    DECODE = auto()               # Standard NTP decode (generation)
    DECODE_MTP = auto()           # Soft-MTP decode (reasoning)

    def is_prefill(self) -> bool:
        return self in (RunPhase.PREFILL, RunPhase.PREFILL_MTP, RunPhase.RESTORE_REASONING, RunPhase.RESTORE_GENERATION)

    def is_reasoning(self) -> bool:
        """Returns True if this phase produces [ntp, mtp] token pairs."""
        return self in (RunPhase.PREFILL_MTP, RunPhase.RESTORE_REASONING, RunPhase.DECODE_MTP)


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

    def _is_restored_in_generation_phase(self, seq: Sequence) -> bool:
        """Check if a restored sequence was in generation phase.
        
        Note: We only consider sequences with actual answer tokens as being in generation phase.
        Sequences that just emitted EOT (last_token == eot, num_ans_tokens == 0) are handled
        via reasoning restore path, with special postprocessing to transition them to generation.
        """
        return seq.num_ans_tokens > 0

    def _is_restored_sequence(self, seq: Sequence) -> bool:
        """Check if a sequence is being restored (was preempted after generating tokens)."""
        return seq.num_cot_tokens > 0 or seq.num_ans_tokens > 0

    def schedule(self) -> tuple[list[Sequence], RunPhase]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # For soft-MTP restored sequences, track the phase of the batch to avoid mixing
        batch_phase: RunPhase | None = None

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break

            # Determine the prefill phase for this sequence
            if self.soft_mtp_enabled:
                is_restored = self._is_restored_sequence(seq)
                if is_restored:
                    seq_phase = RunPhase.RESTORE_GENERATION if self._is_restored_in_generation_phase(seq) else RunPhase.RESTORE_REASONING
                else:
                    seq_phase = RunPhase.PREFILL_MTP
            else:
                seq_phase = RunPhase.PREFILL

            # Ensure we don't mix different phases in the same batch
            if batch_phase is None:
                batch_phase = seq_phase
            elif batch_phase != seq_phase:
                break

            num_seqs += 1
            self.block_manager.prefill_allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            if self.soft_mtp_enabled:
                # For restored sequences, check if they were in generation phase
                if self._is_restored_in_generation_phase(seq):
                    self.running_generation.append(seq)
                else:
                    self.running_reasoning.append(seq)
            else:
                self.running_generation.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, batch_phase

        # prioritize generation over reasoning, since it frees up resources
        # TODO: add heuristic to decide which to prioritize. E.g. prioritize generation if generation batch size is larger than B.
        # generation
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
        if scheduled_seqs:
            self.running_generation.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, RunPhase.DECODE

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
        if scheduled_seqs:
            self.running_reasoning.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, RunPhase.DECODE_MTP

        # If we get here, all running sequences were preempted and are now in waiting.
        # Recursively schedule to pick them up via prefill.
        assert self.waiting, "No sequences to schedule but queues are empty"
        return self.schedule()

    def preempt(self, seq: Sequence):
        # pause this sequence to free up resources
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess_ntp(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if seq.soft_mtp_params is not None and seq.soft_mtp_params.ans_max_tokens is not None:
                is_limit = seq.num_ans_tokens == seq.soft_mtp_params.ans_max_tokens
            else:
                is_limit = seq.num_completion_tokens == seq.max_tokens
            if (not seq.ignore_eos and token_id == self.eos) or is_limit:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running_generation.remove(seq)

    def postprocess_reasoning(self, seqs: list[Sequence], mtp_ids: list[list[int]]):
        for seq, token_ids in zip(seqs, mtp_ids):
            # Special case: restored sequence that already emitted EOT but has no answer tokens yet.
            # This was handled via reasoning restore path, so we get [ntp, mtp] tokens,
            # but we should only use the NTP token as the first answer token and transition.
            # NOTE: During normal decoding, this will never be triggered since we would go into
            # process_ntp instead.
            if seq.last_token == self.eot:
                seq.append_token(token_ids[0])  # Append as regular answer token
                if seq.num_completion_tokens == seq.max_tokens or token_ids[0] == self.eos:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running_reasoning.remove(seq)
                else:
                    self.running_reasoning.remove(seq)
                    self.running_generation.append(seq)
                continue

            if seq.eot_from_mtp_module:
                seq.apply_eot_from_mtp_module()
            else:
                if (seq.soft_mtp_params is not None and seq.soft_mtp_params.cot_max_tokens is not None) \
                    and seq.num_cot_tokens == seq.soft_mtp_params.cot_max_tokens:
                    # force transition to answer by overriding with EOT token
                    token_ids = (self.eot, token_ids[1])
                seq.append_soft_mtp_tokens(token_ids)

            if seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running_reasoning.remove(seq)
            elif token_ids[0] == self.eot:
                # NTP predicted EOT - immediate transition to generation
                # (Don't check MTP token here; if only MTP predicted EOT, 
                # eot_from_mtp_module flag handles the delayed transition)
                self.running_reasoning.remove(seq)
                self.running_generation.append(seq)
