from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import os

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass
class SchedulerStats:
    """Track runtime statistics for adaptive scheduling decisions."""
    total_completed_sequences: int = 0
    total_generated_tokens: int = 0
    # Track recent generation lengths for more responsive adaptation
    recent_gen_lengths: list = field(default_factory=list)
    max_recent_samples: int = 100
    
    @property
    def avg_generation_length(self) -> float:
        """Return average generation length, with conservative default."""
        if self.total_completed_sequences == 0:
            return 512.0  # conservative default
        return self.total_generated_tokens / self.total_completed_sequences
    
    @property
    def recent_avg_generation_length(self) -> float:
        """Return average of recent completions for faster adaptation."""
        if not self.recent_gen_lengths:
            return self.avg_generation_length
        return sum(self.recent_gen_lengths) / len(self.recent_gen_lengths)
    
    def record_completion(self, seq: Sequence):
        """Record a completed sequence's statistics."""
        self.total_completed_sequences += 1
        gen_len = seq.num_completion_tokens
        self.total_generated_tokens += gen_len
        self.recent_gen_lengths.append(gen_len)
        if len(self.recent_gen_lengths) > self.max_recent_samples:
            self.recent_gen_lengths.pop(0)


# Simple CSV logger for scheduler debugging
_metrics_file = None
def _log_metrics(msg: str):
    global _metrics_file
    if _metrics_file is None:
        fname = f"scheduler_metrics_{os.getpid()}_{time.strftime('%H%M%S')}.csv"
        _metrics_file = open(fname, "w")
        _metrics_file.write("time,event,batch,wait,reason,gen,free_blk,preempts\n")
    _metrics_file.write(msg + "\n")
    _metrics_file.flush()


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
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, 
            config.kvcache_block_size, 
            max_soft_mtp_tokens=config.max_soft_mtp_tokens,
            reserved_ratio=config.reserved_block_ratio
        )
        self.waiting: deque[Sequence] = deque()
        self.running_reasoning: deque[Sequence] = deque()
        self.running_generation: deque[Sequence] = deque()
        self.debug = config.debug
        self._preempts = 0  # total preemption count
        
        # Adaptive scheduling parameters
        self.stats = SchedulerStats()
        self.min_decode_batch_size = max(8, int(config.max_num_seqs * config.min_decode_batch_ratio))
        self._min_free_blocks = int(config.num_kvcache_blocks * config.proactive_preempt_threshold)

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

    def _should_proactively_preempt(self) -> bool:
        """Check if we should preempt to maintain memory headroom."""
        if self.block_manager.free_blocks > self._min_free_blocks:
            return False
        # Preempt if waiting queue is backing up and we're low on memory
        num_running = len(self.running_reasoning) + len(self.running_generation)
        return len(self.waiting) > num_running

    def _proactive_preempt(self, count: int = 1):
        """Preempt sequences proactively to free blocks.
        
        Prefer preempting from reasoning (longer sequences, more blocks to free).
        """
        for _ in range(count):
            if self.running_reasoning:
                self.preempt(self.running_reasoning.pop())
            elif self.running_generation:
                self.preempt(self.running_generation.pop())
            else:
                break  # nothing to preempt

    def _should_prioritize_generation(self) -> bool:
        """Decide whether to prioritize generation or reasoning.
        
        Returns True if generation should be prioritized, False if reasoning
        should be prioritized instead (e.g., when generation batch would be tiny
        but reasoning batch would be substantial).
        """
        gen_count = len(self.running_generation)
        reason_count = len(self.running_reasoning)
        
        # Always prioritize if generation batch would be substantial
        if gen_count >= self.min_decode_batch_size:
            return True
        
        # If generation batch is tiny but reasoning is large, defer generation
        if gen_count < self.min_decode_batch_size and reason_count > gen_count * 4:
            return False
        
        # Default: prioritize generation (frees resources faster)
        return True

    def _num_running(self) -> int:
        """Return total number of running sequences."""
        return len(self.running_reasoning) + len(self.running_generation)

    def schedule(self) -> tuple[list[Sequence], RunPhase]:
        # Update block reservation based on recent statistics
        self.block_manager.update_reservation(
            self.stats.recent_avg_generation_length, 
            self._num_running()
        )
        
        # Proactive preemption: if memory is getting tight and waiting queue is backing up,
        # preempt some sequences now to avoid thrashing later
        if self._should_proactively_preempt():
            self._proactive_preempt(count=2)
        
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # For soft-MTP restored sequences, track the phase of the batch to avoid mixing
        batch_phase: RunPhase | None = None
        block_reason = None

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                block_reason = "tokens"
                break
            # Use available_blocks to respect reservation for decode headroom
            if not self.block_manager.can_allocate(seq) or self.block_manager.available_blocks < seq.num_blocks:
                block_reason = "blocks"
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
                block_reason = "phase"
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
            self._log(batch_phase.name, len(scheduled_seqs), block_reason)
            return scheduled_seqs, batch_phase

        # Decide whether to prioritize generation or reasoning based on batch sizes
        prioritize_generation = self._should_prioritize_generation()
        
        if prioritize_generation:
            # Try generation first
            scheduled_seqs = self._schedule_decode_generation(num_seqs)
            if scheduled_seqs:
                # Check minimum batch size - if too small and reasoning has more, try reasoning instead
                if len(scheduled_seqs) < self.min_decode_batch_size and \
                   len(self.running_reasoning) >= self.min_decode_batch_size:
                    # Put generation sequences back
                    self.running_generation.extendleft(reversed(scheduled_seqs))
                    scheduled_seqs = []
                else:
                    self.running_generation.extendleft(reversed(scheduled_seqs))
                    self._log("DECODE", len(scheduled_seqs), None)
                    return scheduled_seqs, RunPhase.DECODE
            
            # Try reasoning if generation didn't produce a batch
            if not scheduled_seqs:
                scheduled_seqs = self._schedule_decode_reasoning(num_seqs)
                if scheduled_seqs:
                    self.running_reasoning.extendleft(reversed(scheduled_seqs))
                    self._log("DECODE_MTP", len(scheduled_seqs), None)
                    return scheduled_seqs, RunPhase.DECODE_MTP
        else:
            # Prioritize reasoning first (generation batch would be too small)
            scheduled_seqs = self._schedule_decode_reasoning(num_seqs)
            if scheduled_seqs:
                self.running_reasoning.extendleft(reversed(scheduled_seqs))
                self._log("DECODE_MTP", len(scheduled_seqs), None)
                return scheduled_seqs, RunPhase.DECODE_MTP
            
            # Fall back to generation if reasoning didn't work
            scheduled_seqs = self._schedule_decode_generation(num_seqs)
            if scheduled_seqs:
                self.running_generation.extendleft(reversed(scheduled_seqs))
                self._log("DECODE", len(scheduled_seqs), None)
                return scheduled_seqs, RunPhase.DECODE

        # If we get here, all running sequences were preempted and are now in waiting.
        # Recursively schedule to pick them up via prefill.
        assert self.waiting, "No sequences to schedule but queues are empty"
        return self.schedule()

    def _schedule_decode_generation(self, num_seqs: int) -> list[Sequence]:
        """Schedule generation (NTP) decode batch."""
        scheduled_seqs = []
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
        return scheduled_seqs

    def _schedule_decode_reasoning(self, num_seqs: int) -> list[Sequence]:
        """Schedule reasoning (MTP) decode batch."""
        scheduled_seqs = []
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
        return scheduled_seqs
    
    def _log(self, phase: str, batch_size: int, block_reason: str | None):
        if not self.debug:
            return
        t = time.strftime("%H:%M:%S")
        wait = len(self.waiting)
        gen = len(self.running_generation) + len(self.running_reasoning)
        free = self.block_manager.free_blocks
        _log_metrics(f"{t},{phase},{batch_size},{wait},{block_reason or ''},{gen},{free},{self._preempts}")

    def preempt(self, seq: Sequence):
        self._preempts += 1
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
                self.stats.record_completion(seq)  # Record completion for adaptive scheduling
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
                    self.stats.record_completion(seq)  # Record completion for adaptive scheduling
                    self.block_manager.deallocate(seq)
                    self.running_reasoning.remove(seq)
                else:
                    self.running_reasoning.remove(seq)
                    self.running_generation.append(seq)
                continue

            if seq.eot_from_mtp_module:
                seq.apply_eot_from_mtp_module()
                token_ids = (self.eot, -1) # swap out token_ids, so we transition to answer phase
            else:
                if (seq.soft_mtp_params is not None and seq.soft_mtp_params.cot_max_tokens is not None) \
                    and seq.num_cot_tokens == seq.soft_mtp_params.cot_max_tokens:
                    # force transition to answer by overriding with EOT token
                    token_ids = (self.eot, token_ids[1])
                seq.append_soft_mtp_tokens(token_ids)

            if seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.stats.record_completion(seq)  # Record completion for adaptive scheduling
                self.block_manager.deallocate(seq)
                self.running_reasoning.remove(seq)
            elif token_ids[0] == self.eot:
                # NTP predicted EOT - immediate transition to generation
                # (Don't check MTP token here; if only MTP predicted EOT, 
                # eot_from_mtp_module flag handles the delayed transition)
                self.running_reasoning.remove(seq)
                self.running_generation.append(seq)
