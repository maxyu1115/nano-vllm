from dataclasses import dataclass
from typing import Optional


@dataclass
class SoftMTPSamplingParams:
    mtp_temperature: float = 1.0

    # If not none, use adaptive decoding. Tuple of (NTP threshold, MTP threshold)
    adaptive_threshold: Optional[tuple[float, float]] = None

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    soft_mtp_params: Optional[SoftMTPSamplingParams] = None
    # Separate cot/ans limits - works as an alternative to max_tokens
    # When set, the model will track tokens after BOT as cot tokens and tokens after EOT as ans tokens
    cot_max_tokens: Optional[int] = None
    ans_max_tokens: Optional[int] = None

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be non-negative"
        if self.soft_mtp_params is not None:
            assert self.soft_mtp_params.mtp_temperature >= 0, "mtp_temperature must be non-negative"
        # Validation for cot/ans limits
        if self.cot_max_tokens is not None or self.ans_max_tokens is not None:
            assert self.ans_max_tokens is not None, "ans_max_tokens must be set if cot_max_tokens is set"
            assert self.cot_max_tokens is not None, "cot_max_tokens must be set if ans_max_tokens is set"
            self.max_tokens = self.cot_max_tokens + self.ans_max_tokens
