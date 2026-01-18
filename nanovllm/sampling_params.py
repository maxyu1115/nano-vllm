from dataclasses import dataclass
from typing import Optional, Literal


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

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be non-negative"
        if self.soft_mtp_params is not None:
            assert self.soft_mtp_params.mtp_temperature >= 0, "mtp_temperature must be non-negative"
