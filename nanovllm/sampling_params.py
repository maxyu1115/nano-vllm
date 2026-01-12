from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class MTPAdaptiveDecodingConfig:
    ntp_threshold: float = 0.3
    mtp_threshold: float = 0.9

@dataclass
class SamplingParams:
    temperature: float = 1.0
    mtp_temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    mtp_adaptive_decoding_config: Optional[MTPAdaptiveDecodingConfig] = None

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be non-negative"
        assert self.mtp_temperature >= 0, "mtp_temperature must be non-negative"
