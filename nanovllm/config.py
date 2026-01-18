import os
from dataclasses import dataclass
from typing import Literal, Optional
from transformers import AutoConfig


@dataclass
class Config:
    model_path: Optional[str] = None
    dist_init_method: str = "tcp://localhost:2333"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    max_soft_mtp_tokens: int = 1 # 1 or less means not using Soft MTP, otherwise max number of tokens produced by Soft MTP module
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    bot: int = -1 # beginning of thought token
    eot: int = -1 # end of thought token
    cot_pad_token: int = -1 # padding token for cot
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    soft_mtp_adaptive_decoding_policy: Literal["threshold", "none"] = "none"

    def __post_init__(self):
        if self.model_path is not None:
            assert os.path.isdir(self.model_path)
            self.hf_config = AutoConfig.from_pretrained(self.model_path)
        else:
            assert self.hf_config is not None
        assert self.kvcache_block_size == 256, "Sequence assumes block size of 256"
        assert 1 <= self.tensor_parallel_size <= 8
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

        if self.max_soft_mtp_tokens > 1:
            assert self.bot >= 0 and self.eot >= 0
            assert self.cot_pad_token >= 0
