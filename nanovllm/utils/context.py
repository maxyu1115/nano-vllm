from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


DEFAULT_CONTEXT_KEY = "default"

_CONTEXTS = {
    DEFAULT_CONTEXT_KEY: Context()
}

def get_context(key: str = DEFAULT_CONTEXT_KEY) -> Context:
    global _CONTEXTS
    return _CONTEXTS.get(key, Context())

def set_context(
    key: str = DEFAULT_CONTEXT_KEY,
    is_prefill=False,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    global _CONTEXTS
    _CONTEXTS[key] = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXTS
    _CONTEXTS = {
        DEFAULT_CONTEXT_KEY: Context()
    }
