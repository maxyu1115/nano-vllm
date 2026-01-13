import torch
from nanovllm.utils.context import DEFAULT_CONTEXT_KEY

class SoftMTPInterface:    
    def compute_logits(self, hidden_states: torch.Tensor, context_key: str = DEFAULT_CONTEXT_KEY) -> torch.Tensor:
        # the context key should be passed as is to the lm_head
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def ntp_prefill(
        self,
        input_ids: torch.Tensor, # (S,)
        positions: torch.Tensor, # (S,)
    ) -> torch.Tensor:
        # returns: ntp_hidden_states
        raise NotImplementedError

    def ntp_decode(
        self,
        multi_input_ids: torch.Tensor, # (B, k)
        positions: torch.Tensor, # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # returns: (ntp_hidden_states, multi_input_embeds)
        raise NotImplementedError

    def mtp_prefill(
        self,
        mtp_input_ids: torch.Tensor, # (S+B,)
        mtp_positions: torch.Tensor, # (S+B,)
    ) -> torch.Tensor:
        # returns: mtp_hidden_states
        raise NotImplementedError

    def mtp_decode(
        self,
        multi_input_embeds: torch.Tensor, # (B, k, H)
        ntp_hidden_states: torch.Tensor, # (B, H)
        ntp_output_ids: torch.Tensor, # (B,)
        mtp_positions: torch.Tensor, # (B,)
    ) -> torch.Tensor:
        # returns: mtp_hidden_states
        raise NotImplementedError
