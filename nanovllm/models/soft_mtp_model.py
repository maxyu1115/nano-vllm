import torch


class SoftMTPInterface:    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def mtp_prefill(
        self,
        input_ids: torch.Tensor, # (S,)
        positions: torch.Tensor, # (S,)
        mtp_input_ids: torch.Tensor, # (S+B,)
        mtp_positions: torch.Tensor, # (S+B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def mtp_forward(
        self,
        multi_input_ids: torch.Tensor, # (B,k)
        positions: torch.Tensor, # (B,)
        mtp_positions: torch.Tensor, # (B,)
        temperatures: torch.Tensor, # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # returns: (ntp_tokens, mtp_hidden_states)
        raise NotImplementedError
