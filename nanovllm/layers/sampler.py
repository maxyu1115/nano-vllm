import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Ensure logits are in floating point for the softmax / sampling math.
        logits = logits.float()

        # Greedy tokens (used when temperature == 0).
        greedy_tokens = logits.argmax(dim=-1)

        # For sampling, avoid division by zero by substituting 1.0 where temperature == 0.
        safe_temperatures = torch.where(
            temperatures == 0,
            torch.ones_like(temperatures),
            temperatures,
        )

        scaled_logits = logits.div(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(scaled_logits, dim=-1)

        # Gumbel-like sampling trick used previously, kept for non-greedy paths.
        noisy_probs = probs.div(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        )
        sampled_tokens = noisy_probs.argmax(dim=-1)

        # Select between greedy and sampled tokens in a tensor-friendly way for torch.compile.
        is_greedy = (temperatures < 1e-6)
        output_tokens = torch.where(is_greedy, greedy_tokens, sampled_tokens)
        return output_tokens
