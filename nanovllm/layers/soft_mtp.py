import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveDecodingThresholdPolicy(nn.Module):
    @torch.compile
    def forward(self, ntp_logits: torch.Tensor, mtp_logits: torch.Tensor, ntp_thresholds: torch.Tensor, mtp_thresholds: torch.Tensor) -> torch.Tensor:
        ntp_probs = F.softmax(ntp_logits, dim=-1)
        mtp_probs = F.softmax(mtp_logits, dim=-1)
        ntp_top_p, _ = ntp_probs.max(dim=-1)
        mtp_top_p, _ = mtp_probs.max(dim=-1)
        ntp_only_mask = ((ntp_top_p < ntp_thresholds) | (mtp_top_p < mtp_thresholds))
        return ntp_only_mask
