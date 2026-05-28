import torch
from torch import nn


class AdaptiveDecodingThresholdPolicy(nn.Module):
    # @torch.compile
    def forward(self, ntp_logits: torch.Tensor, mtp_logits: torch.Tensor, ntp_thresholds: torch.Tensor, mtp_thresholds: torch.Tensor) -> torch.Tensor:
        ntp_logits = ntp_logits.float()
        mtp_logits = mtp_logits.float()
        ntp_top_logit, _ = ntp_logits.max(dim=-1)
        mtp_top_logit, _ = mtp_logits.max(dim=-1)
        ntp_top_p = torch.exp(ntp_top_logit - torch.logsumexp(ntp_logits, dim=-1))
        mtp_top_p = torch.exp(mtp_top_logit - torch.logsumexp(mtp_logits, dim=-1))
        ntp_only_mask = ((ntp_top_p < ntp_thresholds) | (mtp_top_p < mtp_thresholds))
        return ntp_only_mask
