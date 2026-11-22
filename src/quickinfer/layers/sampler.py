import torch
from torch import nn


class Sampler(nn.Module):

    @staticmethod
    def probabilities(logits: torch.Tensor, temperatures: torch.Tensor):
        return torch.softmax(logits.float() / temperatures.unsqueeze(1), dim=-1)

    @staticmethod
    def sample(probs: torch.Tensor):
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        return (probs / noise).argmax(dim=-1)

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        return self.sample(self.probabilities(logits, temperatures))
