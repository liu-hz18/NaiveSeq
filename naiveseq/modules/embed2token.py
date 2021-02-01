import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embed2Token(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        embedding_layer: Optional[nn.Embedding]=None,
        adaptive_softmax: bool = False,
        bias=True
    ):
        # [vocab_size, embed_dim]
        if embedding_layer is not None:
            self.weight = embedding_layer.weight
            assert self.weight.size(0) == vocab_size and self.weight.size(1) == embed_dim
        else:
            self.weight = nn.Parameter(torch.Tensor(vocab_size, embed_dim))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(vocab_size))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, token_embed: torch.Tensor) -> torch.Tensor:
        return F.linear(token_embed, self.weight, self.bias).transpose(0, 1)

    def get_normalized_probs(self, output_logits: torch.Tensor, log_probs: bool = False) -> torch.Tensor:
        if log_probs:
            return F.log_softmax(output_logits, dim=-1)
        else:
            return F.softmax(output_logits, dim=-1)
