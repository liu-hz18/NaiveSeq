import torch
import torch.nn as nn
import torch.nn.functional as F


def get_residual_block(
    embedding_dim: int,
    residual_dropout: float = 0.1,
    residual_policy: str = 'residual'
) -> nn.Module:
    if residual_policy not in ['none', 'residual', 'highway']:
        raise ValueError(
            'residual_policy must be `none` or `residual` or `highway`')
    if not hasattr(get_residual_block, '_residual_map'):
        get_residual_block._residual_map = {
            'none': nn.Identity,
            'residual': ResidualBlock,
            'highway': HighWayBlock,
        }
    return get_residual_block._residual_map[residual_policy](embedding_dim, residual_dropout)


class ResidualBlock(nn.Module):

    def __init__(self, embedding_dim: int, residual_dropout: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.residual_dropout = residual_dropout

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        return before + F.dropout(after, p=residual_dropout, training=self.training, inplace=False)


class HighWayBlock(nn.Module):

    def __init__(self, embedding_dim: int, residual_dropout: float = 0.1):
        super(HighWayBlock, self).__init__()
        self.residual_dropout = residual_dropout
        self.gate = nn.Linear(embedding_dim, 1)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(before))
        after_dropout = F.dropout(after, p=residual_dropout,
                                  training=self.training, inplace=False)
        return before * g + after_dropout * (1. - g)
