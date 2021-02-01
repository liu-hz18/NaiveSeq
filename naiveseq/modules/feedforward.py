
import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq

from fairseq.modules.quant_noise import quant_noise


class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        ffn_dropout: float=0.3,
        ffn_activation_fn: str='relu',
        q_noise: float=0.0,
        qn_block_size: int=8,
    ):
        super(FeedForward, self).__init__()
        self.ffn_dropout = ffn_dropout
        self.ffn_activation_fn = fairseq.utils.get_activation_fn(ffn_activation_fn)
        self.fc1 = quant_noise(
            nn.Linear(embedding_dim, ffn_embedding_dim),
            q_noise,
            qn_block_size
        )
        self.fc2 = quant_noise(
            nn.Linear(ffn_embedding_dim, embedding_dim),
            q_noise,
            qn_block_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ffn_activation_fn(x)
        x = F.dropout(x, p=self.ffn_dropout, training=self.training, inplace=False)
        x = self.fc2(x)
        return x
