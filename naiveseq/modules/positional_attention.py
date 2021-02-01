import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
import fairseq

from fairseq import utils
from fairseq.module import (
    MultiheadAttention as _MultiheadAttention,
    SinusoidalPositionalEmbedding
)


class PositionalAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, bias=True, max_seq_length=128, q_noise=0.0, qn_block_size=8):
        super(PositionalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.onnx_trace = False
        self.padding_idx = 0
        self.multihead_attn = _MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=q_noise,
            qn_block_size=qn_block_size
        )
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            num_embeddings=max_seq_length+1,
            embedding_dim=embed_dim,
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor],
        attn_mask: Optional[torch.BoolTensor],
        need_weights: bool=True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Time x Batch x Embed

        Args:
            key_padding_mask (BoolTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (BoolTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            
        Return: 
            attn: (Time x Batch x Embed)
            attn_weights: (Batch x Time x Time)
        """
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[1], bspair[0]
        max_pos = 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                num_embeddings=max_pos,
                embedding_dim=self.embed_dim
            )
        mask = torch.ones(bsz, seq_len).to(x)
        positions = torch.cumsum(mask, dim=1).type_as(mask).long() + self.padding_idx
        self.weights = self.weights.to(self._float_tensor)
        sin_pos_encoding = (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )
        return self.multihead_attn(
            query=sin_pos_encoding,
            key=sin_pos_encoding.clone(),
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )
