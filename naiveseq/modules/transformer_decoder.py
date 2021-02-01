import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import PositionalEmbedding
from fairseq.modules import quant_noise as apply_quant_noise_
from .transformer_layer import TransformerDecoderLayer
from ..utils import make_attn_mask

class TransformerDecoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        token_embedding: Optional[nn.Embedding] = None,
        padding_idx: int = 0,
        num_layers: int = 6,
        learned_pos_embedding: bool = True,
        embed_dropout: float = 0.1,
        num_heads: int = None,
        attn_dropout: float = 0.0,
        attn_bias: bool = True,
        self_attn_policy: str = 'none',
        ffn_embed_dim: int = None,
        ffn_dropout: float = 0.1,
        ffn_activation_fn: str = 'relu',
        residual_policy: str = 'residual',
        residual_dropout: float = 0.1,
        max_seq_length: int = 128,
        post_norm: bool = True,
        apply_pos_attn: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8
    ):
        super(TransformerDecoder, self).__init__()
        self.self_attn_policy = self_attn_policy
        self.padding_idx = padding_idx
        self.embed_dropout = embed_dropout
        self.embed_scale = math.sqrt(embed_dim)
        if token_embedding is None:
            self.token_embedding = nn.Embedding(
                vocab_size, embed_dim, padding_idx)
        else:
            self.token_embedding = token_embedding
            assert token_embedding.weight.size(
                0) == vocab_size and token_embedding.weight.size(1) == embed_dim
        self.postional_embedding = PositionalEmbedding(
            max_seq_length,
            embed_dim,
            padding_idx=padding_idx,
            learned=learned_pos_embedding,
        )
        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None
        self.decoder_stack = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                attn_bias=attn_bias,
                ffn_embed_dim=ffn_embed_dim,
                ffn_dropout=ffn_dropout,
                ffn_activation_fn=ffn_activation_fn,
                residual_policy=residual_policy,
                residual_dropout=residual_dropout,
                max_seq_length=max_seq_length,
                post_norm=post_norm,
                apply_pos_attn=apply_pos_attn,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            ) for _ in range(num_layers)
        ])

    def forward(
        self,
        tokens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input shape: B x L
        # compute padding mask. This is needed for multi-head attention
        decoder_padding_mask = tokens.eq(self.padding_idx)
        # embedding
        x = self.token_embedding(tokens) * self.embed_scale
        x = x + self.postional_embedding(tokens)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        x = F.dropout(x, p=self.embed_dropout,
                      training=self.training, inplace=False)
        # account for padding while computing the representation
        x = x * (1 - decoder_padding_mask.unsqueeze(-1).type_as(x))
        # self attn mask(2D) L x L
        B, L = tokens.shape
        self_attn_mask = make_attn_mask(B, L, mask_policy=self.self_attn_policy)
        # decode
        # B x T x C -> T x B x C
        inner_states = []
        x = x.transpose(0, 1)
        for layer in self.decoder_stack:
            x, _ = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                self_attn_mask=self_attn_mask
            )
            inner_states.append(x)
        # return: x shape: T x B x C
        return x, inner_states
