

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import LayerNorm, PositionalEmbedding
from .self_attenuation import SelfAttention
from .encdec_attenuation import EncDecAttention
from .positional_attenuation import PositionalAttention
from .feedforward import FeedForward
from .residual_block import get_residual_block


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = None,
        attn_dropout: float = 0.0,
        attn_bias: bool = True,
        ffn_embed_dim: int = None,
        ffn_dropout: float = 0.1,
        ffn_activation_fn: str = 'relu',
        residual_policy: str = 'residual',
        residual_dropout: float = 0.1,
        max_seq_length: int = 128,
        post_norm: bool = True,
        q_noise: float = 0.0,
        qn_block_size: int = 8
    ):
        super(TransformerEncoderLayer, self).__init__()
        num_heads = num_heads if num_heads is not None else embed_dim // 64
        ffn_embed_dim = ffn_embed_dim if ffn_embed_dim is not None else embed_dim * 4
        self.post_norm = post_norm
        # Encoder Self Attention
        self.self_attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            bias=attn_bias,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.attn_layernorm = LayerNorm(normalized_shape=embed_dim)
        self.attn_residual_block = get_residual_block(
            embedding_dim=embed_dim,
            residual_dropout=residual_dropout,
            residual_policy=residual_policy,
        )
        # FeedForward Network
        self.ffn = FeedForward(
            embedding_dim=embed_dim,
            ffn_embedding_dim=ffn_embed_dim,
            ffn_dropout=ffn_dropout,
            ffn_activation_fn=ffn_activation_fn,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.ffn_layernorm = LayerNorm(normalized_shape=embed_dim)
        self.ffn_residual_block = get_residual_block(
            embedding_dim=embed_dim,
            residual_dropout=residual_dropout,
            residual_policy=residual_policy,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor]=None,
        attn_mask: Optional[torch.BoolTensor]=None,
        need_weights: bool=False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Input Shape: L x B x E
        # Encoder Self Attention
        before = x
        if not self.post_norm:
            x = self.attn_layernorm(x)
        x, attn = self.self_attn(
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )
        x = self.attn_residual_block(
            before=before,
            after=x
        )
        if self.post_norm:
            x = self.attn_layernorm(x)
        # FeedForward Network
        before = x
        if not self.post_norm:
            x = self.ffn_layernorm(x)
        x = self.ffn(x)
        x = self.ffn_residual_block(
            before=before,
            after=x
        )
        if self.post_norm:
            x = self.ffn_layernorm(x)
        return x, attn


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = None,
        attn_dropout: float = 0.0,
        attn_bias: bool = True,
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
        super(TransformerDecoderLayer, self).__init__()
        num_heads = num_heads if num_heads is not None else embed_dim // 64
        ffn_embed_dim = ffn_embed_dim if ffn_embed_dim is not None else embed_dim * 4
        self.post_norm = post_norm
        # Decoder Self Attention
        self.self_attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            bias=attn_bias,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.self_attn_layernorm = LayerNorm(normalized_shape=embed_dim)
        self.self_attn_residual_block = get_residual_block(
            embedding_dim=embed_dim,
            residual_dropout=residual_dropout,
            residual_policy=residual_policy,
        )
        # Decoder Positional Attention
        if self.apply_pos_attn:
            self.pos_attn = PositionalAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                bias=attn_bias,
                max_seq_length=max_seq_length,
                q_noise=q_noise,
                qn_block_size=qn_block_size
            )
            self.pos_attn_layernorm = LayerNorm(normalized_shape=embed_dim)
            self.pos_attn_residual_block = get_residual_block(
                embedding_dim=embed_dim,
                residual_dropout=residual_dropout,
                residual_policy=residual_policy,
            )
        # Encoder-Decoder Attention
        self.encdec_attn = EncDecAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            bias=attn_bias,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.encdec_attn_layernorm = LayerNorm(normalized_shape=embed_dim)
        self.encdec_attn_residual_block = get_residual_block(
            embedding_dim=embed_dim,
            residual_dropout=residual_dropout,
            residual_policy=residual_policy,
        )
        # FeedForward Network
        self.ffn = FeedForward(
            embedding_dim=embed_dim,
            ffn_embedding_dim=ffn_embed_dim,
            ffn_dropout=ffn_dropout,
            ffn_activation_fn=ffn_activation_fn,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.ffn_layernorm = LayerNorm(normalized_shape=embed_dim)
        self.ffn_residual_block = get_residual_block(
            embedding_dim=embed_dim,
            residual_dropout=residual_dropout,
            residual_policy=residual_policy,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.BoolTensor] = None,
        decoder_padding_mask: Optional[torch.BoolTensor] = None,
        self_attn_mask: Optional[torch.BoolTensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Input Shape: L x B x E
        # Decoder Self Attention
        before = x
        if not self.post_norm:
            x = self.self_attn_layernorm(x)
        x, attn = self.self_attn(
            x,
            key_padding_mask=decoder_padding_mask,
            attn_mask=self_attn_mask,
            need_weights=need_weights,
        )
        x = self.self_attn_residual_block(
            before=before,
            after=x
        )
        if self.post_norm:
            x = self.self_attn_layernorm(x)
        # Decoder Positional Attention
        if self.apply_pos_attn:
            before = x
            if not self.post_norm:
                x = self.pos_attn_layernorm(x)
            x, attn = self.pos_attn(
                x,
                key_padding_mask=decoder_padding_mask,
                attn_mask=None,
                need_weights=need_weights,
            )
            x = self.pos_attn_residual_block(
                before=before,
                after=x
            )
            if self.post_norm:
                x = self.pos_attn_layernorm(x)
        # Encoder-Decoder Attention
        before = x
        if not self.post_norm:
            x = self.encdec_attn_layernorm(x)
        x, attn = self.encdec_attn(
            dec_query=x,
            enc_key_value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            attn_mask=None,
            need_weights=need_weights,
        )
        x = self.encdec_attn_residual_block(
            before=before,
            after=x
        )
        if self.post_norm:
            x = self.encdec_attn_layernorm(x)
        # FeedForward Network
        before = x
        if not self.post_norm:
            x = self.ffn_layernorm(x)
        x = self.ffn(x)
        x = self.ffn_residual_block(
            before=before,
            after=x
        )
        if self.post_norm:
            x = self.ffn_layernorm(x)
        return x, attn

