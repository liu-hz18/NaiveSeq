import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq

from fairseq.module import MultiheadAttention as _MultiheadAttention


class EncDecAttention(nn.Module):
    """A multi-head attention layer.
    !!! Time First, NOT batch first !!!
    """
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, attn_dropout=0.0, bias=True, q_noise=0.0, qn_block_size=8):
        super(EncDecAttention, self).__init__()
        self.multihead_attn = _MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=attn_dropout,
            self_attention=False,
            encoder_decoder_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
        self,
        dec_query: torch.Tensor,
        enc_key_value: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        attn_mask: Optional[torch.BoolTensor] = None,
        need_weights: bool = True,
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
        return self.multihead_attn(
            query=dec_query,
            key=enc_key_value,
            value=enc_key_value.clone(),
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )
