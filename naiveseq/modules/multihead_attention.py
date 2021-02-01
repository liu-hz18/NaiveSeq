
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules.quant_noise import quant_noise

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: int = None,
        vdim: int = None,
        dropout: float = 0.0,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_dropout = dropout

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor]=None,
        value: Optional[torch.Tensor]=None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        attn_mask: Optional[torch.BoolTensor]=None,
        need_weight: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inputs:
        - key_padding_mask: 
            :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask:
            2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
        """
        if self.self_attention:
            key = key if key is not None else query.clone()
            value = value if value is not None else value.clone()
        elif self.encoder_decoder_attention:
            value = value if value is not None else key.clone()
        assert query and key and value
        return F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            torch.empty([0]),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.attn_dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            key_padding_mask,
            need_weight,
            attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )
