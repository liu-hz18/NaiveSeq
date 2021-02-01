import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import len2mask


class LengthPredictor(nn.Module):

    def __init__(self, embed_dim: int, padding_idx: int, max_seq_length: int = 128):
        self.length_embedding = nn.Embedding(
            max_seq_length, embed_dim, None)
        self.max_seq_length = max_seq_length
        self.length_offset = (max_seq_length + 1) // 2

    def forward(
        self,
        enc_feats: torch.Tensor,
        src_masks: torch.Tensor,
        tgt_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # src_masks: B x T, `1` is padding
        enc_feats = self._mean_pooling(enc_feats, src_masks)
        length_offset_out = F.linear(enc_feats, self.length_embedding.weight)
        # src_lens: B
        src_lens = (~src_masks).type_as(enc_feats).sum(1).long()
        if tgt_tokens is not None:
            # tgt_tokens: B x T
            # real_length_tgt: B
            real_length_tgt = tgt_tokens.ne(self.padding_idx).sum(1).long()
        else:
            real_length_tgt = None
        pred_lengs_offset = length_offset_out.max(-1)[1]
        pred_length_tgt = pred_lengs_offset - self.length_offset + src_lens
        pred_length_tgt = pred_length_tgt.clamp(min=2, max=self.max_seq_length)
        return pred_length_tgt, real_length_tgt
    
    def make_decoder_prev_input_tokens(self, src_tokens, length_tgt, padding_idx: int, bos_idx: int, eos_idx: int):
        # src_tokens: B x T
        # length_tgt: B
        src_masks = src_tokens.ne(self.padding_idx) # 0 is padding
        length_src = (~src_masks).float().sum(1).long() # B
        max_tgt_len = length_tgt.max()
        steps = (length_src.float() - 1) / \
            (length_tgt.float() - 1)  # step-size, 对应为位置相除
        index_t = torch.arange(
            end=max_tgt_len[-1]).cuda().expand(*max_tgt_len).contiguous().float()
        index_t = steps[:, None] * index_t[None, :]  # B x T
        mapped_inputs = torch.round(index_t).long().detach()
        # gather: out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        copied_tokens = torch.gather(src_tokens, dim=1, index=mapped_inputs)
        copied_tokens[:, 0] = bos_idx
        copied_tokens.scatter_(dim=1, index=length_tgt[:, None]-1, src=eos_idx)
        tgt_masks = len2mask(length_tgt, padding_flag=False)  # 0 is padding
        copied_tokens.masked_fill_(~tgt_masks, padding_idx)
        return copied_tokens

    def _mean_pooling(self, enc_feats, src_masks):
        # enc_feats: T x B x C
        # src_masks: B x T or None
        if src_masks is None:
            enc_feats = enc_feats.mean(0)
        else:
            src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
            enc_feats = (
                (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
            ).sum(0)
        return enc_feats
