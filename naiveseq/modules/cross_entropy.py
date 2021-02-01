# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def _cross_entropy_pytorch(logits, target, ignore_index=None, smoothing=0.0, reduction="mean"):
    lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    loss = nll_loss = F.nll_loss(
        lprobs,
        target,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    if smoothing > 0.0:
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        pad_mask = target.ne(ignore_index)
        if reduction == 'mean':
            smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0).mean()
        else:
            smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0).sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = nll_loss * (1. - self.eps) + (smooth_loss * eps_i)

    return loss


try:
    import xentropy_cuda
    from apex.contrib import xentropy

    def cross_entropy(logits, target, ignore_index=-100, smoothing=0.0, reduction="mean"):
        if logits.device == torch.device("cpu"):
            return _cross_entropy_pytorch(logits, target, ignore_index, reduction)
        else:
            if not getattr(cross_entropy, "_has_logged_once", False):
                logger.info("using fused cross entropy")
                cross_entropy._has_logged_once = True

            half_to_float = logits.dtype == torch.half
            losses = xentropy.SoftmaxCrossEntropyLoss.apply(
                logits,
                target,
                smoothing,
                ignore_index,
                half_to_float,
            )
            if reduction == "sum":
                return losses.sum()
            elif reduction == "mean":
                if ignore_index >= 0:
                    return losses.sum() / target.ne(ignore_index).sum()
                else:
                    return losses.mean()
            elif reduction == "none":
                return losses
            else:
                raise NotImplementedError


except ImportError:

    def cross_entropy(logits, target, ignore_index=-100, smoothing=0.0, reduction="mean"):
        return _cross_entropy_pytorch(logits, target, ignore_index, smoothing, reduction)
