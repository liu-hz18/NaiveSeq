from torch import nn
from torch.nn import functional as F

from ..modules import cross_entropy

class LabelSmoothedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy loss with label smoothing.
    For training, the loss is smoothed with parameter eps,
    while for evaluation, the smoothing is disabled.
    """

    def __init__(self, eps, ignore_index=-100, weight=None, reduction='sum'):
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        # [batch, c, d1, ..., dk]
        non_pad_mask = target.ne(self.ignore_index)
        ntokens = non_pad_mask.sum().data.item()

        if self.training:
            loss = cross_entropy(input, target, ignore_index=self.ignore_index, smoothing=self.eps, reduction=self.reduction)
        else:
            loss = cross_entropy(input, target, ignore_index=self.ignore_index, smoothing=0.0, reduction=self.reduction)

        return {
            'loss': loss,
            'ntokens': ntokens,
        }


class LabelSmoothedCrossEntropyLossWithLength(nn.Module):
    """
    Cross Entropy loss with label smoothing and Length loss.
    For training, the loss is smoothed with parameter eps,
    while for evaluation, the smoothing is disabled.
    """

    def __init__(self, eps, ignore_index=-100, reduction='sum'):
        super(LabelSmoothedCrossEntropyLossWithLength, self).__init__()
        self.label_smooth_crossentropy = LabelSmoothedCrossEntropyLoss(
            eps, ignore_index, reduction)
        self.eps = eps
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target, output_lens, target_lens):
        loss, ntokens = *self.label_smooth_crossentropy(
            output, target)
        bsz = output_lens.shape[0]

        length_loss = cross_entropy(
            output_lens, target_lens, reduction=self.reduction)

        return {
            'loss': loss,
            'length_loss': length_loss,
            'ntokens': ntokens,
            'bsz': bsz,
        }
