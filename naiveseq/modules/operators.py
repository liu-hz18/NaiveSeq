
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def gumbel_noise(input, eps=1e-8):
    return (
        input.new_zeros(*input.size())
        .uniform_()
        .add_(eps)
        .log_()
        .neg_()
        .add_(eps)
        .log_()
        .neg_()
    )

@torch.jit.script
def log_gumbel_softmax(logits, tau=1.0):
    # 相当于 np.random.choice 采样多次的效果，但是保证可导
    G = gumbel_noise(logits)
    return F.log_softmax((logits + G) / tau, dim=-1)

class StraightThroughLogits(nn.Module):
    """
    onehot(argmax(x))
    """
    def __init__(self):
        super(StraightThroughLogits, self).__init__()

    def forward(self, logits):
        index = logits.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim=2, index=index, value=1.)
        ret = (y_hard - logits).detach() + logits
        return ret


class StraightThroughSoftmax(nn.Module):
    """
    `onehot(argmax(softmax(x)))`
    """
    def __init__(self, dim=-1):
        super(StraightThroughSoftmax, self).__init__()
        self.dim = dim

    def forward(self, logits):
        logits = F.softmax(logits, dim=self.dim)
        index = logits.max(dim=self.dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim=2, index=index, value=1.)
        ret = (y_hard - logits).detach() + logits
        return ret


class LogGumbelSoftmax(nn.Module):
    '''
    Sampled tensor of same shape as logits from the Gumbel-Softmax distribution
    If `hard=True`, the returned samples will be one-hot,
    e.g. `onehot(argmax(gumbel_softmax(x)))`
    otherwise they will be probability distributions that sum to 1 across dim.
    '''
    def __init__(self, dim=-1, tau=1):
        super(LogGumbelSoftmax, self).__init__()
        self.dim = dim
        self.tau = tau

    def forward(self, logits):
        return log_gumbel_softmax(logits, tau=self.tau)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)


def onehot3d(index, class_num):
    index = index.unsqueeze(-1).repeat(1, 1, class_num)
    return torch.zeros_like(index).scatter_(dim=2, index=index, value=1.).float()


def onehot2d(index, class_num):
    index = index.unsqueeze(-1).repeat(1, class_num)
    return torch.zeros_like(index).scatter_(dim=1, index=index, value=1.).float()
