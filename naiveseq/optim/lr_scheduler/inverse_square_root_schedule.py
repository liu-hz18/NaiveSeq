
from collections import Collection
from .. import set_lr

import torch

def InverseSquareRootScheduler(optimizer, warmup_init_lr, max_lr, warmup_updates):
    if isinstance(max_lr, Collection) and len(max_lr) > 1:
        raise ValueError(
            "Cannot use a fixed learning rate schedule with inverse_sqrt."
        )
    warmup_end_lr = max_lr[0] if isinstance(max_lr, Collection) else max_lr
    if warmup_init_lr < 0:
        warmup_init_lr = 0 if warmup_updates > 0 else warmup_end_lr
    lr_warmup_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
    decay_factor = warmup_end_lr * warmup_updates ** 0.5
    set_lr(optimizer, max_lr)
    def step_lr(num_updates):
        if num_updates < warmup_updates:
            lr = warmup_init_lr + num_updates * lr_warmup_step
        else:
            lr = decay_factor * num_updates ** -0.5
        return lr
    lr_lambda = lambda step: step_lr(step) / max_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
