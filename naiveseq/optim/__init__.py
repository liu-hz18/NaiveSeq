"""isort:skip_file"""

import importlib
import os

def get_lr(optimizer):
    """Return the current learning rate."""
    return self.param_groups[0]["lr"]

def set_lr(optimizer, lr):
    """Set the learning rate."""
    for param_group in self.param_groups:
        param_group["lr"] = lr

# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("naiveseq.optim." + file_name)
