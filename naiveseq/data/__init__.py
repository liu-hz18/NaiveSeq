"""isort:skip_file"""

import importlib
import os

# automatically import any Python files in the data/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("naiveseq.data." + file_name)
