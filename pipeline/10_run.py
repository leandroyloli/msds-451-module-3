"""Run the layers in order.

Python cannot import a module whose name starts with a digit, so each
layer is loaded from its filename. The saved price file is reused unless
--download is passed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent


def load(filename: str):
    path = PIPELINE / filename
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    # Registered so worker processes can find the module's functions by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LAYERS = (
    "03_features.py",
    "04_split.py",
    "05_universe.py",
    "06_selection.py",
    "07_model.py",
    "08_evaluate.py",
    "09_report.py",
)


def main() -> None:
    if "--download" in sys.argv:
        load("02_data.py").main()
    for filename in LAYERS:
        load(filename).main()


if __name__ == "__main__":
    main()
