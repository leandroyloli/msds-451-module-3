"""Run the finished layers in order.

Python cannot import a module whose name starts with a digit, so each
layer is loaded from its filename.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent


def load(filename: str):
    path = PIPELINE / filename
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    data = load("02_data.py")
    data.main()
    features = load("03_features.py")
    features.main()
    split = load("04_split.py")
    split.main()
    selection = load("05_selection.py")
    selection.main()
    model = load("06_model.py")
    model.main()
    evaluate = load("07_evaluate.py")
    evaluate.main()
    report = load("08_report.py")
    report.main()


if __name__ == "__main__":
    main()
