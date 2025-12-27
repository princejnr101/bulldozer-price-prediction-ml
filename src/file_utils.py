from __future__ import annotations

from pathlib import Path
from typing import Iterable


def find_first_existing(paths: Iterable[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of these paths exist:\n" + "\n".join(str(p) for p in paths))


def find_kaggle_csv(filename_prefix: str) -> Path:
    """
    Tries common locations and common naming variations.
    Examples:
      - TrainAndValid.csv
      - TrainAndValid (1).csv
      - TrainAndValid.csv.csv
    """
    cwd = Path(".").resolve()
    candidates = []

    for folder in [cwd, cwd/"data"]:
        candidates.append(folder/f"{filename_prefix}.csv")
        candidates.append(folder/f"{filename_prefix} (1).csv")
        candidates.append(folder/f"{filename_prefix}.csv.csv")

        for p in folder.glob(f"{filename_prefix}*.csv"):
            candidates.append(p)

    seen = set()
    uniq = []
    for p in candidates:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)

    return find_first_existing(uniq)
