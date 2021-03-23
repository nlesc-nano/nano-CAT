from __future__ import annotations

import os
import re
from typing import Any
from pathlib import Path
from itertools import chain, repeat
from collections.abc import Iterator

import pandas as pd

__all__: list[str] = ["_concatenate_csv"]


def _read_empty_dataframe(file: str | bytes | os.PathLike[Any]) -> pd.DataFrame:
    """Read a .csv file that is full of ``nan``s."""
    df = pd.read_csv(file, header=[0, 1])
    df.set_index(("property", "solvent"), inplace=True)
    df.drop("smiles", inplace=True)
    df.index.name = "smiles"
    df.columns.names = ("property", "solvent")
    return df


def _concatenate_csv(output_dir: Path) -> None:
    """Concatenate all ``{i}.tmp.csv`` files into ``cosmo-rs.csv``."""
    pattern = re.compile(r"[0-9]+\.temp\.csv")
    csv_files = [output_dir / i for i in os.listdir(output_dir) if pattern.fullmatch(i) is not None]
    csv_files.sort(key=lambda n: int(n.name.split(".", 1)[0]))
    if not len(csv_files):
        raise FileNotFoundError(f"Failed to identify any files with the {pattern.pattern!r} "
                                f"pattern in {str(output_dir)!r}")

    # Construct the final .csv file
    output_csv = output_dir / "cosmo-rs.csv"
    if not os.path.isfile(output_csv):
        header_iter: Iterator[bool] = chain([True], repeat(False))
    else:
        header_iter = repeat(False)

    # Append its content using that of all other .csv files
    with open(output_csv, "a") as f:
        for file, header in zip(csv_files, header_iter):
            try:
                df = pd.read_csv(file, header=[0, 1], index_col=0)
            except pd.errors.EmptyDataError:
                df = _read_empty_dataframe(file)
            if header:
                df.columns = pd.MultiIndex.from_tuples(
                    [(i, (j if j != "nan" else None)) for i, j in df.columns],
                    names=df.columns.names,
                )
            df.to_csv(f, header=header)
            os.remove(file)
