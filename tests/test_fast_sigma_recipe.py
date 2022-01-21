"""Tests for :mod:`nanoCAT.recipes.fast_sigma`."""

from __future__ import annotations

import os
import shutil
from typing import Mapping, Any, Type
from pathlib import Path
from collections.abc import Hashable

import pytest
import numpy as np
import pandas as pd
from rdkit.Chem import CanonSmiles
from assertionlib import assertion
from nanoCAT.recipes import run_fast_sigma, get_compkf, read_csv, sanitize_smiles_df

PATH = Path("tests") / "test_files"

SMILES = ("CCCO[H]", "CCO[H]", "CO[H]")
SOLVENTS = {
    "water": "$AMSRESOURCES/ADFCRS/Water.coskf",
    "octanol": "$AMSRESOURCES/ADFCRS/1-Octanol.coskf",
}
SOLVENTS2 = {
    "water": PATH / "Water.coskf",
    "octanol": PATH / "1-Octanol.coskf",
}

REF = read_csv(PATH / "cosmo-rs.csv")

SANITIZE_DF = pd.DataFrame(1, index=REF.index.copy(), columns=['a'])

SANITIZE2_DF = SANITIZE_DF.copy()
SANITIZE2_DF.columns = pd.MultiIndex.from_tuples(
    [(i,) for i in SANITIZE_DF.columns],
    names=(SANITIZE_DF.columns.name,),
)

SANITIZE3_DF = SANITIZE_DF.copy()
SANITIZE3_DF.columns = pd.MultiIndex.from_tuples(
    [(i, None) for i in SANITIZE_DF.columns],
    names=(SANITIZE_DF.columns.name, None),
)

SANITIZE4_DF = SANITIZE_DF.copy()
SANITIZE4_DF.index = pd.Index([None] * len(SANITIZE_DF.index), name=SANITIZE_DF.index.name)


def has_env_vars(*env_vars: str) -> bool:
    """Check if the passed environment variables are available."""
    return set(env_vars).issubset(os.environ)


def compare_df(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Compare the content of two dataframes."""
    __tracebackhide__ = True

    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)
    np.testing.assert_array_equal(df1.columns, df2.columns, err_msg="columns")
    np.testing.assert_array_equal(df1.index, df2.index, err_msg="index")

    iterator = ((k, df1[k], df2[k]) for k in df1.keys())
    for k, v1, v2 in iterator:
        if issubclass(v2.dtype.type, np.inexact):
            np.testing.assert_allclose(v1, v2, err_msg=k)
        else:
            np.testing.assert_array_equal(v1, v2, err_msg=k)


@pytest.mark.slow
@pytest.mark.skipif(not has_env_vars("AMSBIN", "AMSHOME", "AMSRESOURCES"), reason="Requires AMS")
class TestFastSigma:
    """Tests for :func:`nanoCAT.recipes.run_fast_sigma`."""

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"return_df": True}, {"chunk_size": 1, "return_df": True}, {"processes": 1}],
        ids=["default", "return_df", "chunk_size", "processes"]
    )
    def test_passes(self, kwargs: Mapping[str, Any]) -> None:
        """Test that whether the code passes as expected."""
        # Can't use the pytest `tmp_path` paramater as the (absolute) filename
        # becomes too long for COSMO-RS
        tmp_path = PATH / "crs"
        try:
            os.mkdir(tmp_path)

            out = run_fast_sigma(SMILES, SOLVENTS, output_dir=tmp_path, **kwargs)
            if out is not None:
                compare_df(out, REF)

            csv_file = tmp_path / "cosmo-rs.csv"
            assertion.isfile(csv_file)

            df = read_csv(csv_file)
            compare_df(df, REF)
        finally:
            shutil.rmtree(tmp_path)

    @pytest.mark.parametrize(
        "solvents,kwargs,exc",
        [
            (SOLVENTS, {"chunk_size": 1.0}, TypeError),
            (SOLVENTS, {"chunk_size": 0}, ValueError),
            (SOLVENTS, {"processes": 1.0}, TypeError),
            (SOLVENTS, {"processes": 0}, ValueError),
            ({}, {}, ValueError),
            (SOLVENTS, {"output_dir": 1.0}, TypeError),
            (SOLVENTS, {"ams_dir": 1.0}, TypeError),
            (SOLVENTS2, {}, FileNotFoundError),
            (SOLVENTS, {"log_options": {"bob": None}}, KeyError),
        ],
    )
    def test_raises(
        self, solvents: Mapping[str, Any], kwargs: Mapping[str, Any], exc: Type[Exception]
    ) -> None:
        """Test that whether appropiate exception is raised."""
        with pytest.raises(exc):
            run_fast_sigma(SMILES, solvents, **kwargs)

    def test_warns(self, tmp_path: Path) -> None:
        """Test that whether appropiate warning is issued."""
        with pytest.warns(RuntimeWarning) as record:
            get_compkf("bob", tmp_path)
        cause: None | BaseException = getattr(record[0].message, "__cause__", None)
        assertion.isinstance(cause, RuntimeError)


class TestReadCSV:
    """Tests for :func:`nanoCAT.recipes.read_csv`."""

    @pytest.mark.parametrize(
        "columns",
        [
            None,
            ["molarvol", "gidealgas", "Activity Coefficient"],
            [("molarvol", None), ("gidealgas", None)],
            ["molarvol"],
            ("molarvol", None),
            "molarvol",
            [("Solvation Energy", "water")],
            ("Solvation Energy", "water"),
            ["Solvation Energy"],
            "Solvation Energy",
        ],
    )
    def test_pass(self, columns: None | Any) -> None:
        """Test that whether the code passes as expected."""
        df = read_csv(PATH / "cosmo-rs.csv", columns=columns)
        if columns is None:
            ref = REF
        elif not isinstance(columns, list):
            ref = REF[[columns]]
        else:
            ref = REF[columns]
        compare_df(df, ref)

    @pytest.mark.parametrize(
        "exc,file,kwargs",
        [
            (FileNotFoundError, "bob.csv", {}),
            (TypeError, "cosmo-rs.csv", {"bob": None}),
            (TypeError, "cosmo-rs.csv", {"usecols": None}),
        ]
    )
    def test_raises(self, exc: Type[Exception], file: str, kwargs: Mapping[str, Any]) -> None:
        """Test that whether appropiate exception is raised."""
        with pytest.raises(exc):
            read_csv(PATH / file, **kwargs)


class TestSanitize:
    """Tests for :func:`nanoCAT.recipes.sanitize_smiles_df`."""

    @pytest.mark.parametrize("column_levels", [1, 2, 3])
    @pytest.mark.parametrize("column_padding", [None, 1.0])
    @pytest.mark.parametrize("df", [SANITIZE_DF, SANITIZE2_DF])
    def test_pass(self, df: pd.DataFrame, column_levels: int, column_padding: Hashable) -> None:
        """Test that whether the code passes as expected."""
        out = sanitize_smiles_df(df, column_levels, column_padding)
        assertion.is_(out, df, invert=True)
        assertion.is_(out.columns, df.columns, invert=True)
        assertion.is_(out.index, df.index, invert=True)

        assertion.eq(len(out.columns.levels), column_levels)
        np.testing.assert_array_equal([CanonSmiles(i) for i in out.index], out.index)

        offset = len(df.columns.levels) if isinstance(df.columns, pd.MultiIndex) else 1
        for idx in out.columns.levels[offset:]:
            np.testing.assert_array_equal(idx, column_padding)

    @pytest.mark.parametrize(
        "exc,kwargs",
        [
            (TypeError, {"column_levels": 1.0}),
            (ValueError, {"column_levels": 0}),
            (ValueError, {"column_levels": 1}),
            (TypeError, {"column_padding": []}),
        ]
    )
    def test_raises(self, exc: Type[Exception], kwargs: Mapping[str, Any]) -> None:
        """Test that an appropiate exception is raised."""
        with pytest.raises(exc):
            sanitize_smiles_df(SANITIZE3_DF, **kwargs)

    def test_warns(self) -> None:
        """Test that an appropiate warning is issued."""
        with pytest.warns(RuntimeWarning):
            sanitize_smiles_df(SANITIZE4_DF)
