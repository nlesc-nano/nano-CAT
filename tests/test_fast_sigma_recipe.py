"""Tests for :mod:`nanoCAT.recipes.fast_sigma`."""

from __future__ import annotations

import os
import shutil
from typing import Mapping, Any, Type
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from assertionlib import assertion
from nanoCAT.recipes import run_fast_sigma, get_compkf, read_csv

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


class TestFastSigma:
    """Tests for :func:`nanoCAT.recipes.run_fast_sigma`."""

    @pytest.mark.slow
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

    @pytest.mark.slow
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
