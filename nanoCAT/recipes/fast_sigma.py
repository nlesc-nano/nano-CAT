"""
nanoCAT.recipes.fast_sigma
==========================

A recipe for calculating specific COSMO-RS properties using the `fast-sigma <https://www.scm.com/doc/COSMO-RS/Fast_Sigma_QSPR_COSMO_sigma-profiles.html>`_ approximation.

Index
-----
.. currentmodule:: nanoCAT.recipes
.. autosummary::
    run_fast_sigma
    get_compkf

API
---
.. autofunction:: run_fast_sigma
.. autofunction:: get_compkf

"""  # noqa: E501

from __future__ import annotations

import os
import sys
import copy
import operator
import subprocess
import tempfile
import warnings
import contextlib
import functools
import multiprocessing
from typing import Any, ContextManager, cast, overload, TYPE_CHECKING
from pathlib import Path
from itertools import chain
from collections.abc import Iterable, Mapping, Callable

import numpy as np
import pandas as pd
from more_itertools import chunked
from qmflows import InitRestart
from scm.plams import CRSJob, CRSResults, Settings
from CAT.utils import get_template
from CAT.data_handling.validate_mol import santize_smiles

if TYPE_CHECKING:
    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal

__all__ = ["get_compkf", "run_fast_sigma"]

LOGP_SETTINGS = get_template('qd.yaml')['COSMO-RS logp']
LOGP_SETTINGS.update(get_template('crs.yaml')['ADF combi2005'])
LOGP_SETTINGS.input.compound.append(
    Settings({"_h": None, "_1": "compkffile"})
)

GAMMA_E_SETTINGS = get_template('qd.yaml')['COSMO-RS activity coefficient']
GAMMA_E_SETTINGS.update(get_template('crs.yaml')['ADF combi2005'])
GAMMA_E_SETTINGS.input.compound[0]._1 = "compkffile"

SOL_SETTINGS = copy.deepcopy(GAMMA_E_SETTINGS)
SOL_SETTINGS.input.property._h = "puresolubility"
SOL_SETTINGS.input.temperature = "298.15 298.15 1"
SOL_SETTINGS.input.pressure = "1.01325"

BP_SETTINGS = copy.deepcopy(GAMMA_E_SETTINGS)
BP_SETTINGS.input.property._h = "pureboilingpoint"
BP_SETTINGS.input.property._1 = "Pure"
BP_SETTINGS.input.temperature = "298.15"
BP_SETTINGS.input.pressure = "1.01325 1.01325 1"
BP_SETTINGS.input.compound[0].frac1 = 1.0
del BP_SETTINGS.input.compound[1]


def get_compkf(
    smiles: str,
    directory: None | str | os.PathLike[str] = None,
    name: None | str = None,
) -> None | str:
    """Estimate the sigma profile of a SMILES string using the COSMO-RS fast-sigma method.

    See the COSMO-RS `docs <https://www.scm.com/doc/COSMO-RS/Fast_Sigma_QSPR_COSMO_sigma-profiles.html>`_ for more details.

    Parameters
    ----------
    smiles : :class:`str`
        The SMILES string of the molecule of interest.
    directory : :class:`str`, optional
        The directory wherein the resulting ``.compkf`` file should be stored.
        If :data:`None`, use the current working directory.
    name : :class:`str`
        The name of the to-be created .compkf file (excluding extensions).
        If :data:`None`, use **smiles**.

    Returns
    -------
    :class:`str`, optional
        The absolute path to the created ``.compkf`` file.
        :data:`None` will be returned if an error is raised by AMS.

    """  # noqa: E501
    filename = smiles if name is None else name
    if directory is None:
        directory = os.getcwd()
    abs_file = os.path.join(directory, f'{filename}.compkf')

    try:
        status = subprocess.run(
            f'"$AMSBIN"/fast_sigma --smiles "{smiles}" -o "{abs_file}"',
            shell=True, check=True, capture_output=True,
        )
        stderr = status.stderr.decode()
        assert not stderr, stderr
    except (AssertionError, subprocess.SubprocessError) as ex:
        warn = RuntimeWarning(f"Failed to compute the sigma profile of {smiles!r}")
        warn.__cause__ = ex
        warnings.warn(warn)
        return None
    return abs_file


def _get_properties(
    smiles: str,
    directory: str | os.PathLike[str],
    solvents: Mapping[str, str],
) -> list[float]:
    smiles_name = santize_smiles(smiles)
    solute = get_compkf(smiles, directory, name=smiles_name)
    if solute is None:
        return (2 + 3 * len(solvents)) * [np.nan]

    ret = _get_boiling_point(solute, smiles_name)
    ret += _get_logp(solute, smiles_name)
    for name, solv in solvents.items():
        for func in [_get_gamma_e, _get_solubility]:
            ret += func(solute, smiles_name, solv, name)
    return ret


def _get_boiling_point(solute: str, solute_name: str) -> list[float]:
    """Perform a boiling point calculation."""
    s = copy.deepcopy(BP_SETTINGS)
    s.input.compound[0]._h = f'"{solute}"'

    return _run_crs(
        s, solute_name,
        boiling_point=lambda r: r.readkf('PUREBOILINGPOINT', 'temperature'),
    )


def _get_logp(solute: str, solute_name: str) -> list[float]:
    """Perform a LogP calculation."""
    s = copy.deepcopy(LOGP_SETTINGS)
    for v in s.input.compound[:2]:
        v._h = v._h.format(os.environ["AMSRESOURCES"])
    s.input.compound[2]._h = f'"{solute}"'

    return _run_crs(
        s, solute_name,
        logp=lambda r: r.readkf('LOGP', 'logp')[2],
    )


def _get_gamma_e(solute: str, solute_name: str, solvent: str, solvent_name: str) -> list[float]:
    """Perform an activity coefficient and solvation energy calculation."""
    s = copy.deepcopy(GAMMA_E_SETTINGS)
    s.input.compound[0]._h = f'"{solute}"'
    s.input.compound[1]._h = f'"{solvent}"'

    return _run_crs(
        s, solute_name, solvent_name,
        activity_coefficient=lambda r: r.get_activity_coefficient(),
        solvation_energy=lambda r: r.get_energy(),
    )


def _get_solubility(solute: str, solute_name: str, solvent: str, solvent_name: str) -> list[float]:
    """Perform a solubility calculation."""
    s = copy.deepcopy(SOL_SETTINGS)
    s.input.compound[0]._h = f'"{solute}"'
    s.input.compound[1]._h = f'"{solvent}"'

    return _run_crs(
        s, solute_name,
        solubility=lambda r: r.readkf('PURESOLUBILITY', 'solubility mol_per_L_solvent')[1],
    )


def _run_crs(
    settings: Settings,
    solute: str,
    solvent: None | str = None,
    **callbacks: Callable[[CRSResults], float],
) -> list[float]:
    """Perform all COSMO-RS calculations."""
    name = f'{solute}' if solvent is None else f'{solute}_{solvent}'
    job = CRSJob(name=name, settings=settings)

    results = job.run()
    if job.status in ('failed', 'crashed'):
        return len(callbacks) * [np.nan]

    ret = []
    for prop, callback in callbacks.items():
        try:
            value = callback(results)
        except Exception as ex:
            msg = f"Failed to extract the {prop!r} property of {solute!r}"
            if solvent is not None:
                msg += f"in {solvent!r}"

            warn = RuntimeWarning(msg)
            warn.__cause__ = ex
            warnings.warn(warn)
            ret.append(np.nan)
        else:
            ret.append(value)
    return ret


def _abspath(path: str | bytes | os.PathLike[Any]) -> str:
    """Path sanitizing."""
    return os.path.abspath(os.path.expandvars(os.fsdecode(path)))


def _inner_loop(
    args: tuple[int, pd.Index],
    columns: pd.MultiIndex,
    output_dir: Path,
    ams_dir: None | str,
    solvents: Mapping[str, str],
) -> pd.DataFrame:
    """Perform the inner loop of :func:`run_fast_sigma`."""
    i, index = args
    if not len(index):
        df = pd.DataFrame(index=index, columns=columns)
        df.sort_index(axis=1, inplace=True)
        return df

    # Skip if a .csv file already exists
    df_filename = output_dir / f"{i}.temp.csv"
    if os.path.isfile(df_filename):
        return pd.read_csv(df_filename, header=[0, 1], index_col=0)

    # Parse the ams directory
    if ams_dir is None:
        ams_dir_cm: ContextManager[str] = tempfile.TemporaryDirectory(dir=output_dir)
    else:
        ams_dir_cm = contextlib.nullcontext(ams_dir)

    # Calculate properties for the given chunk
    with ams_dir_cm as workdir, InitRestart(*os.path.split(workdir)):
        iterator = chain.from_iterable(
            _get_properties(smiles, workdir, solvents) for smiles in index
        )

        count = len(index) * len(columns)
        shape = len(index), len(columns)
        data = np.fromiter(iterator, dtype=np.float64, count=count)
        data.shape = shape

    df = pd.DataFrame(data, index=index, columns=columns)
    df.sort_index(axis=1, inplace=True)
    df.to_csv(df_filename)
    return df


@overload
def run_fast_sigma(
    input_smiles: Iterable[str],
    solvents: Mapping[str, str | bytes | os.PathLike[Any]],
    *,
    output_dir: str | bytes | os.PathLike[Any] = ...,
    ams_dir: None | str | bytes | os.PathLike[Any] = ...,
    chunk_size: int = ...,
    processes: None | int = ...,
    return_df: Literal[False] = ...,
) -> None:
    ...
@overload  # noqa: E302
def run_fast_sigma(
    input_smiles: Iterable[str],
    solvents: Mapping[str, str | bytes | os.PathLike[Any]],
    *,
    output_dir: str | bytes | os.PathLike[Any] = ...,
    ams_dir: None | str | bytes | os.PathLike[Any] = ...,
    chunk_size: int = ...,
    processes: None | int = ...,
    return_df: Literal[True],
) -> pd.DataFrame:
    ...
def run_fast_sigma(  # noqa: E302
    input_smiles: Iterable[str],
    solvents: Mapping[str, str | bytes | os.PathLike[Any]],
    *,
    output_dir: str | bytes | os.PathLike[Any] = "crs",
    ams_dir: None | str | bytes | os.PathLike[Any] = None,
    chunk_size: int = 1000,
    processes: None | int = None,
    return_df: bool = False,
) -> None | pd.DataFrame:
    """Perform (fast-sigma) COSMO-RS property calculations on the passed SMILES and solvents.

    The output is exported to the ``cosmo-rs.csv`` file.

    Includes the following 5 properties:

    * Boiling Point
    * LogP
    * Activety Coefficient
    * Solvation Energy
    * Solubility

    Jobs are performed in parallel, with chunks of a given size being
    distributed to a user-specified number of processes and subsequently cashed.
    After all COSMO-RS calculations have been performed, the temporary
    .csv files are concatenated into ``cosmo-rs.csv``.

    Examples
    --------
    .. code-block:: python

        >>> import os
        >>> import pandas as pd
        >>> from nanoCAT.recipes import run_fast_sigma

        >>> output_dir: str = ...
        >>> smiles_list = ["CO[H]", "CCO[H]", "CCCO[H]"]
        >>> solvent_dict = {
        ...     "water": "$AMSRESOURCES/ADFCRS/Water.coskf",
        ...     "octanol": "$AMSRESOURCES/ADFCRS/1-Octanol.coskf",
        ... }

        >>> run_fast_sigma(smiles_list, solvent_dict, output_dir=output_dir)

        >>> csv_file = os.path.join(output_dir, "cosmo-rs.csv")
        >>> pd.read_csv(csv_file, header=[0, 1], index_col=0)
        property Activity Coefficient             ... Solvation Energy
        solvent               octanol      water  ...          octanol     water
        smiles                                    ...
        CO[H]                1.045891   4.954782  ...        -2.977354 -3.274420
        CCO[H]               0.980956  12.735228  ...        -4.184214 -3.883986
        CCCO[H]              0.905952  47.502557  ...        -4.907177 -3.779867

        [3 rows x 8 columns]

    Parameters
    ----------
    input_smiles : :class:`Iterable[str] <collections.abc.Iterable>`
        The input SMILES strings.
    solvents : :class:`Mapping[str, path-like] <collections.abc.Mapping>`
        A mapping with solvent-names as keys and paths to their respective
        .coskf files as values.

    Keyword Arguments
    -----------------
    output_dir : :term:`path-like object`
        The directory wherein the .csv files will be stored.
        A new directory will be created if it does not yet exist.
    plams_dir : :term:`path-like <path-like object>`, optional
        The directory wherein all COSMO-RS computations will be performed.
        If :data:`None`, use a temporary directory inside **output_dir**.
    chunk_size : :class:`int`
        The (maximum) number of entries to-be stored in a single .csv file.
    processes : :class:`int`, optional
        The number of worker processes to use.
        If :data:`None`, use the number returned by :func:`os.cpu_count()`.
    return_df : :class:`bool`
        If :data:`True`, return a dataframe with the content of ``cosmo-rs.csv``.

    """
    # Parse the `chunk_size`
    chunk_size = operator.index(chunk_size)
    if chunk_size < 1:
        raise ValueError(f"`chunk_size` must be larger than zero; observed value: {chunk_size}")

    # Parse `processes`
    if processes is not None:
        processes = operator.index(processes)
        if processes < 1:
            raise ValueError(f"`processes` must be larger than zero; observed value {processes}")

    # Parse the `solvents`
    if len(solvents) == 0:
        raise ValueError("`solvents` requires at least one solvent")
    solvents = cast("dict[str, str]", {k: _abspath(v) for k, v in solvents.items()})

    # Parse `output_dir`
    output_dir = Path(_abspath(output_dir))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Parse `ams_dir`
    if ams_dir is not None:
        ams_dir = _abspath(ams_dir)

    # Construct the dataframe columns
    prop_names = ["Activity Coefficient", "Solvation Energy", "Solubility"]
    _columns: list[tuple[str, None | str]] = [("Boiling Point", None), ("LogP", None)]
    for solv in solvents:
        _columns += [(prop, solv) for prop in prop_names]
    columns = pd.MultiIndex.from_tuples(_columns, names=["property", "solvent"])

    # Run the workflow
    with multiprocessing.Pool(processes) as pool:
        enumerator = enumerate(
            pd.Index(lst, name="smiles") for lst in chunked(input_smiles, chunk_size)
        )
        func = functools.partial(
            _inner_loop,
            columns=columns, output_dir=output_dir, solvents=solvents, ams_dir=ams_dir,
        )
        if not return_df:
            ret = None
            for _ in pool.imap_unordered(func, enumerator):
                pass
        else:
            ret = pd.concat([df for df in pool.imap_unordered(func, enumerator)])
    _concatenate_csv(output_dir)
    return ret


def _concatenate_csv(output_dir: Path) -> None:
    """Concatenate all ``{i}.tmp.csv`` files into ``cosmo-rs.csv``."""
    csv_files = [output_dir / i for i in os.listdir(output_dir) if
                 os.path.splitext(i)[1] == ".csv" and i != "cosmo-rs.csv"]
    csv_files.sort(key=lambda n: int(n.name.split(".", 1)[0]))
    iterator = iter(csv_files)

    # Construct the final .csv file
    output_csv = output_dir / "cosmo-rs.csv"
    if not os.path.isfile(output_csv):
        file = next(iterator)
        df = pd.read_csv(file, header=[0, 1], index_col=0)
        df.to_csv(output_csv)
        os.remove(file)

    # Append its content using that of all other .csv files
    with open(output_csv, "a") as f:
        for file in iterator:
            df = pd.read_csv(file, header=[0, 1], index_col=0)
            df.to_csv(f, header=False)
            os.remove(file)
