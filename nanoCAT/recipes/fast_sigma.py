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
    read_csv
    sanitize_smiles_df

API
---
.. autofunction:: run_fast_sigma
.. autofunction:: get_compkf
.. autofunction:: read_csv
.. autofunction:: sanitize_smiles_df

"""  # noqa: E501

from __future__ import annotations

import re
import os
import sys
import copy
import types
import hashlib
import operator
import subprocess
import tempfile
import warnings
import contextlib
import functools
import multiprocessing
from typing import Any, ContextManager, cast, overload, TYPE_CHECKING, TypeVar
from pathlib import Path
from itertools import chain, repeat
from collections.abc import Iterable, Mapping, Callable, Iterator, Sequence, Hashable

import numpy as np
import pandas as pd
from more_itertools import chunked
from qmflows import InitRestart
from scm.plams import CRSJob, CRSResults, Settings, KFFile
from rdkit.Chem import CanonSmiles
from CAT.utils import get_template

if TYPE_CHECKING:
    if sys.version_info >= (3, 8):
        from typing import Literal, TypedDict, SupportsIndex
    else:
        from typing_extensions import Literal, TypedDict, SupportsIndex

    _SCT = TypeVar("_SCT", bound=np.generic)
    _NDArray = np.ndarray[Any, np.dtype[_SCT]]

    class _LogOptions(TypedDict, total=False):
        """Verbosity of log messages: 0:none  1:minimal  3:normal  5:verbose  7:extremely talkative."""  # noqa: E501

        #: Verbosity of the log printed to .log file in the main working folder
        file: Literal[0, 1, 3, 5, 7]

        #: Verbosity of the log printed to the standard output
        stdout: Literal[0, 1, 3, 5, 7]

        #: Print time for each log event
        time: bool

        #: Print date for each log event
        date: bool

__all__ = [
    "get_compkf",
    "get_fast_sigma_properties",
    "run_fast_sigma",
    "read_csv",
    "sanitize_smiles_df",
]

LOGP_SETTINGS = get_template('qd.yaml')['COSMO-RS logp']
LOGP_SETTINGS.runscript.nproc = 1
LOGP_SETTINGS.update(get_template('crs.yaml')['ADF combi2005'])
LOGP_SETTINGS.input.property.volumequotient = 6.766

GAMMA_E_SETTINGS = get_template('qd.yaml')['COSMO-RS activity coefficient']
GAMMA_E_SETTINGS.runscript.nproc = 1
GAMMA_E_SETTINGS.update(get_template('crs.yaml')['ADF combi2005'])
del GAMMA_E_SETTINGS.input.compound[0]

SOL_SETTINGS = copy.deepcopy(GAMMA_E_SETTINGS)
SOL_SETTINGS.input.property._h = "puresolubility"
SOL_SETTINGS.input.temperature = "298.15 298.15 1"
SOL_SETTINGS.input.pressure = "1.01325"
SOL_SETTINGS.input.compound = [Settings({"_h": None, "_1": "compkffile"})]

BP_SETTINGS = copy.deepcopy(GAMMA_E_SETTINGS)
BP_SETTINGS.input.property._h = "pureboilingpoint"
BP_SETTINGS.input.property._1 = "Pure"
BP_SETTINGS.input.temperature = "298.15"
BP_SETTINGS.input.pressure = "1.01325 1.01325 1"
del BP_SETTINGS.input.compound

# The default PLAMS `config.log` options
LOG_DEFAULT: _LogOptions = types.MappingProxyType({    # type: ignore[assignment]
    "file": 5,
    "stdout": 3,
    "time": True,
    "date": False,
})


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
    kf_file = os.path.join(directory, f'{filename}.compkf')

    command = f'"$AMSBIN"/fast_sigma --smiles "{smiles}" -o "{kf_file}"'
    output = _run(command, smiles, err_msg="Failed to compute the sigma profile of {!r}")
    return kf_file if output is not None else None


def get_fast_sigma_properties(
    smiles: str,
    directory: None | str | os.PathLike[str] = None,
    name: None | str = None,
) -> None | str:
    """Calculate various pure-compound properties with the COSMO-RS property prediction program.

    See the COSMO-RS `docs <https://www.scm.com/doc/COSMO-RS/Property_Prediction.html>`_ for more details.

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
    kf_file = os.path.join(directory, f'{filename}.compkf')

    command = f'"$AMSBIN"/prop_prediction --smiles "{smiles}" -o "{kf_file}"'
    output = _run(
        command, smiles,
        err_msg="Failed to compute the pure compound properties of {!r}",
    )
    return kf_file if output is not None else None


def _run(command: str, smiles: str, err_msg: str) -> None | subprocess.CompletedProcess[bytes]:
    """Run **command** and return the the status."""
    try:
        status = subprocess.run(command, shell=True, check=True, capture_output=True)
        stderr = status.stderr
        stdout = status.stdout
        if stderr:
            raise RuntimeError(stderr.decode())
        elif b"WARNING" in stdout:
            raise RuntimeError(stdout.decode().strip("\n").rstrip("\n"))
    except (RuntimeError, subprocess.SubprocessError) as ex:
        warn = RuntimeWarning(err_msg.format(smiles))
        warn.__cause__ = ex
        warnings.warn(warn, stacklevel=1)
        return None
    else:
        return status


def _hash_smiles(smiles: str) -> str:
    """Return the sha256 hash of the passed SMILES string."""
    return hashlib.sha256(smiles.encode()).hexdigest()


def _get_compkf(
    smiles_iter: Iterable[str],
    directory: str | os.PathLike[str],
) -> _NDArray[np.object_]:
    """Wrap :func:`get_compkf` in a for-loop."""
    lst = [get_compkf(smiles, directory, name=_hash_smiles(smiles)) for smiles in smiles_iter]
    return np.array(lst, dtype=np.object_)


def _get_fast_sigma_properties(
    smiles_iter: Iterable[str],
    directory: str | os.PathLike[str],
) -> _NDArray[np.object_]:
    """Wrap :func:`get_fast_sigma_properties` in a for-loop."""
    lst = [get_fast_sigma_properties(smiles, directory, name=_hash_smiles(smiles)) for
           smiles in smiles_iter]
    return np.array(lst, dtype=np.object_)


def _set_properties(
    df: pd.DataFrame,
    solutes: _NDArray[np.object_],
    solvents: Mapping[str, str],
    prop_mask: _NDArray[np.bool_],
) -> None:
    df["LogP", None] = _get_logp(solutes)

    for name, solv in solvents.items():
        df[[
            ("Activity Coefficient", name),
            ("Solvation Energy", name),
        ]] = _get_gamma_e(solutes, solv, name)

    prop_array = _get_compkf_prop(solutes, prop_mask)
    iterator = ((k, prop_array[k]) for k in prop_array.dtype.fields)
    for k, v in iterator:
        df[k, None] = v


def _get_compkf_prop(
    solutes: _NDArray[np.object_],
    prop_mask: _NDArray[np.bool_],
) -> _NDArray[np.void]:
    """Extract all (potentially) interesting properties from the compkf file."""
    prop_iter: list[tuple[str, str, type | str]] = [
        ("Compound Data", "Formula", "U160"),
        ("Compound Data", "Molar Mass", np.float64),
        ("Compound Data", "Nring", np.int64),
        ("PROPPREDICTION", "boilingpoint", np.float64),
        ("PROPPREDICTION", "criticalpressure", np.float64),
        ("PROPPREDICTION", "criticaltemp", np.float64),
        ("PROPPREDICTION", "criticalvol", np.float64),
        ("PROPPREDICTION", "density", np.float64),
        ("PROPPREDICTION", "dielectricconstant", np.float64),
        ("PROPPREDICTION", "entropygas", np.float64),
        ("PROPPREDICTION", "flashpoint", np.float64),
        ("PROPPREDICTION", "gidealgas", np.float64),
        ("PROPPREDICTION", "hcombust", np.float64),
        ("PROPPREDICTION", "hformstd", np.float64),
        ("PROPPREDICTION", "hfusion", np.float64),
        ("PROPPREDICTION", "hidealgas", np.float64),
        ("PROPPREDICTION", "hsublimation", np.float64),
        ("PROPPREDICTION", "meltingpoint", np.float64),
        ("PROPPREDICTION", "molarvol", np.float64),
        ("PROPPREDICTION", "parachor", np.float64),
        ("PROPPREDICTION", "solubilityparam", np.float64),
        ("PROPPREDICTION", "tpt", np.float64),
        ("PROPPREDICTION", "vdwarea", np.float64),
        ("PROPPREDICTION", "vdwvol", np.float64),
        ("PROPPREDICTION", "vaporpressure", np.float64),
    ]

    dtype = np.dtype([i[1:] for i in prop_iter])
    fill_value = np.array(tuple(
        (np.nan if field_dtype == np.float64 else np.dtype(field_dtype).type())
        for *_, field_dtype in prop_iter
    ), dtype=dtype)
    ret = np.full_like(solutes, fill_value, dtype=dtype)

    mask = prop_mask & (solutes != None)
    if not mask.any():
        return ret

    iterator = ((i, KFFile(f), f) for i, (f, m) in enumerate(zip(solutes, mask)) if m)
    for i, kf, file in iterator:  # type: int, KFFile, str
        for section, variable, _ in prop_iter:
            try:
                ret[variable][i] = kf.read(section, variable)
            except Exception as ex:
                if kf.reader is None:
                    warn = RuntimeWarning(f"No such file or directory: {file!r}")
                else:
                    smiles = kf.read("Compound Data", "SMILES").strip("\x00")
                    warn = RuntimeWarning(
                        f'Failed to extract the "{section}%{variable}" property of {smiles!r}'
                    )
                warn.__cause__ = ex
                warnings.warn(warn)
    return ret


def _get_logp(solutes: _NDArray[np.object_]) -> _NDArray[np.float64]:
    """Perform a LogP calculation."""
    ret = np.full_like(solutes, np.nan, dtype=np.float64)
    mask = solutes != None
    count = np.count_nonzero(mask)
    if count == 0:
        return ret

    s = copy.deepcopy(LOGP_SETTINGS)
    for v in s.input.compound[:2]:
        v._h = v._h.format(os.environ["AMSRESOURCES"])
    s.input.compound += [Settings({"_h": f'"{sol}"', "_1": "compkffile"}) for sol in solutes[mask]]

    ret[mask] = _run_crs(
        s, count,
        logp=lambda r: r.readkf('LOGP', 'logp')[2:],
    )
    return ret


def _get_gamma_e(
    solutes: _NDArray[np.object_],
    solvent: str,
    solvent_name: str,
) -> _NDArray[np.float64]:
    """Perform an activity coefficient and solvation energy calculation."""
    ret = np.full((len(solutes), 2), np.nan, dtype=np.float64)
    mask = solutes != None
    count = np.count_nonzero(mask)
    if count == 0:
        return ret

    s = copy.deepcopy(GAMMA_E_SETTINGS)
    s.input.compound[0]._h = f'"{solvent}"'
    s.input.compound += [Settings({"_h": f'"{sol}"', "_1": "compkffile"}) for sol in solutes[mask]]

    ret[mask] = _run_crs(
        s, count, solvent_name,
        activity_coefficient=lambda r: r.readkf('ACTIVITYCOEF', 'gamma')[1:],
        solvation_energy=lambda r: r.readkf('ACTIVITYCOEF', 'deltag')[1:],
    )
    return ret


def _run_crs(
    settings: Settings,
    count: int,
    solvent: None | str = None,
    **callbacks: Callable[[CRSResults], float | Sequence[float]],
) -> _NDArray[np.float64]:
    """Perform all COSMO-RS calculations."""
    job = CRSJob(settings=settings)

    results = job.run()
    ret = np.full((len(callbacks), count), np.nan, dtype=np.float64)
    if job.status in ('failed', 'crashed'):
        return ret.T if ret.shape[0] != 1 else np.squeeze(ret, 0)

    for i, (prop, callback) in enumerate(callbacks.items()):
        try:
            value = callback(results)
        except Exception as ex:
            msg = f"Failed to extract the {prop!r} property"
            if solvent is not None:
                msg += f" in {solvent!r}"

            warn = RuntimeWarning(msg)
            warn.__cause__ = ex
            warnings.warn(warn)
        else:
            ret[i] = value
    return ret.T if ret.shape[0] != 1 else np.squeeze(ret, 0)


def _abspath(path: str | bytes | os.PathLike[Any], isfile: bool = False) -> str:
    """Path sanitizing."""
    ret = os.path.abspath(os.path.expandvars(os.fsdecode(path)))
    if isfile and not os.path.isfile(ret):
        open(ret, "r")  # This will raise
        raise
    return ret


def _inner_loop(
    args: tuple[int, pd.Index],
    columns: pd.MultiIndex,
    output_dir: Path,
    ams_dir: None | str,
    solvents: Mapping[str, str],
    log: _LogOptions = LOG_DEFAULT,
) -> tuple[int, pd.DataFrame]:
    """Perform the inner loop of :func:`run_fast_sigma`."""
    i, index = args
    if not len(index):
        df = pd.DataFrame(index=index, columns=columns)
        df.sort_index(axis=1, inplace=True)
        return i, df

    # Skip if a .csv file already exists
    df_filename = output_dir / f"{i}.temp.csv"
    if os.path.isfile(df_filename):
        df = read_csv(df_filename)
        return i, df

    # Parse the ams directory
    if ams_dir is None:
        ams_dir_cm: ContextManager[str] = tempfile.TemporaryDirectory(dir=output_dir)
    else:
        ams_dir_cm = contextlib.nullcontext(ams_dir)

    # Calculate properties for the given chunk
    df = pd.DataFrame(index=index, columns=columns)
    with ams_dir_cm as workdir, InitRestart(*os.path.split(workdir)):
        from scm.plams import config
        config.log.update(log)
        config.job.pickle = False

        compkf_array = _get_compkf(index, workdir)
        prop_mask: _NDArray[np.bool_] = _get_fast_sigma_properties(index, workdir) != None
        _set_properties(df, compkf_array, solvents, prop_mask)

    df.sort_index(axis=1, inplace=True)
    df.to_csv(df_filename)
    return i, df


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
    log_options: _LogOptions = ...,
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
    log_options: _LogOptions = ...,
) -> pd.DataFrame:
    ...
def run_fast_sigma(  # noqa: E302
    input_smiles: Iterable[str],
    solvents: Mapping[str, str | bytes | os.PathLike[Any]],
    *,
    output_dir: str | bytes | os.PathLike[Any] = "crs",
    ams_dir: None | str | bytes | os.PathLike[Any] = None,
    chunk_size: int = 100,
    processes: None | int = None,
    return_df: bool = False,
    log_options: _LogOptions = LOG_DEFAULT,
) -> None | pd.DataFrame:
    """Perform (fast-sigma) COSMO-RS property calculations on the passed SMILES and solvents.

    The output is exported to the ``cosmo-rs.csv`` file.

    Includes the following properties:

    * LogP
    * Activety Coefficient
    * Solvation Energy
    * Formula
    * Molar Mass
    * Nring
    * boilingpoint
    * criticalpressure
    * criticaltemp
    * criticalvol
    * density
    * dielectricconstant
    * entropygas
    * flashpoint
    * gidealgas
    * hcombust
    * hformstd
    * hfusion
    * hidealgas
    * hsublimation
    * meltingpoint
    * molarvol
    * parachor
    * solubilityparam
    * tpt
    * vdwarea
    * vdwvol
    * vaporpressure

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
    log_options : :class:`Mapping[str, Any] <collections.abc.Mapping>`
        Alternative settings for :data:`plams.config.log`.
        See the `PLAMS documentation <https://www.scm.com/doc/plams/components/functions.html#logging>`_ for more details.

    """  # noqa: E501
    # Validation `log_options`
    log_options = dict(log_options)  # type: ignore[assignment]
    illegal_keys = log_options.keys() - {"file", "stdout", "time", "date"}
    if illegal_keys:
        key_str = ", ".join(repr(i) for i in sorted(illegal_keys))
        raise KeyError(f"Invalid `log_options` keys: {key_str}")

    # Parse the `chunk_size`
    chunk_size = operator.index(chunk_size)
    if chunk_size < 1:
        raise ValueError(f"`chunk_size` must be larger than zero; observed value: {chunk_size}")

    # Parse `processes`
    if processes is not None:
        processes = operator.index(processes)
        if processes < 1:
            raise ValueError(f"`processes` must be larger than zero; observed value {processes}")

    # Parse `output_dir`
    output_dir = Path(_abspath(output_dir))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Parse `ams_dir`
    if ams_dir is not None:
        ams_dir = _abspath(ams_dir)

    # Parse the `solvents`
    if len(solvents) == 0:
        raise ValueError("`solvents` requires at least one solvent")
    solvents = cast("dict[str, str]", {k: _abspath(v, True) for k, v in solvents.items()})

    # Construct the dataframe columns
    prop_names = ["Activity Coefficient", "Solvation Energy"]
    _columns: list[tuple[str, None | str]] = [
        ("LogP", None),
        ("Formula", None),
        ("Molar Mass", None),
        ("Nring", None),
        ('boilingpoint', None),
        ('criticalpressure', None),
        ('criticaltemp', None),
        ('criticalvol', None),
        ('density', None),
        ('dielectricconstant', None),
        ('entropygas', None),
        ('flashpoint', None),
        ('gidealgas', None),
        ('hcombust', None),
        ('hformstd', None),
        ('hfusion', None),
        ('hidealgas', None),
        ('hsublimation', None),
        ('meltingpoint', None),
        ('molarvol', None),
        ('parachor', None),
        ('solubilityparam', None),
        ('tpt', None),
        ('vdwarea', None),
        ('vdwvol', None),
        ('vaporpressure', None),
    ]
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
            log=log_options,
        )
        if not return_df:
            ret = None
            for _ in pool.imap_unordered(func, enumerator):
                pass
        else:
            df_idx_list = [i_df for i_df in pool.imap_unordered(func, enumerator)]
            df_idx_list.sort(key=lambda i_df: i_df[0])
            ret = pd.concat(df for _, df in df_idx_list)
    _concatenate_csv(output_dir)
    return ret


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
            df = read_csv(file)
            df.to_csv(f, header=header)
            os.remove(file)


def _read_columns(file: str | bytes | os.PathLike[Any], **kwargs: Any) -> pd.MultiIndex:
    """Extract the dataframe columns from the passed .csv files."""
    kwargs["nrows"] = 0
    df = pd.read_csv(file, header=[0, 1], index_col=0, **kwargs)
    return pd.MultiIndex.from_tuples(
        [(i, (j if j != "nan" else None)) for i, j in df.columns],
        names=df.columns.names,
    )


#: Invalid keyword arguments for :func:`read_csv`.
_INVALID_KWARGS = frozenset({
    "filepath_or_buffer",
    "index_col",
    "header",
    "names",
    "usecols",
})


def read_csv(
    file: str | bytes | os.PathLike[Any],
    *,
    columns: None | Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    r"""Read the passed .csv file as produced by :func:`run_fast_sigma`.

    Examples
    --------
    .. code-block:: python

        >>> from nanoCAT.recipes import read_csv

        >>> file: str = ...

        >>> columns1 = ["molarvol", "gidealgas", "Activity Coefficient"]
        >>> read_csv(file, usecols=columns1)
        property  molarvol  gidealgas Activity Coefficient
        solvent        NaN        NaN              octanol     water
        smiles
        CCCO[H]   0.905952  47.502557          -153.788589  0.078152
        CCO[H]    0.980956  12.735228          -161.094955  0.061220
        CO[H]     1.045891   4.954782                  NaN       NaN

        >>> columns2 = [("Solvation Energy", "water")]
        >>> read_csv(file, usecols=columns2)
        property Solvation Energy
        solvent             water
        smiles
        CCCO[H]         -3.779867
        CCO[H]          -3.883986
        CO[H]           -3.274420

    Parameters
    ----------
    file : :term:`path-like object`
        The name of the to-be opened .csv file.
    columns : key or sequence of keys, optional
        The to-be read columns.
        Note that any passed value must be a valid dataframe (multiindex) key.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :func:`pd.read_csv <pandas.read_csv>`.

    See Also
    --------
    :class:`pd.read_csv <pandas.read_csv>`
        Read a comma-separated values (csv) file into DataFrame.

    """
    # Validate ``kwargs
    if not _INVALID_KWARGS.isdisjoint(kwargs.keys()):
        keys = sorted(_INVALID_KWARGS.intersection(kwargs.keys()))
        raise TypeError(f"Invalid or duplicate keys: {keys}")

    columns_superset = _read_columns(file, **kwargs)
    ref = pd.read_csv(file, index_col=0, skiprows=2, **kwargs)
    ref.columns = columns_superset
    if columns is None:
        df = pd.read_csv(file, index_col=0, skiprows=2, **kwargs)
        df.columns = columns_superset
    else:
        columns_series = pd.Series(np.arange(1, 1 + len(columns_superset)), index=columns_superset)
        columns_idx = np.append(0, columns_series.loc[columns])
        columns_idx2 = columns_idx[1:] - 1

        argsort = np.argsort(columns_idx2)
        df = pd.read_csv(file, usecols=columns_idx, index_col=0, skiprows=2, **kwargs)
        df.sort_index(
            axis=1, inplace=True,
            key=lambda i: i.str.strip("Unnamed: ").astype(np.int64)[argsort],
        )
        df.columns = columns_superset[columns_idx2]

    formula = ("Formula", None)
    if formula in df.columns:
        df.loc[df[formula].isnull(), formula] = ""
    return df


def _canonicalize_smiles(smiles: str) -> None | str:
    """Attempt to canonicalize a **smiles** string."""
    try:
        return CanonSmiles(smiles)
    except Exception as ex:
        warn = RuntimeWarning(f"Failed to canonicalize {smiles!r}")
        warn.__cause__ = ex
        warnings.warn(warn)
        return None


def sanitize_smiles_df(
    df: pd.DataFrame,
    column_levels: SupportsIndex = 2,
    column_padding: Hashable = None,
) -> pd.DataFrame:
    """Sanitize the passed dataframe, canonicalizing the SMILES in its index, converting the columns into a multiIndex and removing duplicate entries.

    Examples
    --------
    .. code-block:: python

        >>> import pandas as pd
        >>> from nanoCAT.recipes import sanitize_smiles_df

        >>> df: pd.DataFrame = ...
        >>> print(df)
                 a
        smiles
        CCCO[H]  1
        CCO[H]   2
        CO[H]    3

        >>> sanitize_smiles_df(df)
                 a
               NaN
        smiles
        CCCO     1
        CCO      2
        CO       3

    Parameters
    ----------
    df : :class:`pd.DataFrame <pandas.DataFrame>`
        The dataframe in question.
        The dataframes' index should consist of smiles strings.
    column_levels : :class:`int`
        The number of multiindex column levels that should be in the to-be returned dataframe.
    column_padding : :class:`~collections.abc.Hashable`
        The object used as padding for the multiindex levels (where appropiate).

    Returns
    -------
    :class:`pd.DataFrame <pandas.DataFrame>`
        The newly sanitized dataframe.
        Returns either the initially passed dataframe or a copy thereof.

    """  # noqa: E501
    # Sanitize `arguments`
    column_levels = operator.index(column_levels)
    if column_levels < 1:
        raise ValueError("`column_levels` must be larger than or equal to 1")
    elif isinstance(df.columns, pd.MultiIndex) and len(df.columns.levels) > column_levels:
        raise ValueError("`column_levels` must be larger than or equal to number "
                         "of MultiIndex levels in `df`")
    elif not isinstance(column_padding, Hashable):
        raise TypeError("`column_padding` expected a hashable object")

    # Sanitize the index
    index = pd.Index(
        [_canonicalize_smiles(i) for i in df.index],
        dtype=df.index.dtype, name=df.index.name,
    )

    # Create or pad a MultiIndex
    padding = (column_levels - 1) * (column_padding,)
    if not isinstance(df.columns, pd.MultiIndex):
        columns = pd.MultiIndex.from_tuples(
            [(i, *padding) for i in df.columns], names=(df.columns.name, *padding)
        )
    elif len(df.columns.levels) < column_levels:
        columns = pd.MultiIndex.from_tuples(
            [(*j, *padding) for j in df.columns], names=(*df.columns.names, *padding)
        )
    else:
        columns = df.columns.copy()

    mask = ~df.index.duplicated(keep='first') & (df.index != None)
    ret = df[mask]
    ret.index = index[mask]
    ret.columns = columns
    return ret
