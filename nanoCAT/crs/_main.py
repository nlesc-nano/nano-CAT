from __future__ import annotations

import os
import sys
import types
import operator
import functools
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload, cast
from collections.abc import Mapping, Iterable

import pandas as pd
from more_itertools import chunked

from ._csv_utils import _concatenate_csv
from ._fast_sigma import _get_compkf

if TYPE_CHECKING:
    from .templates import _JobKeys

    if sys.version_info >= (3, 8):
        from typing import Literal, TypedDict
    else:
        from typing_extensions import Literal, TypedDict

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

__all__ = ["run_fast_sigma"]

# The default PLAMS `config.log` options
LOG_DEFAULT: _LogOptions = types.MappingProxyType({  # type: ignore[assignment]
    "file": 5,
    "stdout": 3,
    "time": True,
    "date": False,
})


def _abspath(path: str | bytes | os.PathLike[Any], isfile: bool = False) -> str:
    """Path sanitizing."""
    ret = os.path.abspath(os.path.expandvars(os.fsdecode(path)))
    if isfile and not os.path.isfile(ret):
        open(ret, "r")  # This will raise
        raise
    return ret


@overload
def run_fast_sigma(
    input_smiles: Iterable[str],
    solvents: Mapping[str, str | bytes | os.PathLike[Any]],
    jobs: Mapping[_JobKeys, Iterable[str]],
    *,
    settings: None | Mapping[_JobKeys, dict[str, Any]] = ...,
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
    jobs: Mapping[_JobKeys, Iterable[str]],
    *,
    settings: None | Mapping[_JobKeys, dict[str, Any]] = ...,
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
    jobs: Mapping[_JobKeys, Iterable[str]],
    *,
    settings: None | Mapping[_JobKeys, dict[str, Any]] = None,
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

    * Boiling Point
    * LogP
    * Activety Coefficient
    * Solvation Energy
    * Vapor Pressure
    * Enthalpy of Vaporization
    * Volume
    * Area
    * Formula
    * Molar Mass
    * Nring

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
        ("Boiling Point", None),
        ("Vapor Pressure", None),
        ("Enthalpy of Vaporization", None),
        ("LogP", None),
        ("Volume", None),
        ("Area", None),
        ("Formula", None),
        ("Molar Mass", None),
        ("Nring", None),
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
