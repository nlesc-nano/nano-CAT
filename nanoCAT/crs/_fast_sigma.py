from __future__ import annotations

import os
import hashlib
import warnings
import subprocess
from typing import TYPE_CHECKING, TypeVar, Any
from collections.abc import Iterable

import numpy as np

if TYPE_CHECKING:
    _SCT = TypeVar("_SCT", bound=np.generic)
    _NDArray = np.ndarray[Any, np.dtype[_SCT]]

__all__ = ["get_compkf"]


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
        if stderr:
            raise RuntimeError(stderr)
    except (RuntimeError, subprocess.SubprocessError) as ex:
        warn = RuntimeWarning(f"Failed to compute the sigma profile of {smiles!r}")
        warn.__cause__ = ex
        warnings.warn(warn)
        return None
    return abs_file


def _hash_smiles(smiles: str) -> str:
    """Return the sha256 hash of the passed SMILES."""
    return hashlib.sha256(smiles.encode()).hexdigest()


def _get_compkf(
    smiles_iter: Iterable[str],
    directory: None | str | os.PathLike[str],
) -> _NDArray[np.object_]:
    """Wrap :func:`get_compkf` in a for-loop."""
    ret = [get_compkf(smiles, directory, name=_hash_smiles(smiles)) for smiles in smiles_iter]
    return np.array(ret, dtype=np.object_)
