"""
nanoCAT.recipes.charges
=======================

A short recipe for calculating and rescaling ligand charges.

Index
-----
.. currentmodule:: nanoCAT.recipes
.. autosummary::
    get_lig_charge

API
---
.. autofunction:: get_lig_charge

"""

from os import PathLike
from typing import Optional, Union, Iterable

import numpy as np
import pandas as pd

from scm.plams import Molecule, Settings, init, finish
from nanoutils import as_nd_array
from nanoCAT.ff import run_match_job

__all__ = ['get_lig_charge']


def get_lig_charge(ligand: Molecule,
                   desired_charge: float,
                   ligand_idx: Union[None, int, Iterable[int], slice] = None,
                   invert_idx: bool = False,
                   settings: Optional[Settings] = None,
                   path: Union[None, str, PathLike] = None,
                   folder: Union[None, str, PathLike] = None) -> pd.Series:
    """Calculate and rescale the **ligand** charges using MATCH_.

    The atomic charges in **ligand_idx** wil be altered such that the molecular
    charge of **ligand** is equal to **desired_charge**.

    .. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

    Examples
    --------
    .. code:: python

        >>> import pandas as pd
        >>> from scm.plams import Molecule

        >>> from CAT.recipes import get_lig_charge

        >>> ligand = Molecule(...)
        >>> desired_charge = 0.66
        >>> ligand_idx = 0, 1, 2, 3, 4

        >>> charge_series: pd.Series = get_lig_charge(
        ...     ligand, desired_charge, ligand_idx
        ... )

        >>> charge_series.sum() == desired_charge
        True


    Parameters
    ----------
    ligand : :class:`~scm.plams.core.mol.molecule.Molecule`
        The input ligand.

    desired_charge : :class:`float`
        The desired molecular charge of the ligand.

    ligand_idx : :class:`int` or :class:`~collections.abc.Iterable` [:class:`int`], optional
        An integer or iterable of integers representing atomic indices.
        The charges of these atoms will be rescaled;
        all others will be frozen with respect to the MATCH output.
        Setting this value to ``None`` means that *all* atomic charges are considered variable.
        Indices should be 0-based.

    invert_idx : :class:`bool`
        If ``True`` invert **ligand_idx**, *i.e.* all atoms specified therein are
        now threated as constants and the rest as variables,
        rather than the other way around.

    settings : :class:`~scm.plams.core.settings.Settings`, optional
        The input settings for :class:`~nanoCAT.ff.match_job.MatchJob`.
        Will default to the ``"top_all36_cgenff_new"`` forcefield if not specified.

    path : :class:`str` or :class:`~os.PathLike`, optional
        The path to the PLAMS workdir as passed to :func:`~scm.plams.core.functions.init`.
        Will default to the current working directory if ``None``.

    folder : :class:`str` or :class:`~os.PathLike`, optional
        The name of the to-be created to the PLAMS working directory
        as passed to :func:`~scm.plams.core.functions.init`.
        Will default to ``"plams_workdir"`` if ``None``.

    Returns
    -------
    :class:`pd.Series` [:class:`str`, :class:`float`]
        A Series with the atom types of **ligand** as keys and atomic charges as values.

    See Also
    --------
    :class:`MatchJob`
        A :class:`~scm.plams.core.basejob.Job` subclass for interfacing with MATCH_:
        Multipurpose Atom-Typer for CHARMM.

    """  # noqa
    if settings is None:
        settings = Settings()
        settings.input.forcefield = 'top_all36_cgenff_new'

    # Run the MATCH Job
    init(path, folder)
    ligand = ligand.copy()
    run_match_job(ligand, settings, action='raise')
    finish()

    # Extract the charges and new atom types
    count = len(ligand)
    charge = np.fromiter((at.properties.charge_float for at in ligand), count=count, dtype=float)
    symbol = np.fromiter((at.properties.symbol for at in ligand), count=count, dtype='<U4')

    # Identify the atom subset
    idx = _parse_ligand_idx(ligand_idx)
    if invert_idx:
        idx = _invert_idx(idx, count)
    try:
        idx_len = len(idx)  # type: ignore
    except TypeError:  # idx is a slice object
        idx_len = len(charge[idx])

    # Correct the charges and return
    charge[idx] -= (charge.sum() - desired_charge) / idx_len
    return pd.Series(charge, index=symbol, name='charge')


def _parse_ligand_idx(idx: Union[None, slice, int, Iterable[int]]) -> Union[slice, np.ndarray]:
    """Parse the **ligand_idx** parameter in :func:`get_lig_charge`."""
    if idx is None:
        return slice(None)
    elif isinstance(idx, slice):
        return idx
    return as_nd_array(idx, dtype=int)


def _invert_idx(idx: np.ndarray, count: int) -> np.ndarray:
    """Parse the **invert_idx** parameter in :func:`get_lig_charge`."""
    ret = np.ones(count, dtype=bool)
    ret[idx] = False
    return ret
