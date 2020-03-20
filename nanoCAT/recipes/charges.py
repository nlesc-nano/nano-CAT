"""
nanoCAT.recipes.charges
=======================

A short recipe for calculating and rescaling ligand charges.

Index
-----
.. currentmodule:: nanoCAT.recipes.charges
.. autosummary::
    get_lig_charge

API
---
.. autofunction:: get_lig_charge

"""

from os import PathLike
from typing import Optional, Union

import pandas as pd
from scm.plams import Molecule, Atom, Settings, MoleculeError, JobError, init, finish

from nanoCAT.ff import run_match_job

__all__ = ['get_lig_charge']


def get_lig_charge(ligand: Molecule,
                   ligand_anchor: int,
                   core_anchor_charge: float,
                   settings: Optional[Settings] = None,
                   path: Union[None, str, PathLike] = None,
                   folder: Union[None, str, PathLike] = None) -> pd.Series:
    """Calculate and rescale the **ligand** charges using MATCH_.

    The atomic charge of **ligand_anchor**, as deterimined by MATCH, is rescaled such that
    the molecular charge of **ligand** will be equal to **core_anchor_charge**.

    .. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

    Examples
    --------
    .. code:: python

        >>> import pandas as pd
        >>> from scm.plams import Molecule

        >>> from CAT.recipes import get_lig_charge

        >>> ligand = Molecule(...)
        >>> ligand_anchor = int(...)  # 1-based index
        >>> core_anchor_charge = float(...)

        >>> charge_series: pd.Series = get_lig_charge(
        ...     ligand, ligand_anchor, core_anchor_charge
        ... )

        >>> charge_series.sum() == core_anchor_charge
        True


    Parameters
    ----------
    ligand : :class:`~scm.plams.core.mol.molecule.Molecule`
        The input ligand.

    ligand_anchor : :class:`int`
        The (1-based) atomic index of the ligand anchor atom.
        The charge of this atom will be rescaled.

    core_anchor_charge : :class:`float`
        The atomic charge of the core anchor atoms.

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
    run_match_job(ligand, settings, action='raise')
    finish()

    # Extract the charges and new atom types
    charge = [at.properties.charge_float for at in ligand]
    symbol = [at.properties.symbol for at in ligand]

    # Correct the charge of ligand[i]
    i = ligand_anchor
    charge[i - 1] -= sum(charge) - core_anchor_charge
    ligand[i].properties.charge_float = charge[i - 1]

    return pd.Series(charge, index=symbol, name='charge')
