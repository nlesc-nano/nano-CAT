"""
CAT.attachment.qd_opt_ff
========================

A module designed for optimizing the combined ligand & core using with forcefields in CP2K.

Index
-----
.. currentmodule:: CAT.attachment.qd_opt_ff
.. autosummary::
    qd_opt_ff
    get_psf
    finalize_lj
    _constrain_charge
    _gather_core_lig_symbols

API
---
.. autofunction:: qd_opt_ff
.. autofunction:: get_psf
.. autofunction:: _constrain_charge
.. autofunction:: finalize_lj
.. autofunction:: _gather_core_lig_symbols

"""

import os
import warnings
from functools import wraps
from collections import abc
from typing import (
    Collection, Iterable, Union, Dict, Tuple, List, Optional, Type, Callable
)

import numpy as np
import pandas as pd

from scm.plams import Molecule, Settings
from scm.plams.core.basejob import Job
from scm.plams.core.results import Results

from FOX import PSFContainer
from FOX.ff.lj_uff import combine_sigma, combine_epsilon

from .ff.cp2k_utils import set_cp2k_element

__all__ = ['qd_opt_ff']


def qd_opt_ff(mol: Molecule, jobs: Tuple[Optional[Type[Job]], ...],
              settings: Tuple[Optional[Settings], ...], name: str = 'QD_opt',
              new_psf: bool = False, job_func: Callable = Molecule.job_geometry_opt) -> Results:
    """Alternative implementation of :func:`.qd_opt` using CP2Ks' classical forcefields.

    Performs an inplace update of **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        The to-be optimized molecule.

    jobs : :class:`tuple`
        A tuple of |plams.Job| types and/or ``None``.

    settings : :class:`tuple`
        A tuple of |plams.Settings| types and/or ``None``.

    name : str
        The name of the job.

    See Also
    --------
    :func:`CAT.attachment.qd_opt.qd_opt`
        Default workflow for optimizing molecules.

    """
    psf_name = os.path.join(mol.properties.path, mol.properties.name + '.psf')

    # Prepare the job settings
    job = jobs[0] if isinstance(jobs, abc.Sequence) else jobs
    s = Settings(settings[0]) if isinstance(settings, abc.Sequence) else Settings(settings)

    s.runscript.pre = (f'ln "{psf_name}" ./"{name}.psf"\n'
                       f'ln "{mol.properties.prm}" ./"{name}.prm"')
    s.input.force_eval.subsys.topology.conn_file_name = f'{name}.psf'
    s.input.force_eval.mm.forcefield.parm_file_name = f'{name}.prm'
    set_cp2k_element(s, mol)

    if not os.path.isfile(psf_name) or new_psf:
        psf = get_psf(mol, s.input.force_eval.mm.forcefield.get('charge', None))
        for at, charge, symbol in zip(mol, psf.charge, psf.atom_type):
            at.properties.charge_float = charge
            at.properties.symbol = symbol
        psf.write(psf_name)

    # Pull any missing non-covalent parameters from UFF
    if s.input.force_eval.mm.forcefield.nonbonded.get('lennard-jones', None):
        try:
            finalize_lj(mol, s.input.force_eval.mm.forcefield.nonbonded['lennard-jones'])
        except TypeError:
            pass
    results = job_func(mol, job, s, name=name, read_template=False, ret_results=True)
    mol.round_coords()

    return results


def get_psf(mol: Molecule, charges: Union[None, Settings, Iterable[Settings]]) -> PSFContainer:
    """Construct and return a :class:`PSF` instance.

    .. _CHARGE: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/MM/FORCEFIELD/CHARGE.html

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule which will be used for constructing the :class:`PSFContainer` instance.

    s : |list|_ [|plams.Settings|_] or |plams.Settings|_
        A list of settings constructed from the CP2K FORCE_EVAL/MM/FORCEFIELD/CHARGE_ block.
        The settings are expected to contain the ``"charge"`` and ``"atom"`` keys.

    """
    # Construct a PSF instance
    psf = PSFContainer()
    psf.generate_bonds(mol)
    psf.generate_angles(mol)
    psf.generate_dihedrals(mol)
    psf.generate_impropers(mol)
    psf.generate_atoms(mol)

    # Update charges based on charges which have been explictly specified by the user
    initial_charge = psf.charge.sum()
    if isinstance(charges, Settings):
        charge_dict = {charges.atom: charges.charge}
    elif isinstance(charges, abc.Iterable):
        charge_dict = {i.atom: i.charge for i in charges}
    elif charges is None:
        return psf
    else:
        raise TypeError(f"The parameter 'charges' is of invalid type: {repr(type(charges))}")

    for at, charge in charge_dict.items():
        psf.update_atom_charge(at, float(charge))

    # Update atomic charges in order to reset the molecular charge to its initial value
    constrain_charge(psf, initial_charge, charge_dict)
    return psf


def constrain_charge(psf: PSFContainer, initial_charge: float = 0.0,
                     atom_set: Optional[Collection[str]] = None) -> None:
    """Set the total molecular charge of **psf** to **initial_charge**.

    Atoms in **psf** whose atomic symbol intersects with **charge_set** will *not*
    be altered.

    Parameters
    ----------
    psf : |nanoCAT.PSFContainer|_
        A :class:`.PSF` instance with newly updated charges and/or atom types.

    initial_charge : float
        The initial charge of the system before the updating of atomic charges.

    atom_set : |Container|_ [|str|_]
        A container with atomic symbols.
        Any atom in **psf** whose atomic symbol intersects with **charge_set** will *not*
        be altered.

    """
    # Check if the molecular charge has remained unchanged
    new_charge = psf.charge.sum()
    if (abs(new_charge) - abs(initial_charge)) < 10**-8:  # i.e. 0.0 +- 10**-8
        return psf

    # Update atomic charges in order to reset the molecular charge to its initial value
    if atom_set is None:
        atom_subset = np.ones(len(psf.atoms), dtype=bool)
    else:
        atom_set_ = set(atom_set)
        atom_subset = np.array([at not in atom_set_ for at in psf.atom_type])

    charge_correction = initial_charge - psf.charge.sum()
    charge_correction /= np.count_nonzero(atom_subset)
    with pd.option_context('mode.chained_assignment', None):
        psf.charge[atom_subset] += charge_correction


@wraps(constrain_charge)
def _constrain_charge(*args, **kwargs):
    msg = "_constrain_charge() is deprecated; use constrain_charge() from now on"
    warnings.warn(msg, category=DeprecationWarning)
    return constrain_charge(*args, **kwargs)


def finalize_lj(mol: Molecule, s: List[Settings]) -> None:
    """Assign UFF Lennard-Jones parameters to all missing non-bonded core/ligand interactions.

    .. _LENNARD_JONES: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/MM/FORCEFIELD/NONBONDED/LENNARD-JONES.html

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule containing a core and ligand(s).

    s : |list|_ [|plams.Settings|_]
        A list of settings constructed from the
        CP2K FORCE_EVAL/MM/FORCEFIELD/NONBONDED/`LENNARD-JONES`_ block.
        The settings are expected to contain the ``"atoms"`` keys.

    """  # noqa
    # Create a set of all core atom types
    core_at, lig_at = _gather_core_lig_symbols(mol)

    # Create a set of all user-specified core/ligand LJ pairs
    if not s:
        s = []
    elif isinstance(s, dict):
        s = [s]
    atom_pairs = {frozenset(s.atoms.split()) for s in s}

    # Check if LJ parameters are present for all atom pairs.
    # If not, supplement them with UFF parameters.
    for at1, symbol1 in core_at.items():
        for at2, symbol2 in lig_at.items():
            at1_at2 = {at1, at2}
            if at1_at2 in atom_pairs:
                continue

            s.append(Settings({
                'atoms': f'{at1} {at2}',
                'epsilon': f'[kcalmol] {round(combine_epsilon(symbol1, symbol2), 4)}',
                'sigma': f'[angstrom] {round(combine_sigma(symbol1, symbol2), 4)}'
            }))


def _gather_core_lig_symbols(mol: Molecule) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Create two dictionaries with atom types and atomic symbols.

    Both dictionaries contain atomic symbols as keys and matching atom types as values;
    dictionary #1 for the core and #2 for the ligand(s).

    Cores (``"COR"``) and ligands (``"LIG"``) are distinguished based on the value of each atoms'
    :attr:`Atom.properties` ``["pdb_info"]["ResidueName"]`` attribute.

    """
    iterator = iter(mol)
    core_at = {}
    lig_at = {}

    # Fill the set with all core atom types
    for at in iterator:  # Iterate until the first ligand is encountered
        if at.properties.pdb_info.ResidueName != 'COR':
            break
        if at.symbol not in core_at:
            core_at[at.symbol] = at.symbol

    # Fill the set with all ligand atom types
    res_number = at.properties.pdb_info.ResidueNumber
    lig_at[at.properties.symbol] = at.symbol
    for at in iterator:  # Iterate through a single ligand until the next ligand is encountered
        if at.properties.pdb_info.ResidueNumber != res_number:
            break
        if at.symbol not in lig_at:
            lig_at[at.properties.symbol] = at.symbol

    return core_at, lig_at
