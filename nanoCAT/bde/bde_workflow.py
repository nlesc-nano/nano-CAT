"""
nanoCAT.bde.bde_workflow
========================

A module with workflows for calculating of Bond Dissociation Energies (BDE).

Index
-----
.. currentmodule:: nanoCAT.bde.bde_workflow
.. autosummary::
    init_bde
    _bde_w_dg
    _bde_wo_dg
    _qd_to_db
    get_recipe
    get_bde_dE
    get_bde_ddG

API
---
.. autofunction:: init_bde
.. autofunction:: _bde_w_dg
.. autofunction:: _bde_wo_dg
.. autofunction:: _qd_to_db
.. autofunction:: get_recipe
.. autofunction:: get_bde_dE
.. autofunction:: get_bde_ddG

"""

from typing import Iterable, Type, List, Tuple, Optional
from itertools import product

import numpy as np

from scm.plams import AMSJob, Molecule, Settings, Cp2kJob
from scm.plams.core.basejob import Job

from CAT.jobs import job_single_point, job_geometry_opt, job_freq  # noqa: F401
from CAT.logger import logger
from CAT.settings_dataframe import SettingsDataFrame
from CAT.workflows import WorkFlow, MOL, JOB_SETTINGS_BDE

from .construct_xyn import get_xyn
from .dissociate_xyn import dissociate_ligand
from ..qd_opt_ff import qd_opt_ff

__all__ = ['init_bde']


def init_bde(qd_df: SettingsDataFrame) -> None:
    """Initialize the ligand dissociation workflow."""
    # import pdb; pdb.set_trace()
    workflow = WorkFlow.from_template(qd_df, name='bde')

    # Create columns
    columns = _construct_columns(workflow, qd_df[MOL])
    import_columns = {(i, j): (np.nan if i != 'label' else None) for i, j in columns}

    # Pull from the database; push unoptimized structures
    idx = workflow.from_db(qd_df, columns=import_columns)
    workflow(start_bde, qd_df, columns=columns, index=idx, workflow=workflow)

    # Convert the datatype from object back to float
    qd_df['BDE dE'] = qd_df['BDE dE'].astype(float, copy=False)
    if workflow.jobs[1]:
        qd_df['BDE ddG'] = qd_df['BDE ddG'].astype(float, copy=False)
        qd_df['BDE dG'] = qd_df['BDE dG'].astype(float, copy=False)

    # Sets a nested list with the filenames of .in files
    # This cannot be done with loc is it will try to expand the list into a 2D array
    qd_df[JOB_SETTINGS_BDE] = workflow.pop_job_settings(qd_df[MOL])

    # Push the optimized structures to the database
    job_recipe = workflow.get_recipe()
    workflow.to_db(qd_df, index=idx, columns=columns, job_recipe=job_recipe)


def _construct_columns(workflow: WorkFlow, mol_list: Iterable[Molecule]) -> List[Tuple[str, str]]:
    """Construct BDE columns for :func:`init_bde`."""
    if workflow.core_index:
        stop = len(workflow.core_index)
    else:  # This takes a but longer, unfortunetly
        try:
            mol = next(mol_list)
        except TypeError:
            mol = next(iter(mol_list))

        qd_iterator = dissociate_ligand(mol, **vars(workflow))
        for stop, _ in enumerate(qd_iterator, 1):
            pass

    super_keys = ('BDE label', 'BDE dE')
    if workflow.jobs[1]:  # i.e. thermochemical corrections are enabled
        super_keys += ('BDE ddG', 'BDE dG')

    sub_keys = np.arange(stop).astype(dtype=str)
    return list(product(super_keys, sub_keys))


def start_bde(mol_list: Iterable[Molecule],
              jobs: Tuple[Type[Job], ...], settings: Tuple[Settings, ...],
              forcefield=None, lig_count=None, core_atom=None, **kwargs) -> List[np.ndarray]:
    """Calculate the BDEs with thermochemical corrections."""
    job1, job2 = jobs
    s1, s2 = settings

    ret = []
    ret_append = ret.append
    for qd_complete in mol_list:
        # Dissociate a XYn molecule from the quantum dot surface
        qd_complete.round_coords()
        XYn: Molecule = get_xyn(qd_complete, lig_count, core_atom)

        # Create all possible quantum dots where XYn is dissociated
        qd_list: List[Molecule] = list(dissociate_ligand(
            qd_complete, lig_count, core_atom=core_atom, **kwargs
        ))

        # Construct labels describing the topology of all XYn-dissociated quantum dots
        labels = [qd.properties.df_index for qd in qd_list]

        # Run the BDE calculations
        dE = get_bde_dE(qd_complete, XYn, qd_list, job1, s1, forcefield)
        ddG = get_bde_ddG(qd_complete, XYn, qd_list, job2, s2)
        dG = dE + ddG

        # Append the to-be returned list
        value = np.concatenate([labels, dE, ddG, dG] if job2 else [labels, dE])
        ret_append(value)  # value is now, unfortunetly, a str array

        # Update the list with all .in files
        try:
            qd_complete.properties.job_path += XYn.properties.pop('job_path')
        except IndexError:
            qd_complete.properties.job_path = XYn.properties.pop('job_path')
        for mol in qd_list:
            qd_complete.properties.job_path += mol.properties.pop('job_path')

    return ret


def get_bde_dE(tot: Molecule, lig: Molecule, core: Iterable[Molecule],
               job: Type[Job], s: Settings, forcefield: bool = False) -> np.ndarray:
    """Calculate the bond dissociation energy: dE = dE(mopac) + (dG(uff) - dE(uff)).

    Parameters
    ----------
    tot : |plams.Molecule|_
        The complete intact quantum dot.

    lig : |plams.Molecule|_
        A ligand dissociated from the surface of the quantum dot

    core : |list|_ [|plams.Molecule|_]
        A list with one or more quantum dots (*i.e.* **tot**) with **lig** removed.

    job : |plams.Job|_
        A :class:`.Job` subclass.

    s : |plams.Settings|_
        The settings for **job**.

    """
    # Optimize XYn
    len_core = len(core)
    if job is AMSJob:
        s_cp = Settings(s)
        s_cp.input.ams.GeometryOptimization.coordinatetype = 'Cartesian'
        lig.job_geometry_opt(job, s_cp, name='E_XYn_opt')
    elif forcefield:
        qd_opt_ff(lig, Settings({'job1': Cp2kJob, 's1': s}), name='E_XYn_opt')
    else:
        lig.job_geometry_opt(job, s, name='E_XYn_opt')

    E_lig = lig.properties.energy.E
    if E_lig in (None, np.nan):
        logger.error('The BDE XYn geometry optimization failed, skipping further jobs')
        return np.full(len_core, np.nan)

    # Perform a single point on the full quantum dot
    if forcefield:
        qd_opt_ff(tot, Settings({'job1': Cp2kJob, 's1': s}), name='E_QD_opt')
    else:
        tot.job_single_point(job, s, name='E_QD_sp')

    E_tot = tot.properties.energy.E
    if E_tot in (None, np.nan):
        logger.error('The BDE quantum dot single point failed, skipping further jobs')
        return np.full(len_core, np.nan)

    # Perform a single point on the quantum dot(s) - XYn
    for mol in core:
        if forcefield:
            qd_opt_ff(mol, Settings({'job1': Cp2kJob, 's1': s}), name='E_QD-XYn_opt')
        else:
            mol.job_single_point(job, s, name='E_QD-XYn_sp')
    E_core = np.fromiter([mol.properties.energy.E for mol in core], count=len_core, dtype=float)

    # Calculate and return dE
    dE = (E_lig + E_core) - E_tot
    return dE


def get_bde_ddG(tot: Molecule, lig: Molecule, core: Iterable[Molecule],
                job: Optional[Type[Job]] = None, s: Optional[Settings] = None) -> np.ndarray:
    """Calculate the bond dissociation energy: dG = dE(lvl1) + (dG(lvl2) - dE(lvl2)).

    Parameters
    ----------
    tot : |plams.Molecule|
        The complete intact quantum dot.

    lig : |plams.Molecule|
        A ligand dissociated from the surface of the quantum dot

    core : :class:`Iterable<collections.abc.Iterable>` [|plams.Molecule|]
        An iterable with one or more quantum dots (*i.e.* **tot**) with **lig** removed.

    job : |plams.Job|
        A :class:`.Job` subclass.
        The dG will be skipped if ``None``.

    s : |plams.Settings|
        The settings for **job**.
        The dG will be skipped if ``None``.

    """
    # The calculation of dG has been disabled by the user
    len_core = len(core)
    if job is None:
        return np.full(len_core, np.nan)

    # Optimize XYn
    s.input.ams.Constraints.Atom = lig.properties.indices
    lig.job_freq(job, s, name='G_XYn_freq')

    # Extract energies
    G_lig = lig.properties.energy.G
    E_lig = lig.properties.energy.E
    if np.nan in (E_lig, G_lig):
        logger.error('The BDE XYn geometry optimization+frequency analysis failed, '
                     'skipping further jobs')
        return np.full(len_core, np.nan)

    # Optimize the full quantum dot
    s.input.ams.Constraints.Atom = tot.properties.indices
    tot.job_freq(job, s, name='G_QD_freq')

    # Extract energies
    G_tot = tot.properties.energy.G
    E_tot = tot.properties.energy.E
    if np.nan in (E_tot, G_tot):
        logger.error('The BDE quantum dot geometry optimization+frequency analysis failed, '
                     'skipping further jobs')
        return np.full(len_core, np.nan)

    # Optimize the quantum dot(s) - XYn
    for mol in core:
        s.input.ams.Constraints.Atom = mol.properties.indices
        mol.job_freq(job, s, name='G_QD-XYn_freq')

    # Extract energies
    G_core = np.fromiter([mol.properties.energy.G for mol in core], count=len_core, dtype=float)
    E_core = np.fromiter([mol.properties.energy.E for mol in core], count=len_core, dtype=float)

    # Calculate and return dG and ddG
    dG = (G_lig + G_core) - G_tot
    dE = (E_lig + E_core) - E_tot
    ddG = dG - dE
    return ddG
