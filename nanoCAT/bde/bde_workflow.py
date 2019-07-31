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

from shutil import rmtree
from typing import (Callable, Optional, Iterable)
from os.path import join
from itertools import product

import numpy as np
import pandas as pd

from scm.plams import AMSJob
from scm.plams.mol.molecule import Molecule
from scm.plams.core.functions import finish
from scm.plams.core.settings import Settings

import qmflows

from CAT.logger import logger
from CAT.jobs import (job_single_point, job_geometry_opt, job_freq)
from CAT.utils import (type_to_string, restart_init)
from CAT.mol_utils import round_coords
from CAT.settings_dataframe import SettingsDataFrame

from .construct_xyn import get_xyn
from .dissociate_xyn import (dissociate_ligand, dissociate_ligand2)

__all__ = ['init_bde']

# Aliases for pd.MultiIndex columns
MOL = ('mol', '')
JOB_SETTINGS_BDE = ('job_settings_BDE', '')
SETTINGS1 = ('settings', 'BDE 1')
SETTINGS2 = ('settings', 'BDE 2')


def init_bde(qd_df: SettingsDataFrame) -> None:
    r"""Initialize the bond dissociation energy calculation; involves 4 distinct steps.

    * Take :math:`n` ligands (:math:`X`) and another atom from the core (:math:`Y`, *e.g.* Cd)
      and create :math:`YX_{n}`.
    * Given a radius :math:`r`, dissociate all possible :math:`YX_{n}` pairs.
    * Calculate :math:`\Delta E`: the "electronic" component of the bond dissociation energy (BDE).
    * (Optional) Calculate :math:`\Delta \Delta G`: the thermal and entropic component of the BDE.

    Parameters
    ----------
    qd_df : |CAT.SettingsDataFrame|_
        A dataframe of quantum dots.

    """
    # Unpack arguments
    settings = qd_df.settings.optional
    db = settings.database.db
    overwrite = db and 'qd' in settings.database.overwrite
    read = db and 'qd' in settings.database.read
    job2 = settings.qd.dissociate.job2
    s2 = settings.qd.dissociate.s2

    # Check if the calculation has been done already
    if not overwrite and read:
        logger.info('Pulling ligand dissociation energies from the database')
        with db.csv_qd.open(write=False) as db_df:
            key_ar = np.array(['BDE label', 'BDE dE', 'BDE dG', 'BDE ddG'])
            bool_ar = np.isin(key_ar, db_df.columns.levels[0])
            for i in db_df[key_ar[bool_ar]]:
                qd_df[i] = np.nan
            db.from_csv(qd_df, database='QD', get_mol=False)
        qd_df.dropna(axis='columns', how='all', inplace=True)

    # Calculate the BDEs with thermochemical corrections
    if job2 and s2:
        _bde_w_dg(qd_df)

    # Calculate the BDEs without thermochemical corrections
    else:
        _bde_wo_dg(qd_df)


def _bde_w_dg(qd_df: SettingsDataFrame) -> None:
    """Calculate the BDEs with thermochemical corrections.

    Parameters
    ----------
    qd_df : |CAT.SettingsDataFrame|_
        A dataframe of quantum dots.

    """
    # Unpack arguments
    settings = qd_df.settings.optional
    keep_files = settings.qd.keep_files
    path = settings.qd.dirname
    job1 = settings.qd.dissociate.job1
    job2 = settings.qd.dissociate.job2
    s1 = settings.qd.dissociate.s1
    s2 = settings.qd.dissociate.s2
    ion = settings.qd.dissociate.core_atom
    lig_count = settings.qd.dissociate.lig_count
    core_index = settings.qd.dissociate.core_index
    write = settings.database.db and 'qd' in settings.database.write

    # Identify previously calculated results
    try:
        has_na = qd_df[['BDE dE', 'BDE dG']].isna().all(axis='columns')
        if not has_na.any():
            logger.info('No new ligand dissociation jobs found\n')
            return
        logger.info('Starting ligand dissociation workflow')
    except KeyError:
        has_na = pd.Series(True, index=qd_df.index)

    df_slice = qd_df.loc[has_na, MOL]
    restart_init(path=path, folder='BDE')
    for idx, mol in df_slice.iteritems():
        # Create XYn and all XYn-dissociated quantum dots
        mol.round_coords()
        xyn = get_xyn(mol, lig_count, ion)
        if not core_index:
            mol_wo_xyn = dissociate_ligand(mol, settings)
        else:
            mol_wo_xyn = dissociate_ligand2(mol, settings)

        # Construct new columns for **qd_df**
        labels = [m.properties.df_index for m in mol_wo_xyn]
        sub_idx = np.arange(len(labels)).astype(str, copy=False)
        try:
            n = qd_df['BDE label'].shape[1]
        except KeyError:
            n = 0
        if len(labels) > n:
            for i in sub_idx[n:]:
                qd_df[('BDE label', i)] = qd_df[('BDE dE', i)] = qd_df[('BDE ddG', i)] = np.nan

        # Prepare slices
        label_slice = idx, list(product(['BDE label'], sub_idx))
        dE_slice = idx, list(product(['BDE dE'], sub_idx))
        ddG_slice = idx, list(product(['BDE ddG'], sub_idx))

        # Run the BDE calculations
        mol.properties.job_path = []
        qd_df.loc[label_slice] = labels
        qd_df.loc[dE_slice] = get_bde_dE(mol, xyn, mol_wo_xyn, job=job1, s=s1)
        qd_df.loc[ddG_slice] = get_bde_ddG(mol, xyn, mol_wo_xyn, job=job2, s=s2)
        mol.properties.job_path += xyn.properties.pop('job_path')
        for m in mol_wo_xyn:
            mol.properties.job_path += m.properties.pop('job_path')
    logger.info('Finishing ligand dissociation workflow\n')
    finish()
    if not keep_files:
        rmtree(join(path, 'BDE'))

    qd_df['BDE dG'] = qd_df['BDE dE'] + qd_df['BDE ddG']

    job_settings = []
    for mol in qd_df[MOL]:
        try:
            job_settings.append(mol.properties.pop('job_path'))
        except KeyError:
            job_settings.append([])
    qd_df[JOB_SETTINGS_BDE] = job_settings

    # Update the database
    if write:
        with pd.option_context('mode.chained_assignment', None):
            _qd_to_db(qd_df, has_na, with_dg=True)


def _bde_wo_dg(qd_df: SettingsDataFrame) -> None:
    """ Calculate the BDEs without thermochemical corrections.

    Parameters
    ----------
    qd_df : |CAT.SettingsDataFrame|_
        A dataframe of quantum dots.

    """
    # Unpack arguments
    settings = qd_df.settings.optional
    keep_files = settings.qd.keep_files
    path = settings.qd.dirname
    job1 = settings.qd.dissociate.job1
    s1 = settings.qd.dissociate.s1
    ion = settings.qd.dissociate.core_atom
    lig_count = settings.qd.dissociate.lig_count
    core_index = settings.qd.dissociate.core_index
    write = settings.database.db and 'qd' in settings.database.write

    # Identify previously calculated results
    try:
        has_na = qd_df['BDE dE'].isna().all(axis='columns')
        if not has_na.any():
            logger.info('No new ligand dissociation jobs found\n')
            return
        logger.info('Starting ligand dissociation workflow')
    except KeyError:
        has_na = pd.Series(True, index=qd_df.index)

    df_slice = qd_df.loc[has_na, MOL]
    restart_init(path=path, folder='BDE')
    for idx, mol in df_slice.iteritems():
        # Create XYn and all XYn-dissociated quantum dots
        mol.round_coords()
        xyn = get_xyn(mol, lig_count, ion)

        if not core_index:
            mol_wo_xyn = dissociate_ligand(mol, settings)
        else:
            mol_wo_xyn = dissociate_ligand2(mol, settings)

        # Construct new columns for **qd_df**
        labels = [m.properties.df_index for m in mol_wo_xyn]
        sub_idx = np.arange(len(labels)).astype(str)
        try:
            n = qd_df['BDE label'].shape[1]
        except KeyError:
            n = 0
        if len(labels) > n:
            for i in sub_idx[n:]:
                qd_df[('BDE label', i)] = qd_df[('BDE dE', i)] = np.nan

        # Prepare slices
        label_slice = idx, list(product(['BDE label'], sub_idx))
        dE_slice = idx, list(product(['BDE dE'], sub_idx))

        # Run the BDE calculations
        mol.properties.job_path = []
        qd_df.loc[label_slice] = labels
        qd_df.loc[dE_slice] = get_bde_dE(mol, xyn, mol_wo_xyn, job=job1, s=s1)
        mol.properties.job_path += xyn.properties.pop('job_path')
        for m in mol_wo_xyn:
            mol.properties.job_path += m.properties.pop('job_path')
    logger.info('Finishing ligand dissociation workflow\n')
    finish()
    if not keep_files:
        rmtree(join(path, 'BDE'))

    job_settings = []
    for mol in qd_df[MOL]:
        try:
            job_settings.append(mol.properties.pop('job_path'))
        except KeyError:
            job_settings.append([])
    qd_df[JOB_SETTINGS_BDE] = job_settings

    # Update the database
    if write:
        with pd.option_context('mode.chained_assignment', None):
            _qd_to_db(qd_df, has_na, with_dg=False)


def _qd_to_db(qd_df: SettingsDataFrame,
              idx: pd.Series,
              with_dg: bool = True) -> None:
    # Unpack arguments
    settings = qd_df.settings.optional
    db = settings.database.db
    overwrite = db and 'qd' in settings.database.overwrite
    j1 = settings.qd.dissociate.job1
    s1 = settings.qd.dissociate.s1

    qd_df.sort_index(axis='columns', inplace=True)
    kwarg = {'database': 'QD', 'overwrite': overwrite}
    if with_dg:
        j2 = settings.qd.dissociate.job2
        s2 = settings.qd.dissociate.s2
        kwarg['job_recipe'] = get_recipe(j1, s1, j2, s2)
        kwarg['columns'] = [JOB_SETTINGS_BDE, SETTINGS1, SETTINGS2]
        column_tup = ('BDE label', 'BDE dE', 'BDE ddG', 'BDE dG')
    else:
        kwarg['job_recipe'] = get_recipe(j1, s1)
        kwarg['columns'] = [JOB_SETTINGS_BDE, SETTINGS1]
        column_tup = ('BDE label', 'BDE dE')
    kwarg['columns'] += [(i, j) for i, j in qd_df.columns if i in column_tup]

    db.update_csv(qd_df[idx], **kwarg)


def get_recipe(job1: Callable,
               s1: Settings,
               job2: Optional[Callable] = None,
               s2: Optional[Callable] = None) -> Settings:
    """Return the a dictionary with job types and job settings."""
    ret = Settings()
    value1 = qmflows.singlepoint['specific'][type_to_string(job1)].copy()
    value1.update(s1)
    ret['BDE 1'] = {'key': job1, 'value': value1}

    if job2 is not None and s2 is not None:
        value2 = qmflows.freq['specific'][type_to_string(job2)].copy()
        value2.update(s2)
        ret['BDE 2'] = {'key': job2, 'value': value2}

    return ret


def get_bde_dE(tot: Molecule,
               lig: Molecule,
               core: Iterable[Molecule],
               job: Callable,
               s: Settings) -> np.ndarray:
    """Calculate the bond dissociation energy: dE = dE(mopac) + (dG(uff) - dE(uff))."""
    # Optimize XYn
    if job == AMSJob:
        s_cp = s.copy()
        s_cp.input.ams.GeometryOptimization.coordinatetype = 'Cartesian'
        lig.job_geometry_opt(job, s_cp, name='E_XYn_opt')
    else:
        lig.job_geometry_opt(job, s, name='E_XYn_opt')

    E_lig = lig.properties.energy.E
    if E_lig is np.nan:
        logger.error('The BDE XYn geometry optimization failed, skipping further jobs')
        return np.full(len(core), np.nan)

    # Perform a single point on the full quantum dot
    tot.job_single_point(job, s, name='E_QD_sp')
    E_tot = tot.properties.energy.E
    if E_tot is np.nan:
        logger.error('The BDE quantum dot single point failed, skipping further jobs')
        return np.full(len(core), np.nan)

    # Perform a single point on the quantum dot(s) - XYn
    for mol in core:
        mol.job_single_point(job, s, name='E_QD-XYn_sp')
    E_core = np.array([mol.properties.energy.E for mol in core])

    # Calculate and return dE
    dE = (E_lig + E_core) - E_tot
    return dE


def get_bde_ddG(tot: Molecule,
                lig: Molecule,
                core: Iterable[Molecule],
                job: Callable,
                s: Settings) -> np.ndarray:
    """Calculate the bond dissociation energy: dE = dE(mopac) + (dG(uff) - dE(uff))."""
    # Optimize XYn
    s.input.ams.Constraints.Atom = lig.properties.indices
    lig.job_freq(job, s, name='G_XYn_freq')

    # Extract energies
    G_lig = lig.properties.energy.G
    E_lig = lig.properties.energy.E
    if np.nan in (E_lig, G_lig):
        logger.error('The BDE XYn geometry optimization+frequency analysis failed, '
                     'skipping further jobs')
        return np.full(len(core), np.nan)

    # Optimize the full quantum dot
    s.input.ams.Constraints.Atom = tot.properties.indices
    tot.job_freq(job, s, name='G_QD_freq')

    # Extract energies
    G_tot = tot.properties.energy.G
    E_tot = tot.properties.energy.E
    if np.nan in (E_tot, G_tot):
        logger.error('The BDE quantum dot geometry optimization+frequency analysis failed, '
                     'skipping further jobs')
        return np.full(len(core), np.nan)

    # Optimize the quantum dot(s) - XYn
    for mol in core:
        s.input.ams.Constraints.Atom = mol.properties.indices
        mol.job_freq(job, s, name='G_QD-XYn_freq')

    # Extract energies
    G_core = np.array([mol.properties.energy.G for mol in core])
    E_core = np.array([mol.properties.energy.E for mol in core])

    # Calculate and return dG and ddG
    dG = (G_lig + G_core) - G_tot
    dE = (E_lig + E_core) - E_tot
    ddG = dG - dE
    return ddG
