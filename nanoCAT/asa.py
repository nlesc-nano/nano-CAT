"""
nanoCAT.asa
===========

A module related to performing activation strain analyses.

Index
-----
.. currentmodule:: nanoCAT.asa
.. autosummary::
    init_asa
    get_asa_energy

API
---
.. autofunction:: init_asa
.. autofunction:: get_asa_energy

"""

from typing import Collection

import numpy as np

from scm.plams import Settings, Molecule
import scm.plams.interfaces.molecule.rdkit as molkit

import rdkit
from rdkit.Chem import AllChem

from CAT.logger import logger
from CAT.settings_dataframe import SettingsDataFrame

__all__ = ['init_asa']

# Aliases for pd.MultiIndex columns
MOL = ('mol', '')
ASA_INT = ('ASA', 'E_int')
ASA_STRAIN = ('ASA', 'E_strain')
ASA_E = ('ASA', 'E')
SETTINGS1 = ('settings', 'ASA 1')


def init_asa(qd_df: SettingsDataFrame) -> None:
    """Initialize the activation-strain analyses (ASA).

    The ASA (RDKit UFF level) is conducted on the ligands in the absence of the core.

    Parameters
    ----------
    |CAT.SettingsDataFrame|_
        A dataframe of quantum dots.

    """
    # Unpack arguments
    settings = qd_df.settings.optional
    db = settings.database.db
    overwrite = db and 'qd' in settings.database.overwrite
    write = db and 'qd' in settings.database.write
    logger.info('Starting ligand activation strain analysis')

    # Prepare columns
    columns = [ASA_INT, ASA_STRAIN, ASA_E]
    for i in columns:
        qd_df[i] = np.nan

    # Fill columns
    qd_df['ASA'] = get_asa_energy(qd_df[MOL])
    logger.info('Finishing ligand activation strain analysis')

    # Calculate E_int, E_strain and E
    if write:
        recipe = Settings()
        recipe['ASA 1'] = {'key': 'RDKit_' + rdkit.__version__, 'value': 'UFF'}
        db.update_csv(
            qd_df,
            columns=[SETTINGS1]+columns,
            job_recipe=recipe,
            database='QD',
            overwrite=overwrite
        )


def get_asa_energy(mol_colllection: Collection[Molecule]) -> np.ndarray:
    """Perform an activation strain analyses (ASA).

    The ASA calculates the interaction, strain and total energy.
    The ASA is performed on all ligands in the absence of the core at the UFF level (RDKit).

    Parameters
    ----------
    mol_series : |pd.Series|_
        A series of PLAMS molecules.

    Returns
    -------
    :math:`n*3` |np.ndarray|_ [|np.float64|_]
        An array containing E_int, E_strain and E for all *n* molecules in **mol_series**.

    """
    ret = np.zeros((len(mol_colllection), 4))

    for i, mol in enumerate(mol_colllection):
        logger.info(f'UFFGetMoleculeForceField: {mol.properties.name} activation strain '
                    'analysis has started')

        ligands = mol.copy()
        uff = AllChem.UFFGetMoleculeForceField

        # Calculate the total energy of all perturbed ligands in the absence of the core
        core_atoms = [at for at in ligands if at.properties.pdb_info.ResidueName == 'COR']
        for atom in core_atoms:
            ligands.delete_atom(atom)

        rdmol = molkit.to_rdmol(ligands)
        E_ligands = uff(rdmol, ignoreInterfragInteractions=False).CalcEnergy()

        # Calculate the total energy of the isolated perturbed ligands in the absence of the core
        ligand_list = ligands.separate()
        E_ligand = 0.0
        for ligand in ligand_list:
            rdmol = molkit.to_rdmol(ligand)
            E_ligand += uff(rdmol, ignoreInterfragInteractions=False).CalcEnergy()

        # Calculate the total energy of the optimized ligand
        uff(rdmol, ignoreInterfragInteractions=False).Minimize()
        E_ligand_opt = uff(rdmol, ignoreInterfragInteractions=False).CalcEnergy()

        # Update ret with the new activation strain terms
        ret[i] = E_ligands, E_ligand, E_ligand_opt, len(ligand_list)

        logger.info(f'UFFGetMoleculeForceField: {mol.properties.name} activation strain '
                    'analysis is successful')

    # Post-process and return
    ret[:, 0] -= ret[:, 1]
    ret[:, 1] -= ret[:, 2] * ret[:, 3]
    ret[:, 2] = ret[:, 0] + ret[:, 1]
    return ret[:, 0:3]
