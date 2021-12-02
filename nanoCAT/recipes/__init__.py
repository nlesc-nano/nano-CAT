"""
CAT.recipes
===========

A number of recipes constructed using the CAT and Nano-CAT packages.

Examples
--------
.. code:: python

    >>> from nanoCAT.recipes import bulk_workflow
    >>> from nanoCAT.recipes import get_lig_charge
    >>> from nanoCAT.recipes import replace_surface
    >>> from nanoCAT.recipes import get_coordination_number, coordination_outer
    >>> from nanoCAT.recipes import dissociate_surface, row_accumulator, dissociate_bulk
    >>> from nanoCAT.recipes import get_mol_length, filter_mol, filter_data
    >>> from nanoCAT.recipes import run_jobs, get_global_descriptors, cdft
    >>> from nanoCAT.recipes import get_entropy
    >>> from nanoCAT.recipes import run_fast_sigma, get_compkf, read_csv
    ...

"""

from .bulk import bulk_workflow, fast_bulk_workflow
from .charges import get_lig_charge
from .mark_surface import replace_surface
from .dissociation import dissociate_surface, row_accumulator, dissociate_bulk
from .coordination_number import get_coordination_number, coordination_outer
from .multi_lig_job import multi_ligand_job
from .mol_filter import get_mol_length, filter_mol, filter_data
from .cdft_utils import run_jobs, get_global_descriptors, cdft
from .entropy import get_entropy
from .fast_sigma import run_fast_sigma, get_compkf, read_csv, sanitize_smiles_df

__all__ = [
    'bulk_workflow', 'fast_bulk_workflow',
    'replace_surface', 'dissociate_surface', 'dissociate_bulk',
    'row_accumulator', 'get_lig_charge',
    'get_coordination_number', 'coordination_outer',
    'multi_ligand_job',
    'get_mol_length', 'filter_mol', 'filter_data',
    'run_jobs', 'get_global_descriptors', 'cdft',
    'get_entropy',
    'run_fast_sigma', 'get_compkf', 'read_csv', 'sanitize_smiles_df',
]
