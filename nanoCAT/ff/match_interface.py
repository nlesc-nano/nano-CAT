"""
nanoCAT.ff.match_interface
==========================

A module for interfacing with MATCH: Multipurpose Atom-Typer for CHARMM.

Index
-----
.. currentmodule:: nanoCAT.ff.match_interface
.. autosummary::
    interface
    df_match_interface
    _clean_rtf
    _clean_prm
    _parse_rtf

API
---
.. autofunction:: match_interface
.. autofunction:: df_match_interface
.. autofunction:: _clean_rtf
.. autofunction:: _clean_prm
.. autofunction:: _parse_rtf

"""

import os
import subprocess
from typing import Optional
from os.path import (join, basename)
from tempfile import NamedTemporaryFile

import pandas as pd

from scm.plams import Molecule
import scm.plams.interfaces.molecule.rdkit as molkit

__all__ = ['match_interface', 'df_match_interface']

# Alias for dataframe columns
MOL = ('mol', '')

# MATCH-related paths
try:
    MATCH = join(os.environ['MATCH'], 'scripts', 'MATCH.pl')
    RESOURCES = join(os.environ['MATCH'], 'resources')
except KeyError:
    MATCH = RESOURCES = None


def df_match_interface(df: pd.DataFrame) -> None:
    """Iterate over all molecules in **df** and set atom types using :func:`.match_interface`."""
    for mol in df[MOL]:
        match_interface(mol)


def match_interface(mol: Molecule, forcefield: str = 'top_all36_cgenff_new',
                    output_dir: Optional[str] = None) -> None:
    """Interface with MATCH_ (Multipurpose Atom-Typer for CHARMM), producing a .prm file.

    .. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    forcefield : str
        The name of the requested forcefield.
        Available options displayed below (*i.e.* all .par file in ``$MATCH/resources/``):
            * ``"top_all22_prot"``
            * ``"top_all27_na"``
            * ``"top_all35_carb"``
            * ``"top_all35_ether"``
            * ``"top_all36_cgenff"``
            * ``"top_all36_cgenff_new"``
            * ``"top_all36_lipid"``

    output_dir : str
        Optional: The output directory of the resulting .prm file.
        Will default to the current working directory if ``None``.

    Raises
    ------
    EnvironmentError:
        Raised if the ``"MATCH"`` and ``"PerlChemistry"`` environment variables have not been set.

    FileNotFoundError:
        Raised if **forcefield** cannot be found in ``$MATCH/resources/``

    RuntimeError:
        Raised if MATCH failed to parse the supplied molecule.

    See Also
    --------
    Publication: `MATCH: An atom-typing toolset for molecular mechanics force fields, J.D. Yesselman, D.J. Price, J.L. Knight and C.L. Brooks III, J. Comput. Chem., 2011 <https://doi.org/10.1002/jcc.21963>`_

    """  # noqa
    # Check if $MATCH and $PerlChemistry have been set
    if MATCH is RESOURCES is None:
        raise EnvironmentError("The 'MATCH' and/or 'PerlChemistry' environment "
                               "variables have not been set")

    # Check if the supplied forcefield is available
    if forcefield + '.par' not in os.listdir(RESOURCES):
        ff_dir = join('$MATCH', 'resources') + os.sep
        raise FileNotFoundError(f'Supplied forcefield, "{forcefield}", not available in "{ff_dir}"')

    with NamedTemporaryFile('w', suffix='.pdb') as f:
        molkit.writepdb(mol, f)
        f.seek(0)
        args = MATCH, '-forcefield', forcefield, f.name
        out = subprocess.getoutput(' '.join(i for i in args))

    if 'Success!' not in out:
        raise RuntimeError(f'MATCH failed to parse {mol.properties.name}; output: {out}')

    _clean_rtf(mol, f.name)
    _clean_prm(mol, f.name, output_dir)


def _clean_rtf(mol: Molecule, filename: str) -> None:
    """Clean the .rtf files produced by MATCH."""
    # Delete both .rtf files
    top_rtf = join(os.getcwd(), 'top_' + basename(filename)).replace('.pdb', '.rtf')
    rtf = join(os.getcwd(), basename(filename)).replace('.pdb', '.rtf')

    _parse_rtf(mol, rtf)
    os.remove(top_rtf)
    os.remove(rtf)


def _clean_prm(mol: Molecule, filename: str, output_dir: Optional[str] = None) -> None:
    """Clean the .prm file produced by MATCH."""
    _output_dir = output_dir or os.getcwd()
    mol_name = mol.properties.name or mol.get_formula()

    prm_in = join(os.getcwd(), basename(filename)).replace('.pdb', '.prm')
    mol.properties.prm = prm_out = join(_output_dir, mol_name + '.prm')
    os.rename(prm_in, prm_out)


def _parse_rtf(mol: Molecule, filename: str) -> None:
    """Parse the content of a CHARMM RTF file (.rtf) produced by MATCH."""
    with open(filename, 'r') as f:
        for i in f:
            if i[:5] == 'GROUP':  # Ignore all parts preceding 'GROUP'
                break

        for i, at in zip(f, mol):
            if i[:4] == 'BOND':    # Ignore all parts succeeding 'BOND'
                break
            *_, at.properties.atom_type, charge = i.split()
            at.properties.charge = float(charge)
