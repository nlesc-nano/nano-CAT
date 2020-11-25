"""
nanoCAT.recipes.bulk
====================

A short recipe for accessing the ligand-bulkiness workflow.

Index
-----
.. currentmodule:: nanoCAT.recipes.bulk
.. autosummary::
    bulk_workflow

API
---
.. autofunction:: bulk_workflow

"""

from typing import Iterable, List, Iterator, Callable, Generator, Tuple, Optional
from itertools import chain

import numpy as np

from scm.plams import Molecule
from CAT.data_handling.mol_import import read_mol
from CAT.data_handling.validate_mol import validate_mol
from CAT.attachment.ligand_anchoring import _smiles_to_rdmol, find_substructure
from CAT.attachment.ligand_opt import optimize_ligand, allign_axis

from nanoCAT.mol_bulk import get_V, get_lig_radius

__all__ = ['bulk_workflow']


def bulk_workflow(smiles_list: Iterable[str],
                  anchor: str = 'O(C=O)[H]',
                  anchor_condition: Optional[Callable[[int], bool]] = None,
                  diameter: Optional[float] = 4.5,
                  height_lim: Optional[float] = 10.0,
                  optimize: bool = True) -> Tuple[List[Molecule], np.ndarray]:
    """Start the CAT ligand bulkiness workflow with an iterable of smiles strings.

    Examples
    --------
    .. code:: python

        >>> from CAT.recipes import bulk_workflow

        >>> smiles_list = [...]
        >>> mol_list, bulk_array = bulk_workflow(smiles_list, optimize=True)


    Parameters
    ----------
    smiles_list : :class:`~collections.abc.Iterable` [:class:`str`]
        An iterable of SMILES strings.

    anchor : :class:`str`
        A SMILES string representation of an anchor group such as ``"O(C=O)[H]"``.
        The first atom will be marked as anchor atom while the last will be dissociated.
        Used for filtering molecules in **smiles_list**.

    anchor_condition : :class:`Callable[[int], bool]<collections.abc.Callable>`, optional
        If not :data:`None`, filter ligands based on the number of identified functional groups.
        For example, ``anchor_condition = lambda n: n == 1`` will only accept ligands with
        a single **anchor** group,
        ``anchor_condition = lambda n: n >= 3`` requires three or more anchors and
        ``anchor_condition = lambda n: n < 2`` requires fewer than two anchors.

    diameter : :class:`float`, optional
        The lattice spacing, *i.e.* the average nearest-neighbor distance between the anchor atoms
        of all ligads.
        Set to :data:`None` to ignore the lattice spacing.
        Units should be in Angstrom.

    height_lim : :class:`float`, optional
        A cutoff above which all atoms are ignored.
        Set to :data:`None` to ignore the height cutoff.
        Units should be in Angstrom.

    optimize : :class:`bool`
        Enable or disable the ligand geometry optimization.

    Returns
    -------
    :class:`list` [:class:`Molecule<scm.plams.mol.molecule.Molecule>`] & :class:`numpy.ndarray`
        A list of plams Molecules and a matching array of :math:`V_{bulk}` values.

    """
    _mol_list = read_smiles(smiles_list)  # smiles to molecule
    # filter based on functional groups
    mol_list = list(_filter_mol(_mol_list, anchor=anchor, condition=anchor_condition))

    opt_and_allign(mol_list, opt=optimize)  # optimize and allign
    V_bulk = bulkiness(mol_list, diameter=diameter, height_lim=height_lim)  # calculate bulkiness
    return mol_list, V_bulk


def read_smiles(smiles_list: Iterable[str]) -> List[Molecule]:
    """Convert smiles strings into CAT-compatible plams molecules."""
    input_mol = list(smiles_list)
    validate_mol(input_mol, 'input_ligands')
    return read_mol(input_mol)


def _filter_mol(mol_list: Iterable[Molecule],
                anchor: str = 'O(C=O)[H]',
                condition: Optional[Callable[[int], bool]] = None) -> Iterator[Molecule]:
    """Filter all input molecules based on the presence of a functional group (the "anchor")."""
    anchor_rdmols = (_smiles_to_rdmol(anchor),)
    return chain.from_iterable(
        find_substructure(mol, anchor_rdmols, condition=condition) for mol in mol_list
    )


def opt_and_allign(mol_list: Iterable[Molecule], opt: bool = True) -> None:
    """Optimize all molecules and allign them along the x-axis; set :code:`opt=False` to disable the optimization."""  # noqa
    def _allign_axis(mol: Molecule) -> None:
        return allign_axis(mol, mol.properties.dummies)

    process_mol: Callable[[Molecule], None] = optimize_ligand if opt else _allign_axis
    for mol in mol_list:
        process_mol(mol)


def bulkiness(mol_list: Iterable[Molecule], diameter: Optional[float] = 4.5,
              height_lim: Optional[float] = 10.0) -> np.ndarray:
    r"""Calculate the ligand bulkiness descriptor :math:`V_{bulk}`.

    .. math::

        V(d, h_{lim}) =
        \sum_{i=1}^{n} e^{r_{i}} (\frac{2 r_{i}}{d} - 1)^{+} (1 - \frac{h_{i}}{h_{lim}})^{+}

    """
    def _iterate(mol_list: Iterable[Molecule]) -> Generator[float, None, None]:
        for mol in mol_list:
            radius, height = get_lig_radius(mol)  # From cartesian to cylindrical
            yield get_V(radius, height, diameter, None, h_lim=height_lim)

    try:
        count = len(mol_list)  # type: ignore
    except TypeError:
        count = -1
    return np.fromiter(_iterate(mol_list), count=count, dtype=float)
