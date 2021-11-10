"""
nanoCAT.recipes.bulk
====================

A short recipe for accessing the ligand-bulkiness workflow.

Index
-----
.. currentmodule:: nanoCAT.recipes
.. autosummary::
    bulk_workflow
    fast_bulk_workflow

API
---
.. autofunction:: bulk_workflow
.. autofunction:: fast_bulk_workflow

"""

from __future__ import annotations

import sys
import warnings
import functools
from itertools import chain
from typing import (
    Any,
    Iterable,
    List,
    Iterator,
    Callable,
    Generator,
    Tuple,
    Union,
    Sequence,
    SupportsFloat,
    TYPE_CHECKING,
    TypeVar,
)


import numpy as np

from scm.plams import Molecule, Atom, Bond
from rdkit import Chem
from CAT.data_handling.mol_import import read_mol
from CAT.data_handling.validate_mol import validate_mol
from CAT.data_handling.anchor_parsing import parse_anchors
from CAT.attachment.ligand_anchoring import _smiles_to_rdmol, find_substructure
from CAT.attachment.ligand_opt import optimize_ligand, allign_axis

from nanoCAT.mol_bulk import get_V, get_lig_radius
from nanoCAT.bulk import yield_distances, GraphConstructor

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import SupportsIndex

if sys.version_info > (3, 8):
    _WeightReturn = Union[str, bytes, SupportsFloat, "SupportsIndex"]
else:
    _WeightReturn = Union[str, bytes, SupportsFloat]

_WT = TypeVar("_WT", bound=_WeightReturn)
_WeightFunc1 = Callable[[np.float64], _WT]
_WeightFunc2 = Callable[[np.float64, np.float64], _WT]

__all__ = ['bulk_workflow', 'fast_bulk_workflow']


def bulk_workflow(
    smiles_list: Iterable[str],
    anchor: str = 'O(C=O)[H]',
    *,
    anchor_condition: None | Callable[[int], bool] = None,
    diameter: None | float = 4.5,
    height_lim: None | float = 10.0,
    optimize: bool = True,
) -> Tuple[List[Molecule], npt.NDArray[np.float64]]:
    """Start the CAT ligand bulkiness workflow with an iterable of smiles strings.

    Examples
    --------
    .. code:: python

        >>> from CAT.recipes import bulk_workflow

        >>> smiles_list = [...]
        >>> mol_list, bulk_array = bulk_workflow(smiles_list, optimize=True)


    Parameters
    ----------
    smiles_list : :class:`Iterable[str] <collections.abc.Iterable>`
        An iterable of SMILES strings.
    anchor : :class:`str`
        A SMILES string representation of an anchor group such as ``"O(C=O)[H]"``.
        The first atom will be marked as anchor atom while the last will be dissociated.
        Used for filtering molecules in **smiles_list**.
    anchor_condition : :class:`Callable[[int], bool] <collections.abc.Callable>`, optional
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
    :class:`list[plams.Molecule] <list>` and :class:`np.ndarray[np.float64] <numpy.ndarray>`
        A list of plams Molecules and a matching array of :math:`V_{bulk}` values.

    """
    _mol_list = read_smiles(smiles_list)  # smiles to molecule
    # filter based on functional groups
    mol_list = list(_filter_mol(_mol_list, anchor=anchor, condition=anchor_condition))

    opt_and_allign(mol_list, opt=optimize)  # optimize and allign
    V_bulk = bulkiness(mol_list, diameter=diameter, height_lim=height_lim)  # calculate bulkiness
    return mol_list, V_bulk


def _weight(
    r: np.float64,
    h: np.float64,
    w_func: _WeightFunc1[_WT],
    h_max: float = 10,
    r_min: float = 4.5,
) -> _WT | int:
    if r < r_min or h > h_max:
        return 0
    else:
        return w_func(r - (r_min / 2))


def fast_bulk_workflow(
    smiles_list: Iterable[str],
    anchor: str = 'O(C=O)[H]',
    *,
    anchor_condition: None | Callable[[int], bool] = None,
    diameter: None | float = 4.5,
    height_lim: None | float = 10.0,
    func: None | _WeightFunc1 = np.exp,
) -> Tuple[List[Molecule], npt.NDArray[np.float64]]:
    """Start the ligand fast-bulkiness workflow with an iterable of smiles strings.

    Examples
    --------
    .. code:: python

        >>> from CAT.recipes import fast_bulk_workflow

        >>> smiles_list = [...]
        >>> mol_list, bulk_array = fast_bulk_workflow(smiles_list, optimize=True)


    Parameters
    ----------
    smiles_list : :class:`Iterable[str] <collections.abc.Iterable>`
        An iterable of SMILES strings.
    anchor : :class:`str`
        A SMILES string representation of an anchor group such as ``"O(C=O)[H]"``.
        The first atom will be marked as anchor atom while the last will be dissociated.
        Used for filtering molecules in **smiles_list**.
    anchor_condition : :class:`Callable[[int], bool] <collections.abc.Callable>`, optional
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
    func : :class:`Callable[[np.float64], Any] <collections.abc.Callable>`
        A function for weighting each radial distance.
        Defaults to :func:`np.exp <numpy.exp>`.

    Returns
    -------
    :class:`list[plams.Molecule] <list>` and :class:`np.ndarray[np.float64] <numpy.ndarray>`
        A list of plams Molecules and a matching array of :math:`V_{bulk}` values.

    Raises
    ------
    RuntimeWarning
        Issued if an exception is encountered when constructing or traversing
        one of the molecular graphs.
        The corresponding bulkiness value will be set to ``nan`` in such case.

    """
    mol_list = list(read_smiles2(smiles_list, anchor=anchor, condition=anchor_condition))

    if height_lim is None:
        height_lim = np.inf
    if diameter is None:
        diameter = -np.inf

    if func is None:
        w_func = lambda x, y: x
    else:
        w_func = functools.partial(_weight, h_max=height_lim, r_min=diameter, w_func=func)

    arg_iter = ((m, m.properties.dummies) for m in mol_list)
    iterator = _fast_bulkiness_iter(arg_iter, w_func)
    V_bulk = np.fromiter(iterator, dtype=np.float64, count=len(mol_list))
    return mol_list, V_bulk


def _fast_bulkiness_iter(
    iterator: Iterator[Tuple[Molecule, Atom]],
    func: _WeightFunc2[_WT],
) -> Generator[_WT | float, None, None]:
    for mol, atom in iterator:
        try:
            graph = GraphConstructor(mol)
            dct = graph(atom)
        except Exception as ex1:
            warn_ = RuntimeWarning("Failed to construct the molecular graph "
                                   f"of {mol.properties.smiles!r}")
            warn_.__cause__ = ex1
            warnings.warn(warn_, stacklevel=2)
            yield np.nan
            continue

        try:
            ret = sum(i for i, *_ in yield_distances(dct, func=func))
        except Exception as ex2:
            warn_ = RuntimeWarning("Failed to traverse the molecular graph "
                                   f"of {mol.properties.smiles!r}")
            warn_.__cause__ = ex2
            warnings.warn(warn_, stacklevel=2)
            yield np.nan
        else:
            yield ret


def read_smiles(smiles_list: Iterable[str]) -> List[Molecule]:
    """Convert smiles strings into CAT-compatible plams molecules."""
    input_mol = list(smiles_list)
    validate_mol(input_mol, 'input_ligands')
    return read_mol(input_mol)


def _iter_mol_block(
    mol_block: str,
    size: int,
) -> Generator[Tuple[str, Tuple[str, str, str]], None, None]:
    """Yield the symbols and coordinates embedded within the passed ``MolBlock``."""
    slc = np.s_[4:4 + size]
    lst = mol_block.split("\n")[slc]
    for i in lst:
        x, y, z, symbol, *_ = i.split()
        yield symbol, (x, y, z)


def _molecule_from_rdmol(
    rdmol: Chem.Mol,
    smiles: str,
    matches: Iterable[Sequence[int]],
    split: bool = True,
) -> Generator[Molecule, None, None]:
    """Construct a PLAMS molecule from the passed rdkit mol's ``MolBlock``."""
    for tup in matches:
        try:
            i, *_, j = tup  # type: int, Any, None | int
        except ValueError:
            i = tup[0]
            j = None

        # Split the capping atom (j) from the main molecule
        if j is not None and split:
            if i > j:
                i -= 1
            rdmol_edit = Chem.EditableMol(rdmol)
            rdmol_edit.RemoveAtom(j)
            rdmol_new = rdmol_edit.GetMol()
            anchor = rdmol_new.GetAtoms()[i]
            anchor.SetFormalCharge(anchor.GetFormalCharge() - 1)
        else:
            rdmol_new = rdmol

        # Parse the .mol block and convert it into a PLAMS molecule
        mol_block = Chem.MolToMolBlock(rdmol)
        iterator = _iter_mol_block(mol_block, size=len(rdmol.GetAtoms()))
        mol = Molecule()
        mol.atoms = [Atom(symbol=symbol, coords=xyz, mol=mol) for symbol, xyz in iterator]
        for bond in rdmol.GetBonds():
            at1 = mol.atoms[bond.GetBeginAtomIdx()]
            at2 = mol.atoms[bond.GetEndAtomIdx()]
            mol.add_bond(Bond(at1, at2, order=bond.GetBondTypeAsDouble()))

        # Set properties and yield
        mol.properties.smiles = smiles
        mol.properties.dummies = mol.atoms[i]
        mol.properties.anchor = f"{mol.properties.dummies.symbol}{i + 1}"
        yield mol


def read_smiles2(
    smiles_list: Iterable[str],
    anchor: str = "O(C=O)[H]",
    condition: None | Callable[[int], bool] = None,
) -> Generator[Molecule, None, None]:
    """Convert smiles strings into CAT-compatible plams molecules."""
    anchor_rdmol = _smiles_to_rdmol(anchor)
    for smiles in smiles_list:
        # Read the SMILES strings
        try:
            canon_smiles = Chem.CanonSmiles(smiles)
            rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        except Exception as ex1:
            warn_ = RuntimeWarning(f"Failed to parse {smiles!r}")
            warn_.__cause__ = ex1
            warnings.warn(warn_, stacklevel=2)
            continue

        # Perform a substructure match
        try:
            matches = rdmol.GetSubstructMatches(anchor_rdmol, useChirality=True)
            n_matches = len(matches)

            assertable = bool(n_matches)
            if condition is not None:
                assertable &= condition(n_matches)
            assert assertable
        except AssertionError:
            continue
        except Exception as ex2:
            warn_ = RuntimeWarning(f"Failed to match {smiles!r} and {anchor!r}")
            warn_.__cause__ = ex2
            warnings.warn(warn_, stacklevel=2)
            continue

        # Construct a new PLAMS molecule from the rdkit molecule
        try:
            yield from _molecule_from_rdmol(rdmol, canon_smiles, matches)
        except Exception as ex3:
            warn_ = RuntimeWarning(f"Failed to construct a PLAMS Molecule from {smiles!r}")
            warn_.__cause__ = ex3
            warnings.warn(warn_, stacklevel=2)
            continue


def _filter_mol(
    mol_list: Iterable[Molecule],
    anchor: str = 'O(C=O)[H]',
    condition: None | Callable[[int], bool] = None,
) -> Iterator[Molecule]:
    """Filter all input molecules based on the presence of a functional group (the "anchor")."""
    anchor_rdmols = parse_anchors(anchor)
    return chain.from_iterable(
        find_substructure(mol, anchor_rdmols, condition=condition) for mol in mol_list
    )


def opt_and_allign(mol_list: Iterable[Molecule], opt: bool = True) -> None:
    """Optimize all molecules and allign them along the x-axis; set :code:`opt=False` to disable the optimization."""  # noqa
    process_mol: Callable[[Molecule], None] = optimize_ligand if opt else allign_axis
    for mol in mol_list:
        process_mol(mol)


def bulkiness(
    mol_list: Iterable[Molecule],
    diameter: None | float = 4.5,
    height_lim: None | float = 10.0,
) -> npt.NDArray[np.float64]:
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
    return np.fromiter(_iterate(mol_list), count=count, dtype=np.float64)
