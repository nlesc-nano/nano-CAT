"""A module with functionalities for molecular graphs.

Index
-----
.. currentmodule:: nanoCAT.bulk
.. autosummary::
    GraphConstructor
    NeighborTuple
    yield_distances

API
---
.. autoclass:: GraphConstructor
.. autoclass:: NeighborTuple
.. autofunction:: yield_distances

"""

from __future__ import annotations

from types import MappingProxyType
from itertools import chain
from collections.abc import Iterable
from typing import (
    Dict,
    Tuple,
    NamedTuple,
    Generator,
    TYPE_CHECKING,
    Callable,
    TypeVar,
    Mapping,
)

import numpy as np
from scm.plams import Molecule, Atom, Bond, MoleculeError

from CAT.attachment.mol_split_cm import SplitMol
from CAT.attachment.ligand_opt import split_mol
from CAT.attachment.edge_distance import edge_dist

if TYPE_CHECKING:
    import numpy.typing as npt

_T = TypeVar("_T")

__all__ = ["GraphConstructor", "NeighborTuple", "yield_distances"]


class NeighborTuple(NamedTuple):
    """A namedtuple with the output of :meth:`GraphConstructor.__call__`."""

    aligned: np.bool_
    mol: Molecule
    mol_next: Dict[Molecule, Tuple[float, Atom]]
    mol_prev: None | Molecule
    mol_dist_mat: npt.NDArray[np.float64]
    start: int


class GraphConstructor:
    """A namedtuple with the output of :meth:`GraphConstructor.__call__`.

    Examples
    --------
    .. code-block:: python

        >>> from scm.plams import Molecule, Atom
        >>> from nanoCAT.bulk import GraphConstructor, yield_distances

        >>> mol: Molecule = ...
        >>> anchor: Atom = ...

        >>> constructor = GraphConstructor(mol)
        >>> graph_dict = constructor(anchor)
        >>> yield_distances(graph_dict)
        21.58389271

    """
    id_set: set[int]
    neighbor_dict: dict[Molecule, NeighborTuple]

    @property
    def mol(self) -> Molecule:
        """Return the underlying :class:`plams.Molecule <scm.plams.mol.molecule.Molecule>`"""
        return self._mol

    @property
    def bond_mapping(self) -> MappingProxyType[Atom, Atom]:
        """Return the graph-based distance matrix of :attr:`mol`."""
        return self._bond_mapping

    @property
    def dist_mat(self) -> npt.NDArray[np.float64]:
        """Return the graph-based distance matrix of :attr:`mol`."""
        return self._dist_mat

    def __init__(self, mol: Molecule) -> None:
        """Initialize the instance."""
        mol.set_atoms_id(start=0)

        self._mol = mol
        self._dist_mat = _get_dist_mat(mol)
        self._dist_mat.setflags(write=False)
        self.id_set = set()
        self.neighbor_dict = {}

    def __call__(self, anchor: Atom) -> Dict[Molecule, NeighborTuple]:
        """Construct a directed graph starting from **anchor**."""
        if len(self.id_set):
            self.id_set = set()
        if len(self.neighbor_dict):
            self.neighbor_dict = {}

        bonds = split_mol(self.mol, anchor)
        iterator = chain.from_iterable(((i, j), (j, i)) for i, j in bonds)
        self._bond_mapping = MappingProxyType(dict(iterator))

        with SplitMol(self.mol, bonds) as mol_tup:
            mol_start = self._find_start(mol_tup, anchor)
            self._dfs(mol_start, anchor)

        return self.neighbor_dict

    def _dfs(
        self,
        mol: Molecule,
        start: Atom,
        aligned: np.bool_ = np.True_,
        mol_prev: None | Molecule = None,
    ) -> None:
        """Depth-first search helper method for :meth:`__call__`."""
        tup = self._find_neighbors(mol, aligned, start, mol_prev)
        self.neighbor_dict[mol] = tup
        for m, (_, start) in tup.mol_next.items():
            self._dfs(m, start, ~aligned, mol)

    @staticmethod
    def _find_start(mol_list: Iterable[Molecule], atom: Atom) -> Molecule:
        """Find the molecule in **mol_list** containing **atom**."""
        for mol in mol_list:
            if atom in mol:
                return mol
        raise MoleculeError(f"{atom!r} is not in any of the passed molecules")

    def _find_neighbors(
        self,
        mol: Molecule,
        aligned: np.bool_,
        start: Atom,
        mol_prev: None | Molecule = None,
    ) -> NeighborTuple:
        """Construct a :class:`NeighborTuple` for **mol**."""
        intersection = self.bond_mapping.keys() & set(mol)

        mol_next = {}
        iterator = ((at, self.bond_mapping[at]) for at in intersection)
        for at1, at2 in iterator:
            id = hash(at1) ^ hash(at2)
            if id not in self.id_set:
                self.id_set.add(id)
                dist = self.dist_mat[at1.id, start.id].item()
                mol_next[at2.mol] = (dist, at2)

        return NeighborTuple(
            aligned,
            mol,
            mol_next,
            mol_prev,
            _get_dist_mat(mol),
            mol.atoms.index(start),
        )


def _idx_iter(atoms: Iterable[Atom], bonds: Iterable[Bond]) -> Generator[int, None, None]:
    """Helper function for :func:`_get_dist_mat`."""
    atom_dict = {at: i for i, at in enumerate(atoms)}
    for at1, at2 in bonds:
        # Filter out the temporary capping atoms added by `SplitMol`;
        # those won't have the `id` attribute and are thus absent from `atom_dict`
        try:
            i = atom_dict[at1]
            j = atom_dict[at2]
        except KeyError:
            continue
        yield from (i, j, j, i)


def _get_dist_mat(mol: Molecule) -> np.ndarray:
    """Construct a distance matrix from the molecular graph of **mol**."""
    atom_list = [at for at in mol if hasattr(at, "id")]
    idx_ar = np.fromiter(_idx_iter(atom_list, mol.bonds), dtype=np.intp).reshape(-1, 2)
    mol_array = np.fromiter(chain.from_iterable(atom_list), dtype=np.float64).reshape(-1, 3)
    return edge_dist(mol_array, edges=idx_ar)


def _find_start(mol_graph: Mapping[Molecule, NeighborTuple]) -> Tuple[Molecule, NeighborTuple]:
    """Find the start of the molecular graph."""
    for m, tup in mol_graph.items():
        if tup.mol_prev is None:
            return m, tup
    else:
        raise ValueError("Failed to identify the start of the molecular graph")


def yield_distances(
    mol_graph: Mapping[Molecule, NeighborTuple],
    func: Callable[[np.float64, np.float64], _T],
    offset_xy: float = 0,
    offset_z: float = 0,
    start: None | Molecule = None,
) -> Generator[Tuple[_T, np.float64, np.float64], None, None]:
    """Traverse the graph and yield the weighted distances.

    Parameters
    ----------
    mol_graph : :class:`dict[Molecule, NeighborTuple] <dict>`
        The molecular graph as constructed by :class:`GraphConstructor`.
    func : :class:`Callable[[np.float64, np.float64], Any] <collections.abc.Callable>`, optional
        An optional function for creating weighted distances.
        If provided, the function should take the radial distance and height as parameters
        and return a new value.
    offset_xy : :class:`float`
        The radial offset.
    offset_z : :class:`float`
        The offset of the height.
    start : :class:`Molecule`, optional
        The starting point of the graph.

    Yields
    ------
    :class:`float`, :class:`float` & :class:`float`
        Three floats, representing the weighted radial distance,
        the radial distance and the height.

    """
    if start is None:
        start, tup = _find_start(mol_graph)
    else:
        tup = mol_graph[start]
    aligned = tup.aligned

    # Traverse the distance matrix of `start`, starting from its starting point
    for i in tup.mol_dist_mat[tup.start]:  # type: np.float64
        ret_xy = (i * ~aligned) + offset_xy
        ret_z = (i * aligned) + offset_z
        ret_xy_weight = func(ret_xy, ret_z)
        yield (ret_xy_weight, ret_xy, ret_z)

    # Calculate the new offsets and continue traversing the graph in a dfs-based manner
    for m, (delta, _) in tup.mol_next.items():
        offset_xy_new = offset_xy + (delta * ~aligned)
        offset_z_new = offset_z + (delta * aligned)
        yield from yield_distances(
            mol_graph,
            func=func,
            offset_xy=offset_xy_new.item(),
            offset_z=offset_z_new.item(),
            start=m,
        )
