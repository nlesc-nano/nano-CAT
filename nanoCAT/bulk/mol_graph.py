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
from typing import Dict, Tuple, NamedTuple, Generator, TYPE_CHECKING, Callable

import numpy as np
from scm.plams import Molecule, Atom, MoleculeError

from CAT.attachment.mol_split_cm import SplitMol
from CAT.attachment.ligand_opt import split_mol
from CAT.attachment.edge_distance import edge_dist

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["GraphConstructor", "NeighborTuple", "yield_distances"]


class NeighborTuple(NamedTuple):
    """A namedtuple with the output of :meth:`GraphConstructor.__call__`."""

    alligned: bool
    mol: Molecule
    mol_next: Dict[Molecule, Tuple[float, Atom]]
    mol_prev: None | Molecule


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
        >>> mol_start, graph_dict = constructor(anchor)
        >>> yield_distances(graph_dict, mol_start)
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

    def __call__(self, anchor: Atom) -> Tuple[Molecule, Dict[Molecule, NeighborTuple]]:
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
            self._dfs(mol_start, anchor, True)

        dct = self.neighbor_dict
        return mol_start, dct

    def _dfs(
        self,
        mol: Molecule,
        start: Atom,
        alligned: bool,
        mol_prev: None | Molecule = None,
    ) -> None:
        """Depth-first search helper method for :meth:`__call__`."""
        tup = self._find_neighbors(mol, alligned, start, mol_prev)
        self.neighbor_dict[mol] = tup
        for m, (_, start) in tup.mol_next.items():
            self._dfs(m, start, not(alligned), mol)

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
        alligned: bool,
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
                dist: float = self.dist_mat[at1.id, start.id].item()
                mol_next[at2.mol] = (dist, at1)
        return NeighborTuple(alligned, mol, mol_next, mol_prev)


def _get_dist_mat(mol: Molecule) -> np.ndarray:
    """Construct a distance matrix from the molecular graph of **mol**."""
    bonds = chain.from_iterable((i.id, j.id, j.id, i.id) for i, j in mol.bonds)

    shape = 2, 2 * len(mol.bonds)
    count = 4 * len(mol.bonds)
    idx_ar = np.fromiter(bonds, dtype=np.intp, count=count).reshape(shape)
    return edge_dist(mol, edges=idx_ar.T)


def yield_distances(
    mol_graph: Dict[Molecule, NeighborTuple],
    start: Molecule,
    offset: float = 0,
    func: None | Callable[[float], float] = None,
) -> Generator[float, None, None]:
    """Traverse the graph and sum the distances."""
    tup = mol_graph[start]
    alligned = tup.alligned
    for m, (i, _) in tup.mol_next.items():
        ret = (i * offset) + alligned
        yield ret if func is None else func(ret)
        yield from yield_distances(mol_graph, m, offset=ret, func=func)
