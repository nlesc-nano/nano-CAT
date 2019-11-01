import collections
from itertools import chain, combinations, product
from typing import (
    Union, Mapping, Iterable, Tuple, Dict, List, Optional, Hashable, FrozenSet, Generator, Iterator,
    Callable, Any
)

import numpy as np
from scipy.spatial.distance import cdist

from scm.plams import Molecule, Atom

from assertionlib.dataclass import AbstractDataClass
from nanoCAT.bde.guess_core_dist import guess_core_core_dist


class MolDissociater(AbstractDataClass):
    """MolDissociater.

    Parameters
    ----------
    mol : |plams.Molecule|
        A PLAMS molecule consisting of cores and ligands.

    core_idx : :class:`int` or :class:`Iterable<colelctions.abc.Iterable>` [:class:`int`]
        An iterable with (1-based) atomic indices of all core atoms valid for dissociation.

    ligand_count : :class:`int`
        The number of ligands to-be dissociation with a single atom from
        :attr:`MolDissociater.core_idx`.

    max_dist : :class:`float`, optional
        Optional: The maximum distance between core atoms for them to-be considered neighbours.
        If ``None``, this value will be guessed based on the radial distribution function of
        **mol**.

    topology : :class:`dict` [:class:`int`, :class:`str`], optional
        A mapping of neighbouring atom counts to a user-specified topology descriptor.

    """

    """####################################### Properties #######################################"""

    @property
    def core_idx(self) -> np.ndarray: return self._core_idx

    @core_idx.setter
    def core_idx(self, value: Union[int, Iterable[int]]) -> None:
        if isinstance(value, collections.abc.Iterator):
            self._core_idx = np.fromiter(value, dtype=int)
        else:
            self._core_idx = np.array(value, dtype=int, ndmin=1)
        self._core_idx -= 1

    @property
    def max_dist(self) -> float: return self._max_dist

    @max_dist.setter
    def max_dist(self, value: Optional[float]) -> None:
        if value is not None:
            self._max_dist = float(value)
        else:
            idx = 1 + int(self.core_idx[0])
            self._max_dist = guess_core_core_dist(self.mol, self.mol[idx])

    @property
    def topology(self) -> Mapping[int, str]: return self._topology

    @topology.setter
    def topology(self, value: Optional[Mapping[int, str]]) -> None:
        self._topology = value if value is not None else {}

    _PRIVATE_ATTR: FrozenSet[str] = frozenset({'_coords', '_partition_dict'})

    def __init__(self, mol: Molecule,
                 core_idx: Union[int, Iterable[int]],
                 ligand_count: int,
                 max_dist: Optional[float] = None,
                 topology: Optional[Mapping[int, str]] = None) -> None:
        """Initialize a :class:`MolDissociater` instance."""
        super().__init__()

        self.mol: Molecule = mol
        self.core_idx: np.ndarray = core_idx
        self.ligand_count: int = ligand_count
        self.max_dist: float = max_dist
        self.topology: Mapping[int, str] = topology

        # Private instance variables
        self._coords: np.ndarray = mol.as_array()
        self._partition_dict: Optional[Mapping[int, List[Atom]]] = None

    @AbstractDataClass.inherit_annotations()
    def _str_iterator(self):
        return ((k.strip('_'), v) for k, v in super()._str_iterator())

    """################################## Topology assignment ##################################"""

    def assign_topology(self) -> None:
        """Assign a topology to all core atoms in :attr:`MolDissociater.core_idx`.

        The topology descriptor is based on:
        * The number of neighbours within a radius defined by :attr:`MolDissociater.max_dist`.
        * The mapping defined in :attr:`MolDissociater.topology`,
          which maps the number of neighbours to a user-defined topology description.

        If no topology description is available for a particular neighbouring atom count,
        then a generic :code:`str(i) + "_neighbours"` descriptor is used
        (where `i` is the neighbouring atom count).

        Performs an inplace update of all |Atom.properties| ``["topology"]`` values.

        """
        # Extract variables
        mol: Molecule = self.mol
        xyz: np.ndarray = self._coords
        i: np.ndarray = self.core_idx
        max_dist: float = self.max_dist

        # Create a distance matrix and find all elements with a distance smaller than **max_dist**
        dist = cdist(xyz[i], xyz[i])
        np.fill_diagonal(dist, max_dist)

        # Find all valid core atoms and create a topology indicator
        valid_core, _ = np.where(dist <= max_dist)
        neighbour_count = np.bincount(valid_core, minlength=len(i))
        neighbour_count -= 1
        topology: List[str] = self._get_topology(neighbour_count)

        for j, top in zip(self.core_idx, topology):
            j = 1 + int(j)  # Switch from 0-based to 1-based indices
            mol[j].properties.topology = top

    def _get_topology(self, neighbour_count: Iterable[int]) -> List[str]:
        """Translate the number of neighbouring atoms (**bincount**) into a list of topologies.

        If a specific number of neighbours (*i*) is absent from **topology_dict** then that
        particular element is set to a generic :code:`str(i) + '_neighbours'`.

        Parameters
        ----------
        neighbour_count : :math:`n` :class:`numpy.ndarray` [:class:`int`]
            An array representing the number of neighbouring atoms per array-element.

        Returns
        -------
        :math:`n` :class:`Iterator` [:class:`str`]
            A list of topologies for all :math:`n` atoms in **bincount**.

        See Also
        --------
        :attr:`MolDissociater.topology`
            A dictionary that maps neighbouring atom counts to a user-specified topology descriptor.

        """
        topology: Mapping[int, str] = self.topology
        return [topology.get(i, f'{i}_neighbours') for i in neighbour_count]

    """############################ core/ligand pair identification ############################"""

    def get_pairs_closest(self, anchor_getter: Optional[Callable[[Atom], Any]] = None
                          ) -> np.ndarray:
        """Create and return the indices of each core atom and the :math:`n` closest ligands.

        :math:`n` is defined according to :attr:`MolDissociater.ligand_count`.

        Parameters
        ----------
        anchor_getter : :data:`Callable<typing.Callable>`, optional
            A callable which takes an Atom as argument and returns an object for truth-testing.
            Atoms whose return-value evaluates to ``True`` will be treated as anchor atoms.
            If ``None``, use the |Atom.properties| ``["anchor"]`` attribute.

        Returns
        -------
        :math:`m*n` |np.ndarray|_ [|np.int64|_]
            An array with the indices of all :math:`m` valid ligand/core pairs
            (as determined by **max_dist**).

        """
        # Extract instance variables
        xyz: np.ndarray = self._coords
        i: np.ndarray = self.core_idx
        j: np.ndarray = self._gather_anchors(anchor_getter)
        n: int = self.ligand_count

        # Find all core atoms within a radius **max_dist** from a ligand
        dist = cdist(xyz[i], xyz[j])

        # Create an array with all n ligand/core pairs
        # The first column contains all core indices
        # The remaining n columns contain the n closest ligands for their respective cores
        cor_lig_pairs = np.empty((len(dist), 1+n), dtype=int)
        cor_lig_pairs[:, 0] = i
        row = np.arange(len(dist))
        for k in range(1, 1+n):
            cor_lig_pairs[:, k] = np.nanargmin(dist, axis=1)

            # Replace the min dist with nan
            # The time np.nanargmin(dist, axis=1) is called it will return the 2nd min dist etc.
            idx2nan = row, cor_lig_pairs[:, k]
            dist[idx2nan] = np.nan

        return cor_lig_pairs

    def get_pairs_distance(self, anchor_getter: Optional[Callable[[Atom], Any]] = None,
                           max_dist: float = 5.0) -> np.ndarray:
        """Create and return the indices of each core atom and all ligand pairs with **radius**.

        :math:`n` is defined according to :attr:`MolDissociater.ligand_count`.

        Parameters
        ----------
        anchor_getter : :data:`Callable<typing.Callable>`, optional
            A callable which takes an Atom as argument and returns an object for truth-testing.
            Atoms whose return-value evaluates to ``True`` will be treated as anchor atoms.
            If ``None``, use the |Atom.properties| ``["anchor"]`` attribute.

        max_dist : :class:`float`
            The radius used as cutoff.

        Returns
        -------
        :math:`m*n` |np.ndarray|_ [|np.int64|_]
            An array with the indices of all :math:`m` valid ligand/core pairs
            (as determined by **max_dist**).

        """
        # Extract instance variables
        xyz: np.ndarray = self._coords
        i: np.ndarray = self.core_idx
        j: np.ndarray = self._gather_anchors(anchor_getter)
        n: int = self.ligand_count

        # Find all core atoms within a radius **max_dist** from a ligand
        dist = cdist(xyz[i], xyz[j])
        np.fill_diagonal(dist, max_dist)

        # Construct a mapping with core atoms and keys and all matching ligands as values
        idx = np.where(dist < max_dist)
        pair_mapping: Dict[int, List[int]] = {}
        for x, y in zip(*idx):
            try:
                pair_mapping[x].append(y)
            except KeyError:
                pair_mapping[x] = [y]

        # Return a 2D array with all valid core/ligand pairs
        iterator = pair_mapping.items()
        cor_lig_pairs = list(chain.from_iterable(
            ((k,) + item for item in combinations(v, n)) for k, v in iterator if len(v) >= n
        ))

        return np.array(cor_lig_pairs)

    def get_combinations(self, cor_lig_pairs: np.ndarray) -> Dict[int, Iterator[Tuple[Tuple[int]]]]:
        """Create a dictionary with combinations."""
        # Extract instance variables
        n: int = self.ligand_count

        self._partition_dict: Mapping[int, List[Atom]] = self.partition_mol()
        partition_dict = self._partition_dict

        # Change the ligand anchor index into a residue ResidueNumber
        cor_res_pairs = cor_lig_pairs + 1
        cor_res_pairs[:, 1:] += 1

        # Fill the to-be returned dictionary
        ret: Dict[int, Iterator[Tuple[List[int], ...]]] = {}
        for core, *row in cor_res_pairs.tolist():
            lig_indices = (partition_dict[i] for i in row)
            ret[core] = combinations(lig_indices, n)

        return ret

    def _gather_anchors(self, anchor_getter: Optional[Callable[[Atom], Any]] = None) -> np.ndarray:
        """Return an array with the (0-based) indices of all anchor atoms."""
        def _default_getter(at: Atom) -> Any: return at.properties.anchor

        func = anchor_getter if anchor_getter is not None else _default_getter
        return np.array([i for i, at in enumerate(self.mol) if func(at)])

    """################################# Molecule dissociation #################################"""

    def __call__(self, combinations: Dict[int, Iterator[Tuple[List[int], ...]]]) -> Generator:
        """Get this party started."""
        # Extract instance variables
        mol: Molecule = self.mol

        # Construct new indices
        indices = self._get_new_indices()

        for core_idx, iterator in combinations.items():
            for lig_pair in iterator:
                # Create a new molecule
                mol_new = mol.copy()
                s = mol_new.properties

                # Create a list of to-be removed atoms
                core: Atom = mol_new[core_idx]
                delete_at = [core]
                delete_at += [mol_new[i] for i in chain.from_iterable(lig_pair)]

                # Update the Molecule.properties attribute of the new molecule
                s.indices = indices
                s.job_path = []
                s.core_topology = f'{core.properties.topology}_{core_idx}'
                s.lig_residue = sorted(
                    [mol_new[i[0]].properties.pdb_info.ResidueNumber for i in lig_pair]
                )
                s.df_index: str = s.core_topology + ' '.join(str(i) for i in s.lig_residue)

                for at in delete_at:
                    mol_new.delete_atom(at)
                yield mol_new

    def _get_new_indices(self) -> List[int]:
        """Return an updated version of :attr:`MolDissociater.mol` ``.properties.indices``."""
        n: int = self.ligand_count
        mol: Molecule = self.mol
        partition_dict: Mapping[int, List[Atom]] = self._partition_dict

        if not mol.properties.indices:
            mol.properties.indices = indices = []
            return indices

        # Delete the indices of the last n ligands
        ret = mol.properties.indices.copy()
        for _ in range(n):
            del ret[-1]

        # Delete the index of the last core atom
        core_max = next(partition_dict.values())[-1]
        idx = ret.index(core_max)
        del ret[idx]

        # Update the indices of all remaining ligands
        for i in ret:
            i -= 1
        return ret

    """########################################## Misc ##########################################"""

    def partition_mol(self, key_getter: Optional[Callable[[Atom], Hashable]] = None
                      ) -> Dict[int, List[int]]:
        """Partition the atoms within a molecule based on a user-specified criteria.

        Parameters
        ----------
        key_getter : :data:`Callable<typing.Callable>`, optional
            A callable which takes an Atom as argument and
            returns a :class:`Hashable<collections.abc.Hashable>` object.
            If ``None``, use the |Atom.properties| ``["pdb_info"]["ResidueNumber"]`` attribute.

        Returns
        -------
        :class:`dict` [:class:`int`, :class:`list` [:class:`int`]]
            A dictionary with keys construcetd by **key_getter** and values consisting of
            lists with matching atomic indices.

        """
        _key_getter = key_getter if key_getter is not None else self._get_residue_number
        ret = {}
        for i, at in enumerate(self.mol, 1):
            key = _key_getter(at)
            try:
                ret[key].append(i)
            except KeyError:
                ret[key] = [i]
        return ret

    @staticmethod
    def _get_residue_number(atom: Atom) -> Hashable: return atom.properties.pdb_info.ResidueNumber


from scm.plams import readpdb

filename = '/Users/basvanbeek/Downloads/Cd68Cl26Se55__26_C#CCNCC[=O][O-]@O8.pdb'
mol = readpdb(filename)
for at in mol:
    if at.properties.charge == -1:
        at.properties.anchor = True


def workflow(mol, max_dist=None, topology=None):
    # Construct parameters
    core_idx = (i for i, at in enumerate(mol, 1) if at.symbol == 'Cd')
    lig_count = 2

    # Construct a MolDissociater instance
    dissociate = MolDissociater(mol, core_idx, lig_count, max_dist, topology)

    dissociate.assign_topology()
    cor_lig_pairs = dissociate.get_pairs()
    cor_lig_combinations = dissociate.get_combinations(cor_lig_pairs)
    return dissociate(cor_lig_combinations)
