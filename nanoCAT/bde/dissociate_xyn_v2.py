from itertools import (chain, combinations)
from collections import OrderedDict
from typing import (
    Any, Iterable, Tuple, Dict, List, Optional, Hashable, FrozenSet, Callable, Iterator, Generator
)

import numpy as np
from scipy.spatial.distance import cdist

from scm.plams import (Molecule, Settings)

from CAT.abc.dataclass import AbstractDataClass
from nanoCAT.bde.guess_core_dist import guess_core_core_dist


class MolDissociater(AbstractDataClass):
    """

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule consisting of cores and ligands.

    core_idx : |Iterable|_ [|int|_]
        An iterable with atomic indices of all core atoms valid for dissociation.

    ligand_count : int
        The number of ligands to-be dissociation with a single atom from
        :attr:`MolDissociater.core_idx`.

    max_dist : float
        Optional: The maximum distance between core atoms for them to-be considered neighbours.
        If ``None``, this value will be guessed based
        on the radial distribution function of **mol**.

    topology : |dict|_ [|int|_, |str|_]
        Optional: A dictionary that maps neighbouring atom counts to a user-specified
        topology descriptor.

    """

    _PRIVATE_ATTR: FrozenSet[str] = frozenset({'_coords', '_partition_dict'})

    def __init__(self, mol: Molecule,
                 core_index: Iterable[int],
                 ligand_count: int,
                 cc_dist: Optional[float] = None,
                 topology: Optional[Dict[int, str]] = None) -> None:
        """Initialize a :class:`MolDissociater` instance."""
        self.mol: Molecule = mol
        self.core_idx: np.ndarray = np.array(core_index, dtype=int, copy=False) - 1
        self.ligand_count: int = ligand_count
        self.max_dist: float = float(cc_dist) if cc_dist is not None else guess_core_core_dist()
        self.topology: Dict[int, str] = topology if topology is not None else {}

        self._coords: np.ndarray = mol.as_array()
        self._partition_dict: Optional[OrderedDict] = None

    def run(self) -> Generator:
        self.assign_topology()
        cor_lig_pairs = self.filter_lig_core()
        cor_lig_combinations = self.get_combinations(cor_lig_pairs)
        return self.dissociate(cor_lig_combinations)

    """################################## Topology assignment ##################################"""

    def assign_topology(self) -> None:
        """Assign a toology to all atoms in :attr:`MolDissociater.mol`

        A topology is assigned to aforementioned atoms based on the number of neighbouring atoms.

        Parameters
        ----------
        xyz_array : :math:`n*3` |np.ndarray|_ [|np.float64|_]
            An array with the cartesian coordinates of a molecule with :math:`n` atoms.

        idx : |np.ndarray|_ [|np.int64|_]
            An array of atomic indices in **xyz_array**.

        topology_dict : |dict|_ [|int|_, |str|_]
            A dictionary which maps the number of neighbours (per atom) to a
            user-specified topology.

        max_dist : float
            The radius (Angstrom) for determining if an atom counts as a neighbour or not.

        Returns
        -------
        |np.ndarray|_ [|np.int64|_] and |np.ndarray|_ [|np.int64|_]
            The indices of all atoms in **xyz_array[idx]** exposed to the surface and
            the topology of atoms in **xyz_array[idx]**.

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
        neighbour_count = np.bincount(valid_core, minlength=len(i)) - 1
        topology = self._get_topology(neighbour_count)

        for j, top in zip(self.core_idx, topology):
            j = 1 + int(j)
            mol[j].properties.topology = top

    def _get_topology(self, neighbour_count: Iterable[int]) -> List[str]:
        """Translate the number of neighbouring atoms (**bincount**) into a list of topologies.

        If a specific number of neighbours (*i*) is absent from **topology_dict** then that
        particular element is set to a generic str(*i*) + '_neighbours'.

        Parameters
        ----------
        neighbour_count : :math:`n` :class:`numpy.ndarray` [:class:`int`]
            An array representing the number of neighbouring atoms per array-element.

        Returns
        -------
        :math:`n` :class:`list` [:class:`str`]
            A list of topologies for all :math:`n` atoms in **bincount**.

        See also
        --------
        :attr:`MolDissociater.topology`
            A dictionary that maps neighbouring atom counts to a user-specified topology descriptor.

        """
        topology = self.topology
        return [(topology[i] if i in topology else f'{i}_neighbours') for i in neighbour_count]

    """############################ core/ligand pair identification ############################"""

    def filter_lig_core(self, anchor_marker: Hashable = 'anchor') -> np.ndarray:
        """Create and return the indices of all possible ligand/core atom pairs.

        Parameters
        ----------

        idx_lig : |np.ndarray|_ [|np.int64|_]
            An array of all ligand anchor atoms (Y).

        Returns
        -------
        :math:`m*2` |np.ndarray|_ [|np.int64|_]
            An array with the indices of all :math:`m` valid ligand/core pairs
            (as determined by **max_dist**).

        """
        # Extract instance variables
        xyz: np.ndarray = self._coords
        i: np.ndarray = self.core_idx
        j: np.ndarray = self._gather_anchors(anchor_marker)
        n: int = self.ligand_count

        # Find all core atoms within a radius **max_dist** from a ligand
        dist = cdist(xyz[i], xyz[j])

        # Create an array with all n ligand/core pairs
        # The first column contains all core indices
        # The remaining n columns contain the n closest ligands for their respective cores
        cor_lig_pairs = np.empty((len(dist), 1+n), dtype=int)
        cor_lig_pairs[:, 0] = np.arange(len(dist))
        for i in range(1, 1+n):
            cor_lig_pairs[i] = np.nanargmin(cor_lig_pairs, axis=0)

            # Replace the min dist with nan;
            # the next iteration will now return the second most min dist etc..
            idx2nan = tuple(cor_lig_pairs[:, (0, i)].tolist())
            dist[idx2nan] = np.nan

        # Find np.nan values; np.nan values appear if a particular core atom has
        # less than n ligand neighbours
        invalid = np.isnan(cor_lig_pairs).any(axis=0)
        valid = np.invert(invalid)[None, :]
        return cor_lig_pairs[valid]

    def get_combinations(self, cor_lig_pairs: np.ndarray,
                         key_tup: Iterable[Hashable] = ('properties', 'pdb_info', 'ResidueNumber'),
                         getter: Callable = getattr) -> Dict[int, Iterator[Tuple[Tuple[int]]]]:
        self.partition_mol(key_tup, getter)

        # Extract instance variables
        mol: Molecule = self.mol
        n: int = self.ligand_count

        partition_dict = self._partition_dict
        ret: Dict[int, Iterator[Tuple[int]]] = {}

        # Change the ligand anchor index into a residue ResidueNumber
        cor_res_pairs = cor_lig_pairs + 1

        # Fill the to-be returned dictionary
        for core, (_, *row) in enumerate(cor_res_pairs):
            res_number = (self._recursive_get(mol[i], key_tup, getter) for i in row.tolist())
            lig_indices = (partition_dict[i] for i in res_number)
            ret[core] = combinations(lig_indices, n)

        return ret

    def _gather_anchors(self, key_tup: Iterable[Hashable] = ('properties', 'anchor'),
                        getter: Callable = getattr) -> np.ndarray:
        """Construct an array with the atomic indices (0-based) of all ligand anchor atoms."""
        return np.array([
            i for i, at in enumerate(self.mol) if self._recursive_get(at, key_tup, getter)
        ])

    """################################# Molecule dissociation #################################"""

    def dissociate(self, combinations: Dict[int, Iterator[Tuple[Tuple[int]]]]):
        # Extract instance variables
        mol: Molecule = self.mol

        # Construct new indices
        properties = mol.properties
        indices = self._get_new_indices()

        for _core, iterator in combinations.items():
            for lig_pair in iterator:
                # Create a new molecule
                mol_new = mol.copy()
                mol_new.properties = s = Settings()

                # Create a list of to-be removed atoms
                core = mol_new[_core]
                delete_at = [core]
                delete_at += [mol_new[i] for i in chain.from_iterable(lig_pair)]

                # Update the Molecule.properties attribute of the new molecule
                s.name = properties.name
                s.path = properties.path
                s.prm = properties.prm
                s.indices = indices
                s.job_path = []
                s.core_topology = f'{str(mol[core].properties.topology)}_{core}'
                s.lig_residue = sorted(
                    [mol[i[0]].properties.pdb_info.ResidueNumber for i in lig_pair]
                )
                s.df_index = (mol_new.properties.core_topology +
                              ' '.join(str(i) for i in mol_new.properties.lig_residue))

                # if core.bonds:
                #     raise NotImplementedError

                for at in delete_at:
                    mol_new.delete_atom(at)

                yield mol_new

    def _get_new_indices(self) -> List[int]:
        """Return an updated version of :attr:`MolDissociater.mol` ``.properties.indices``."""
        n: int = self.ligand_count
        mol: Molecule = self.mol
        partition_dict: OrderedDict = self._partition_dict

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

    def partition_mol(self,
                      key_tup: Iterable[Hashable] = ('properties', 'pdb_info', 'ResidueNumber'),
                      getter: Callable = getattr) -> OrderedDict:
        """Partition the atoms within a molecule based on a list of user specified keys."""
        ret = OrderedDict()
        for i, at in enumerate(self.mol, 1):
            key = self._recursive_get(at, key_tup, getter)
            try:
                ret[key].append(i)
            except KeyError:
                ret[key] = [i]
        return ret

    @staticmethod
    def _recursive_get(obj: Any, key_tup: Iterable[Hashable], getter: Callable = getattr):
        """Recursivelly call **getter** on **obj** untill all keys in **key_tup** are exhausted."""
        ret = obj
        for k in key_tup:
            ret = getter(ret, k)
        return ret
