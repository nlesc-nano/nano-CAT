"""
nanoCAT.bde.dissociate_xyn
==========================

A module for constructing :math:`XYn`-dissociated quantum dots.

"""

from itertools import chain, combinations, repeat
from typing import (
    Union, Mapping, Iterable, Tuple, Dict, List, Optional, FrozenSet, Generator, Iterator,
    Any, TypeVar, Hashable, SupportsInt, MutableMapping, Type, Set, Collection
)

import numpy as np
from scipy.spatial.distance import cdist

from scm.plams import Molecule, Atom, MoleculeError
from scm.plams.interfaces.molecule import rdkit as molkit
from assertionlib.dataclass import AbstractDataClass

from CAT.mol_utils import to_atnum
from CAT.attachment.ligand_anchoring import _smiles_to_rdmol
from nanoCAT.bde.guess_core_dist import guess_core_core_dist

__all__ = ['MolDissociater']

T = TypeVar('T')
CombinationsTuple = Tuple[FrozenSet[int], FrozenSet[int]]
IdxMapping = Mapping[int, Collection[int]]


def dissociate_ligand(mol: Molecule,
                      lig_count: int,
                      lig_pairs: int = 1,
                      lig_core_dist: Optional[float] = None,
                      core_atom: Union[None, int, str] = None,
                      core_index: Union[None, int, Iterable[int]] = None,
                      core_smiles: Optional[str] = None,
                      core_core_dist: Optional[float] = None,
                      topology: Optional[Mapping[int, str]] = None,
                      **kwargs: Any) -> Generator[Molecule, None, None]:
    r"""Remove :math:`XY_{n}` from **mol** with the help of the :class:`MolDissociater` class.

    The dissociation process consists of 5 general steps:

    * Constructing a :class:`MolDissociater` instance for managing the dissociation workflow.
    * Assigning a topology-descriptor to each atom with :meth:`MolDissociater.assign_topology`.
    * Identifying all valid core/ligand pairs using either :meth:`MolDissociater.get_pairs_closest`
      or :meth:`MolDissociater.get_pairs_distance`.
    * Creating all to-be dissociated core/ligand combinations with
      :meth:`MolDissociater.get_combinations`.
    * Start the dissociation process by calling the earlier
      created :class:`MolDissociater` instance.

    Examples
    --------
    .. code:: python

        >>> from typing import Iterator

        >>> import numpy as np
        >>> from scm.plams import Molecule

        # Define parameters
        >>> mol = Molecule(...)
        >>> core_idx = [1, 2, 3, 4, 5]
        >>> lig_idx = [10, 20, 30, 40]
        >>> lig_count = 2

        # Start the workflow
        >>> dissociate = MolDissociater(mol, core_idx, lig_count)
        >>> dissociate.assign_topology()
        >>> pairs: np.ndarray = dissociate.get_pairs_closest(lig_idx)
        >>> combinations: Iterator[tuple] = dissociate.get_combinations(pairs)

        # Create the final iterator
        >>> mol_iterator: Iterator[Molecule] = dissociate(cor_lig_combinations)

    Parameters
    ----------
    mol : |plams.Molecule|
        A molecule.

    lig_count : :class:`int`
        The number of to-be dissociated ligands per core atom/molecule.

    lig_pairs : :class:`int`
        The number of to-be dissociated core/ligand pairs per core atom.
        Core/ligand pairs are picked based on whichever ligands are closest to each core atom.
        This option is irrelevant if a distance based criterium is used (see **lig_dist**).

    lig_core_dist : :class:`float`, optional
        Instead of dissociating a given number of core/ligand pairs (see **lig_pairs**) dissociate
        all pairs within a given distance from a core atom.

    core_index : :class:`int` or :class:`Iterable<collections.abc.Iterable>` [:class:`int`]
        An index or set of indices with all to-be dissociated core atoms.
        See **core_atom** to define **core_idx** based on a common atomic symbol/number.

    core_atom : :class:`int` or :class:`str`, optional
        An atomic number or symbol used for automatically defining **core_idx**.
        Core atoms within the bulk (rather than on the surface) are ignored.

    core_smiles : :class:`str`, optional
        A SMILES string representing molecule containing **core_idx**.
        Provide a value here if one wants to disociate an entire molecules from the core and
        not just atoms.

    core_core_dist : :class:`float`, optional
        A value representing the mean distance between the core atoms in **core_idx**.
        If ``None``, guess this value based on the radial distribution function of **mol**
        (this is generally recomended).

    topology : :class:`Mapping<collections.abc.Mapping>` [:class:`int`, :class:`str`], optional
        A mapping neighbouring of atom counts to a user specified topology descriptor
        (*e.g.* ``"edge"``, ``"vertice"`` or ``"face"``).

    \**kwargs : :data:`Any<typing.Any>`
        For catching excess keyword arguments.

    Returns
    -------
    :class:`Generator<collections.abc.Generator>` [|plams.Molecule|]
        A generator yielding new molecules with :math:`XY_{n}` removed.

    """
    if core_atom is None and core_index is None:
        raise TypeError("The 'core_atom' and 'core_idx' parameters cannot be both 'None'")

    # Set **core_idx** to all atoms within **mol** matching **core_atom**
    if core_atom is not None:
        atnum = to_atnum(core_atom)
        core_index = [i for i, at in enumerate(mol, 1) if at.atnum == atnum]

    # Construct a MolDissociater instance
    dissociate = MolDissociater(mol, core_index, lig_count, core_core_dist, topology)
    if core_atom:
        dissociate.remove_bulk()  # Remove atoms not exposed to the surface
    dissociate.assign_topology()

    # Construct an array with all core/ligand pairs
    lig_idx = np.fromiter((i for i, at in enumerate(mol, 1) if at.properties.anchor), dtype=int)
    if lig_core_dist is None:  # Create n pairs regardless of any radius
        cl_pairs = dissociate.get_pairs_closest(lig_idx, n_pairs=lig_pairs)
    else:  # Create all pairs within a given radius
        cl_pairs = dissociate.get_pairs_distance(lig_idx, max_dist=lig_core_dist)

    # Dissociate the ligands
    lig_mapping = _lig_mapping(mol, lig_idx)
    core_mapping = _core_mapping(mol, dissociate.core_idx+1, core_smiles) if core_smiles else None
    cl_combinations = dissociate.combinations(cl_pairs, lig_mapping, core_mapping)
    return dissociate(cl_combinations)


def _lig_mapping(mol: Molecule, idx: Iterable[int]) -> IdxMapping:
    """Map **idx** to all atoms with the same residue number."""
    idx = as_array(idx, dtype=int).tolist()  # 1-based indices

    iterator = ((i, at.properties.pdb_info.ResidueNumber) for i, at in enumerate(mol, 1))
    lig_mapping = group_by_values(iterator)

    valid_keys = (mol[i].properties.pdb_info.ResidueNumber for i in idx)
    return {i: lig_mapping[k] for i, k in zip(idx, valid_keys)}


def _core_mapping(mol: Molecule, idx: Iterable[int], smiles: str) -> IdxMapping:
    """Map **idx** to all atoms part of the same substructure (see **smiles**)."""
    idx = as_array(idx, dtype=int)  # 1-based indices

    rdmol = molkit.to_rdmol(mol)
    rd_smiles = _smiles_to_rdmol(smiles)

    values: np.ndarray = np.array(rdmol.GetSubstructMatches(rd_smiles, useChirality=True))
    values += 1
    keys = np.intersect1d(idx, values)
    len_keys, len_values = len(keys), len(values)
    if len_keys != len_values:
        raise ValueError("Keys and values are of non-equal length: {len_keys} & {len(values)}")

    iterator = zip(keys, values.tolist())
    return dict(iterator)


class DummyGetter:
    """A mapping placeholder; calling `__getitem__` will return the supplied key embedded within a tuple."""  # noqa

    def __getitem__(self, key: SupportsInt) -> Tuple[int]: return (int(key),)


_DUMMY_GETTER = DummyGetter()


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
        self._core_idx = core_idx = as_array(value, dtype=int, ndmin=1, copy=True)
        core_idx -= 1
        core_idx.sort()

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
        self._topology = value or {}

    _PRIVATE_ATTR: FrozenSet[str] = frozenset({'_coords', '_core_is_lig'})

    def __init__(self, mol: Molecule,
                 core_idx: Union[int, Iterable[int]],
                 ligand_count: int,
                 max_dist: Optional[float] = None,
                 topology: Optional[Mapping[int, str]] = None) -> None:
        """Initialize a :class:`MolDissociater` instance."""
        super().__init__()

        # Public instance variables
        self.mol: Molecule = mol
        self.core_idx: np.ndarray = core_idx
        self.ligand_count: int = ligand_count
        self.max_dist: float = max_dist
        self.topology: Mapping[int, str] = topology

        # Private instance variables
        self._coords: np.ndarray = mol.as_array()
        self._core_is_lig: bool = False

    @AbstractDataClass.inherit_annotations()
    def _str_iterator(self):
        return ((k.strip('_'), v) for k, v in super()._str_iterator())

    def remove_bulk(self, max_vec_len: float = 0.5) -> None:
        """Remove out atoms specified in :attr:`MolDissociater.core_idx` which are present in the bulk.

        The function searches for all neighbouring core atoms within a radius
        :attr:`MolDissociater.max_dist`.
        Vectors are then constructed from the core atom to the mean positioon of its neighbours.
        Vector lengths close to 0 thus indicate that the core atom is surrounded in a (nearly)
        spherical pattern,
        *i.e.* it's located in the bulk of the material and not on the surface.

        Performs in inplace update of :attr:`MolDissociater.core_idx`.

        Parameters
        ----------
        max_vec_len : :class:`float`
            The maximum length of an atom vector to-be considered part of the bulk.
            Atoms producing smaller values are removed from :attr:`MolDissociater.core_idx`.
            Units are in Angstroem.

        """  # noqa
        xyz: np.ndarray = self._coords
        i: np.ndarray = self.core_idx
        max_dist: float = self.max_dist

        # Construct the distance matrix and fill the diagonal
        dist = cdist(xyz[i], xyz[i])
        np.fill_diagonal(dist, max_dist)

        x, y = np.where(dist <= max_dist)
        bincount = np.bincount(x, minlength=len(i))

        # Slice xyz_array, creating arrays of reference atoms and neighbouring atoms
        ref_at = xyz[i]
        neighbour_at = xyz[i[y]]

        # Calculate the vector length from each reference atom to the mean position
        # of its neighbours
        # A vector length close to 0.0 implies that a reference atom is surrounded by neighbours in
        # a more or less spherical pattern:
        # i.e. the reference atom is in the bulk and not on the surface
        indices = np.zeros(len(bincount), dtype=int)
        indices[1:] = np.cumsum(bincount[:-1])
        average = np.add.reduceat(neighbour_at, indices)
        average /= bincount[:, None]
        vec = ref_at - average

        vec_norm = np.linalg.norm(vec, axis=1)
        norm_accept, *_ = np.where(vec_norm > max_vec_len)
        self.core_idx = i[norm_accept]

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

        core_idx = (1 + self.core_idx).tolist()  # Switch from 0-based to 1-based indices
        for j, top in zip(core_idx, topology):
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
        :math:`n` :class:`list` [:class:`str`]
            A list of topologies for all :math:`n` atoms in **bincount**.

        See Also
        --------
        :attr:`MolDissociater.topology`
            A dictionary that maps neighbouring atom counts to a user-specified topology descriptor.

        """
        topology: Mapping[int, str] = self.topology
        return [topology.get(i, f'{i}_neighbours') for i in neighbour_count]

    """############################ core/ligand pair identification ############################"""

    def get_pairs_closest(self, lig_idx: Union[int, Iterable[int]],
                          n_pairs: int = 1) -> np.ndarray:
        """Create and return the indices of each core atom and the :math:`n` closest ligands.

        :math:`n` is defined according to :attr:`MolDissociater.ligand_count`.
        Pairs are chosen based on the norm of the :math:`n` core/ligand distances;
        *i.e.* the :math:`n` pairs with the :math:`n` lowest norms are are returned.

        Parameters
        ----------
        lig_idx : :data:`Callable<typing.Callable>`, optional
            The (1-based) indices of all ligand anchor atoms.

        n_pairs : :class:`int`
            The number of to-be returned pairs per core atom.
            If :code:`n_pairs > 1` than each successive set of to-be dissociated ligands is
            determined by the norm of the :math:`n` distances.

        Returns
        -------
        :math:`(m*i) * (n+1)` |np.ndarray|_ [|np.int64|_]
            An array with the indices of all :math:`m` valid ligand/core pairs
            with :code:`n=self.ligand_count` and :math:`i=n_pairs`.

        """
        if n_pairs <= 0:
            raise ValueError(f"The 'n_pairs' parameter should be larger than 0")

        # Extract instance variables
        xyz: np.ndarray = self._coords
        i: np.ndarray = self.core_idx
        j: np.ndarray = as_array(lig_idx, dtype=int) - 1
        n: int = self.ligand_count

        # Find all core atoms within a radius **max_dist** from a ligand
        dist = cdist(xyz[i], xyz[j])
        if n_pairs == 1:
            lig_idx = np.argsort(dist, axis=1)[:, :n]
            core_idx = i[:, None]
            return np.hstack([core_idx, j[lig_idx]])

        # Shrink the distance matrix, keep the n_pairs smallest distances per row
        stop = max(1 + n, 1 + n_pairs)
        idx_small = np.argsort(dist, axis=1)[:, :stop]
        dist_smallest = np.take_along_axis(dist, idx_small, axis=1)

        # Create an array of combinations
        combine = np.fromiter(chain.from_iterable(combinations(range(stop), n)), dtype=int)
        combine.shape = -1, n

        # Accept the n_pair entries (per row) based on the norm
        norm = np.linalg.norm(dist_smallest[:, combine], axis=2)
        idx_accept = combine[np.argsort(norm, axis=1)[:, :n_pairs]]
        idx_accept.shape = len(idx_accept), -1

        # Create an array with all core/ligand pairs
        lig_idx = np.take_along_axis(idx_small, idx_accept, axis=1)
        lig_idx.shape = -1, n
        core_idx = np.fromiter(iter_repeat(i, n_pairs), dtype=int)[:, None]
        return np.hstack([core_idx, j[lig_idx]])

    def get_pairs_distance(self, lig_idx: Union[int, Iterable[int]],
                           max_dist: float = 5.0) -> np.ndarray:
        """Create and return the indices of each core atom and all ligand pairs with **max_dist**.

        :math:`n` is defined according to :attr:`MolDissociater.ligand_count`.

        Parameters
        ----------
        lig_idx : :data:`Callable<typing.Callable>`, optional
            The (1-based) indices of all ligand anchor atoms.

        max_dist : :class:`float`
            The radius (Angstroem) used as cutoff.

        Returns
        -------
        :math:`m*(n+1)` |np.ndarray|_ [|np.int64|_]
            An array with the indices of all :math:`m` valid ligand/core pairs
            and with :code:`n=self.ligand_count`.

        """
        if max_dist <= 0.0:
            raise ValueError(f"The 'max_dist' parameter should be larger than 0.0")

        # Extract instance variables
        xyz: np.ndarray = self._coords
        i: np.ndarray = self.core_idx
        j: np.ndarray = as_array(lig_idx, dtype=int) - 1
        n: int = self.ligand_count

        # Find all core atoms within a radius **max_dist** from a ligand
        dist = cdist(xyz[j], xyz[i])
        np.fill_diagonal(dist, max_dist)

        # Construct a mapping with core atoms and keys and all matching ligands as values
        idx = np.where(dist < max_dist)
        pair_mapping: Dict[int, List[int]] = group_by_values(zip(*idx))

        # Return a 2D array with all valid core/ligand pairs
        items = pair_mapping.items()
        cor_lig_pairs = list(chain.from_iterable(
            ((k,) + n_tup for n_tup in combinations(v, n)) for k, v in items if len(v) >= n
        ))

        ret = np.array(cor_lig_pairs)
        try:
            ret[:, 1:] = j[ret[:, 1:]]
            ret[:, 0] = i[ret[:, 0]]
        except IndexError:
            if not idx[0].any():
                raise MoleculeError(f"No ligands found within a radius of {max_dist} Angstroem")
            else:
                raise MoleculeError(f"Not enough ligands found (>= {n}) within a radius of "
                                    "{max_dist} Angstroem")
        return ret

    def combinations(self, cor_lig_pairs: np.ndarray,
                     lig_mapping: Optional[IdxMapping] = None,
                     core_mapping: Optional[IdxMapping] = None) -> Set[CombinationsTuple]:
        """Create a list with all to-be removed atom combinations.

        Parameters
        ----------
        cor_lig_pairs : :class:`numpy.ndarray`
            An array with the indices of all core/ligand pairs.

        lig_mapping : :class:`Mapping<collections.abc.Mapping>`, optional
            A mapping for translating (1-based) atomic indices in `cor_lig_pairs[:, 0]` to
            lists of (1-based) atomic indices.
            Used for mapping ligand anchor atoms to the rest of the to-be dissociated ligands.

        core_mapping : :class:`Mapping<collections.abc.Mapping>`, optional
            A mapping for translating (1-based) atomic indices in `cor_lig_pairs[:, 1:]` to
            lists of (1-based) atomic indices.
            Used for mapping core atoms to the to-be dissociated sub structures.

        Returns
        -------
        :class:`set [:class:`tuple`]
            A set of 2-tuples.
            The first element of each tuple is a :class:`frozenset` with the (1-based) indices of
            all to-be removed core atoms.
            The second element contains a :class:`frozenset` with the (1-based) indices of
            all to-be removed ligand atoms.

        """
        c_map = core_mapping if core_mapping is not None else _DUMMY_GETTER
        l_map = lig_mapping if lig_mapping is not None else _DUMMY_GETTER

        # Switch from 0-based to 1-based indices
        pairs = cor_lig_pairs + 1
        if pairs.shape[1] == 2:
            pair_intersection = np.intersect1d(*pairs.T)
            if len(pair_intersection) == len(np.unique(pairs[:, 0])):
                self._core_is_lig = True

        # Commence the iteration!
        cores = pairs[:, 0]
        ligands = pairs[:, 1:]
        core_iterator = (frozenset(c_map[cor]) for cor in cores)
        lig_iterator = (frozenset(chain.from_iterable(l_map[i] for i in lig)) for lig in ligands)
        return set(zip(core_iterator, lig_iterator))

    """################################# Molecule dissociation #################################"""

    def __call__(self, combinations: Iterable[CombinationsTuple]) -> Iterator[Molecule]:
        """Get this party started."""
        # Extract instance variables
        mol: Molecule = self.mol

        # Construct new indices
        indices = self._get_new_indices()

        for core_idx, lig_idx in combinations:
            # Create a new molecule
            mol_new = mol.copy()
            s = mol_new.properties

            # Create a list of to-be removed atoms
            core: Atom = mol_new[next(iter(core_idx))]
            delete_at = {mol_new[i] for i in core_idx}
            delete_at.update(mol_new[i] for i in lig_idx)

            # Update the Molecule.properties attribute of the new molecule
            s.name = (s.name or 'mol') + '_wo_XYn'
            s.indices = indices
            s.job_path = []
            s.core_topology = f'{core.properties.topology}_{next(iter(core_idx))}'
            s.lig_residue = sorted({
                mol_new[i].properties.pdb_info.ResidueNumber for i in lig_idx
            })
            s.df_index: str = s.core_topology + ''.join(f' {i}' for i in s.lig_residue)

            for at in delete_at:
                mol_new.delete_atom(at)
            yield mol_new

    def _get_new_indices(self) -> List[int]:
        """Return an updated version of :attr:`MolDissociater.mol` ``.properties.indices``."""
        n: int = self.ligand_count
        mol: Molecule = self.mol

        if not mol.properties.indices:
            mol.properties.indices = indices = []
            return indices

        # Delete the indices of the last n ligands
        ret = mol.properties.indices.copy()
        for _ in range(n):
            del ret[-1]

        if self._core_is_lig:
            return ret

        # Delete the index of the last core atom
        core_max = int(self.core_idx[-1])
        idx = ret.index(core_max)
        del ret[idx]

        # Update the indices of all remaining ligands
        for i in ret:
            i -= 1
        return ret


def group_by_values(iterable: Iterable[Tuple[Any, Hashable]],
                    mapping_type: Type[MutableMapping] = dict
                    ) -> MutableMapping[Hashable, List[Any]]:
    """Take an iterable, yielding 2-tuples, and group all first elements by the second.

    Exameple
    --------
    .. code:: python

        >>> from typing import Dict, List

        >>> str_list = ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b']
        >>> iterable = enumerate(str_list)
        >>> new_dict: Dict[str, List[int]] = group_by_values(iterable)

        >>> print(new_dict)
        {'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8]}

    Parameter
    ---------
    iterable : :class:`Iterable<collections.abc.Iterable>`
        An iterable yielding 2 elements upon iteration
        (*e.g.* :meth:`dict.items` or :func:`enumerate`).
        The second element must be a :class:`Hashable<collections.abc.Hashable>` and will be used
        as key in the to-be returned dictionary.

    mapping_type : :class:`type` [:class:`MutableMapping<collections.abc.MutableMapping>`]
        The type of to-be returned (mutable) mapping.

    Returns
    -------
    :class:`MutableMapping<collections.abc.MutableMapping>`
    [:class:`Hashable<collections.abc.Hashable>`, :class:`list` [:data:`Any<typing.Any>`]]
        A grouped mapping.

    """
    ret: Mapping[Hashable, List[Any]] = mapping_type()
    ret_append: Dict[Hashable, list.append] = {}
    for value, key in iterable:
        try:
            ret_append[key](value)
        except KeyError:
            ret[key] = [value]
            ret_append[key] = ret[key].append
    return ret


def iter_repeat(iterable: Iterable[T], times: int) -> Iterator[T]:
    """Iterate over an iterable and apply :func:`itertools.repeat` to each element.

    Examples
    --------
    .. code:: python

        >>> iterable = range(3)
        >>> times = 2
        >>> iterator = iter_repeat(iterable, n)
        >>> for i in iterator:
        ...     print(i)
        0
        0
        1
        1
        2
        2

    Parameters
    ----------
    iterable : :class:`Iterable<collections.abc.Iterable>`
        An iterable.

    times : :class:`int`
        The number of times each element should be repeated.

    Returns
    -------
    :class:`Iterator<collections.abc.Iterator>`
        An iterator that yields each element from **iterable** multiple **times**.

    """
    return chain.from_iterable(repeat(i, times) for i in iterable)


def as_array(iterable: Iterable, dtype: Union[None, str, type, np.dtype] = None,
             copy: bool = False, ndmin: int = 0) -> np.ndarray:
    """Convert a generic iterable (including iterators) into a NumPy array.

    See :func:`numpy.array` for an extensive description of all parameters.

    """
    try:
        ret = np.array(iterable, dtype=dtype, copy=copy)
    except TypeError:  # **iterable** is an iterator
        ret = np.fromiter(iterable, dtype=dtype)

    if ret.ndim < ndmin:
        ret.shape += (1,) * (ndmin - ret.ndmin)
    return ret
