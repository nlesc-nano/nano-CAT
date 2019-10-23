"""
nanoCAT.bde.dissociate_xyn
==========================

A module for constructing :math:`XYn`-dissociated quantum dots.

Index
-----
.. currentmodule:: nanoCAT.bde.dissociate_xyn
.. autosummary::
    dissociate_ligand
    dissociate_ligand2
    remove_ligands
    filter_lig_core
    filter_lig_core2
    filter_core
    get_topology
    get_combinations

API
---
.. autofunction:: dissociate_ligand
.. autofunction:: dissociate_ligand2
.. autofunction:: remove_ligands
.. autofunction:: filter_lig_core
.. autofunction:: filter_lig_core2
.. autofunction:: filter_core
.. autofunction:: get_topology
.. autofunction:: get_combinations

"""

from itertools import (chain, combinations)
from typing import (Iterable, Tuple, Sequence, Dict, List, Optional)

import numpy as np
from scipy.spatial.distance import cdist

from scm.plams import (Molecule, Atom, Settings)

from nanoCAT.bde.guess_core_dist import guess_core_core_dist

__all__ = ['dissociate_ligand']


def dissociate_ligand(mol: Molecule, workflow: 'WorkFlow') -> List[Molecule]:
    """Create all XYn dissociated quantum dots.

    Parameter
    ---------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    settings : |plams.Settings|_
        A settings object containing all (optional) arguments.

    Returns
    -------
    |list|_ [|plams.Molecule|_]
        A list of XYn dissociated quantum dots.

    """
    # Unpack arguments
    atnum = workflow.core_atom
    l_count = workflow.lig_count
    cc_dist = workflow.core_core_dist
    lc_dist = workflow.lig_core_dist
    top_dict = workflow.topology

    # Parameter not provided, just guess it
    if not cc_dist:
        cc_dist = guess_core_core_dist(mol, atnum)

    # Convert **mol** to an XYZ array
    mol.set_atoms_id()
    xyz_array = mol.as_array()

    # Create a nested list of atoms,
    # each nested element containing all atoms with a given residue number
    res_list = gather_residues(mol)

    # Create a list of all core indices and ligand anchor indices
    idx_c_old = np.array([j for j, at in enumerate(res_list[0]) if at.atnum == atnum])
    idx_c, topology = filter_core(xyz_array, idx_c_old, top_dict, cc_dist)
    idx_l = np.array(get_anchor_idx(mol)) - 1

    # Mark the core atoms with their topologies
    for i, top in zip(idx_c_old, topology):
        mol[int(i+1)].properties.topology = top

    # Create a dictionary with core indices as keys and all combinations of 2 ligands as values
    xy = filter_lig_core(xyz_array, idx_l, idx_c, lc_dist, l_count)
    combinations_dict = get_combinations(xy, res_list, l_count)

    # Create and return new molecules
    indices = [at.id for at in res_list[0][:-l_count]]
    indices += (idx_l[:-l_count] + 1).tolist()
    return remove_ligands(mol, combinations_dict, indices)


def dissociate_ligand2(mol: Molecule, workflow: 'WorkFlow') -> List[Molecule]:
    """Create all XYn dissociated quantum dots.

    Parameter
    ---------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    settings : |plams.Settings|_
        A settings object containing all (optional) arguments.

    Returns
    -------
    |list|_ [|plams.Molecule|_]
        A list of XYn dissociated quantum dots.

    """
    # Unpack arguments
    l_count = workflow.lig_count
    cc_dist = workflow.core_core_dist
    idx_c_old = np.array(workflow.core_index) - 1
    top_dict = workflow.topology

    # Convert **mol** to an XYZ array
    mol.set_atoms_id()
    xyz_array = mol.as_array()

    # Create a list of all core indices and ligand anchor indices
    _, topology = filter_core(xyz_array, idx_c_old, top_dict, cc_dist)

    # Mark the core atoms with their topologies
    for i, top in zip(workflow.core_index, topology):
        mol[i].properties.topology = top

    # Create a dictionary with core indices as keys and all combinations of 2 ligands as values
    res_list = gather_residues(mol)
    anchor_idx = get_anchor_idx(mol)
    xy = filter_lig_core2(xyz_array, anchor_idx, idx_c_old, l_count)
    combinations_dict = get_combinations(xy, res_list, l_count)

    # Create and return new molecules
    _anchor_idx = (1 + anchor_idx).tolist()
    indices = [at.id for at in res_list[0][:-l_count]] + _anchor_idx[:-l_count]
    return remove_ligands(mol, combinations_dict, indices)


def get_anchor_idx(mol: Molecule) -> np.ndarray:
    """Create a list of (1-based) indices of all ligand anchor atoms."""
    list_ = [i for i in mol.properties.indices if mol[i].properties.pdb_info.ResidueName == 'LIG']
    ret = np.array(list_)
    ret -= 1
    return ret


def gather_residues(mol: Molecule) -> List[List[Atom]]:
    """Create a nested list of atoms using their residue number."""
    ret = []
    for at in mol:
        i = at.properties.pdb_info.ResidueNumber - 1
        try:
            ret[i].append(at)
        except IndexError:
            ret.append([at])
    return ret


def filter_lig_core2(xyz_array: np.ndarray, idx_lig: Sequence[int],
                     idx_core: Sequence[int], lig_count: int = 2) -> np.ndarray:
    """Create and return the indices of all possible ligand/core pairs.

    Parameters
    ----------
    xyz_array : :math:`n*3` |np.ndarray|_ [|np.float64|_]
        An array with the cartesian coordinates of a molecule with *n* atoms.

    idx_lig : |np.ndarray|_ [|np.int64|_]
        An array of all ligand anchor atoms (Y).

    idx_core : |np.ndarray|_ [|np.int64|_]
        An array of all core atoms (X).

    max_dist : float
        The maximum distance for considering :math:`XY_{n}` pairs.

    lig_count : int
        The number of ligand (*n*) in :math:`XY_{n}`.

    Returns
    -------
    :math:`m*2` |np.ndarray|_ [|np.int64|_]
        An array with the indices of all :math:`m` valid  ligand/core pairs.

    """
    dist = cdist(xyz_array[idx_lig], xyz_array[idx_core])
    xy = []
    for _ in range(lig_count):
        xy.append(np.array(np.where(dist == np.nanmin(dist, axis=0))))
        dist[xy[-1][0], xy[-1][1]] = np.nan
    xy = np.hstack(xy)
    xy = xy[[1, 0]]
    xy = xy[:, xy.argsort(axis=1)[0]]

    bincount = np.bincount(xy[0])
    xy = xy[:, [i for i, j in enumerate(xy[0]) if bincount[j] >= lig_count]]
    xy[0] = idx_core[xy[0]]
    xy[1] += 1
    return xy


def get_fragment(mol: Molecule, atom: Atom) -> List[int]:
    ret = []
    atom._visited = True

    def dfs(at: Atom) -> None:
        for bond in at.bonds:
            at_new = bond.other_end(at)
            if hasattr(at_new, '_visited'):
                continue
            ret.append(at_new.id)
            at_new._visited = True
            dfs(at_new)

    dfs(atom)
    for at in mol:
        del at._visited
    return ret


def remove_ligands(mol: Molecule, combinations_dict: dict,
                   indices: Sequence[int]) -> List[Molecule]:
    """ """
    ret = []
    mol.set_atoms_id()
    for core in combinations_dict:
        for lig in combinations_dict[core]:
            mol_tmp = mol.copy()
            mol_tmp.properties = prop = Settings()

            prop.indices = indices
            prop.lig_residue = sorted([mol[i[0]].properties.pdb_info.ResidueNumber for i in lig])
            prop.job_path = []
            prop.core_topology = f'{str(mol[core].properties.topology)}_{core}'
            prop.df_index = (mol_tmp.properties.core_topology +
                             ' '.join(str(i) for i in mol_tmp.properties.lig_residue))
            prop.name = mol.properties.name + '_wo_XYn'
            prop.path = mol.properties.path
            prop.prm = mol.properties.prm

            delete_idx = [core]
            delete_idx += chain.from_iterable(lig)
            core_at = mol_tmp[core]
            if core_at.bonds:
                delete_idx += get_fragment(mol_tmp, core_at)
            delete_idx.sort(reverse=True)

            for i in delete_idx:
                mol_tmp.delete_atom(mol_tmp[i])

            ret.append(mol_tmp)
    mol.unset_atoms_id()
    return ret


def filter_core(xyz_array: np.ndarray,
                idx: np.ndarray,
                topology_dict: Dict[int, str] = {6: 'vertice', 7: 'edge', 9: 'face'},
                max_dist: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Find all atoms (**idx**) in **xyz_array** which are exposed to the surface.

    A topology is assigned to aforementioned atoms based on the number of neighbouring atoms.

    Parameters
    ----------
    xyz_array : :math:`n*3` |np.ndarray|_ [|np.float64|_]
        An array with the cartesian coordinates of a molecule with :math:`n` atoms.

    idx : |np.ndarray|_ [|np.int64|_]
        An array of atomic indices in **xyz_array**.

    topology_dict : |dict|_ [|int|_, |str|_]
        A dictionary which maps the number of neighbours (per atom) to a user-specified topology.

    max_dist : float
        The radius (Angstrom) for determining if an atom counts as a neighbour or not.

    Returns
    -------
    |np.ndarray|_ [|np.int64|_] and |np.ndarray|_ [|np.int64|_]
        The indices of all atoms in **xyz_array[idx]** exposed to the surface and
        the topology of atoms in **xyz_array[idx]**.

    """
    # Create a distance matrix and find all elements with a distance smaller than **max_dist**
    dist = cdist(xyz_array[idx], xyz_array[idx])
    np.fill_diagonal(dist, max_dist)
    xy = np.array(np.where(dist <= max_dist))
    bincount = np.bincount(xy[0], minlength=len(idx))

    # Slice xyz_array, creating arrays of reference atoms and neighbouring atoms
    x = xyz_array[idx]
    y = xyz_array[idx[xy[1]]]

    # Calculate the length of a vector from a reference atom to the mean position of its neighbours
    # A vector length close to 0.0 implies that a reference atom is surrounded by neighbours in
    # a more or less spherical pattern (i.e. the reference atom is in the bulk, not on the surface)
    vec_length = np.empty((bincount.shape[0], 3), dtype=float)
    k = 0
    for i, j in enumerate(bincount):
        vec_length[i] = x[i] - np.average(y[k:k+j], axis=0)
        k += j

    vec_norm = np.linalg.norm(vec_length, axis=1)
    return idx[np.where(vec_norm > 0.5)[0]], get_topology(bincount, topology_dict)


def get_topology(bincount: Iterable[int],
                 topology_dict: Optional[Dict[int, str]] = None) -> List[str]:
    """Translate the number of neighbouring atoms (**bincount**) into a list of topologies.

    If a specific number of neighbours (*i*) is absent from **topology_dict** then that particular
    element is set to a generic str(*i*) + '_neighbours'.

    Parameters
    ----------
    bincount : :math:`n` |np.ndarray|_ [|np.int64|_]
        An array with the number of neighbours per atom for a total of :math:`n` atoms.

    topology_dict : |dict|_ [|int|_, |str|_]
        Optional: A dictionary which maps the number of neighbours (per atom) to
        a user-specified topology.

    Returns
    -------
    :math:`n` |list|_
        A list of topologies for all :math:`n` atoms in **bincount**.

    """
    if isinstance(topology_dict, Settings):
        dict_ = topology_dict.as_dict()
    elif topology_dict is None:
        dict_ = {}
    else:
        dict_ = topology_dict.copy()

    return [(dict_[i] if i in dict_ else f'{i}_neighbours') for i in bincount]


def filter_lig_core(xyz_array: np.ndarray,
                    idx_lig: Sequence[int],
                    idx_core: Sequence[int],
                    max_dist: float = 5.0,
                    lig_count: int = 2) -> np.ndarray:
    """Create and return the indices of all possible ligand/core atom pairs.

    Ligand/core atom pair construction is limited to a given radius (**max_dist**).

    Parameters
    ----------
    xyz_array : :math:`n*3` |np.ndarray|_ [|np.float64|_]
        An array with the cartesian coordinates of a molecule with *n* atoms.

    idx_lig : |np.ndarray|_ [|np.int64|_]
        An array of all ligand anchor atoms (Y).

    idx_core : |np.ndarray|_ [|np.int64|_]
        An array of all core atoms (X).

    max_dist : float
        The maximum distance for considering :math:`XY_{n}` pairs.

    lig_count : int
        The number of ligand (*n*) in :math:`XY_{n}`.

    Returns
    -------
    :math:`m*2` |np.ndarray|_ [|np.int64|_]
        An array with the indices of all :math:`m` valid ligand/core pairs
        (as determined by **max_dist**).

    """
    dist = cdist(xyz_array[idx_core], xyz_array[idx_lig])
    _xy = np.array(np.where(dist <= max_dist))
    bincount = np.bincount(_xy[0])

    xy = _xy[:, [i for i, j in enumerate(_xy[0]) if bincount[j] >= lig_count]]
    xy[0] = idx_core[xy[0]]
    xy[1] += 1
    return xy


def get_combinations(xy: np.ndarray, res_list: Sequence[Sequence[Atom]],
                     lig_count: int = 2) -> dict:
    """Given an array of indices (**xy**) and a nested list of atoms **res_list**.

    Parameters
    ----------
    xy : :math:`m*2` |np.ndarray|_ [|np.int64|_]
        An array with the indices of all :math:`m` core/ligand pairs.

    res_list : |list|_ [|tuple|_ [|plams.Atom|_]]
        A list of PLAMS atoms, each nested tuple representing all atoms within a given residue.

    lig_count : int
        The number of ligand (*n*) in :math:`XY_{n}`.

    Returns
    -------
    |dict|_
        A dictionary with core/ligand combinations.

    """
    dict_ = {}
    for core, lig in xy.T:
        try:
            dict_[res_list[0][core].id].append([at.id for at in res_list[lig]])
        except KeyError:
            dict_[res_list[0][core].id] = [[at.id for at in res_list[lig]]]
    return {k: combinations(v, lig_count) for k, v in dict_.items()}
