"""
nanoCAT.recipes.coordination_number
===================================

A recipe for calculating atomic coordination numbers.

Index
-----
.. currentmodule:: nanoCAT.recipes
.. autosummary::
    get_coordination_number
    coordination_outer

API
---
.. autofunction:: get_coordination_number
.. autofunction:: coordination_outer

"""

from typing import Dict, Tuple, List, Optional
from itertools import combinations

import numpy as np

from scm.plams import Molecule

from nanoutils import group_by_values
from nanoCAT.bde.guess_core_dist import guess_core_core_dist

__all__ = ['get_coordination_number', 'coordination_outer']

#: A nested dictonary
NestedDict = Dict[str, Dict[int, List[int]]]


def get_indices(mol: Molecule) -> Dict[str, np.ndarray]:
    """Construct a dictionary with atomic symbols as keys and arrays of indices as values."""
    elements = [at.symbol for at in mol.atoms]
    symbol_enumerator = enumerate(elements)
    idx_dict = group_by_values(symbol_enumerator)
    for k, v in idx_dict.items():
        idx_dict[k] = np.fromiter(v, dtype=int, count=len(v))
    return idx_dict


def idx_pairs(idx_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Construct two arrays of indice-pairs of all possible combinations in idx_dict.

    The combinations, by definition, do not contain any atom pairs where ``at1.symbol == at2.symbol``
    """  # noqa
    x, y = [], []
    symbol_combinations = combinations(idx_dict.keys(), r=2)
    for symbol1, symbol2 in symbol_combinations:
        idx1 = idx_dict[symbol1]
        idx2 = idx_dict[symbol2]
        _x, _y = np.meshgrid(idx1, idx2)
        x += _x.ravel().tolist()
        y += _y.ravel().tolist()

    x = np.fromiter(x, dtype=int, count=len(x))
    y = np.fromiter(y, dtype=int, count=len(y))
    return x, y


def upper_dist(xyz: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Construct the upper distance matrix."""
    shape = len(xyz), len(xyz)
    dist = np.full(shape, fill_value=np.nan)
    dist[x, y] = np.linalg.norm(xyz[x] - xyz[y], axis=1)
    return dist


def radius_inner(mol: Molecule, idx_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Search the threshold radius of the inner coordination shell for each atom type.

    This radius is calculated as the distance with the first neighbors:
    in case of heterogeneous coordinations, only the closest neighbors are considered.
    """
    d_inner = {}
    symbol_combinations = combinations(idx_dict.keys(), r=2)
    for symbol1, symbol2 in symbol_combinations:
        d_pair = guess_core_core_dist(mol, (symbol1, symbol2))
        if d_pair < d_inner.get(symbol1, np.inf):
            d_inner[symbol1] = d_pair
        if d_pair < d_inner.get(symbol2, np.inf):
            d_inner[symbol2] = d_pair
    return d_inner


def coordination_inner(dist: np.ndarray, d_inner: Dict[str, float],
                       idx_dict: Dict[str, np.ndarray], length: int) -> NestedDict:
    """Calculate the coordination number relative to the inner shell (first neighbors)."""
    coord_inner = np.empty(length, int)
    for k, v in idx_dict.items():
        a, b = np.where(dist <= d_inner[k])
        count = np.bincount(a, minlength=length) + np.bincount(b, minlength=length)
        coord_inner[v] = count[v]
    return coord_inner


def coordination_outer(dist: np.ndarray, d_outer: float, length: int) -> np.ndarray:
    """Calculate the coordination number relative to the outer shell."""
    a, b = np.where(dist <= d_outer)
    return np.bincount(a, minlength=length) + np.bincount(b, minlength=length)


def map_coordination(coord: np.ndarray, idx_dict: Dict[str, np.ndarray]) -> NestedDict:
    """Map atoms according to their atomic symbols and coordination number.

    Construct a nested dictionary ``{'atom_type1': {coord1: [indices], ...}, ...}``.
    """
    cn_dict = {}
    for k, v in idx_dict.items():
        cn = coord[v]
        mapping = {i: (v[cn == i] + 1).tolist() for i in np.unique(cn)}
        cn_dict[k] = mapping
    return cn_dict


def get_coordination_number(mol: Molecule, shell: str = 'inner',
                            d_outer: Optional[float] = None) -> NestedDict:
    """Take a molecule and identify the coordination number of each atom.

    The function first compute the pair distance between all reference atoms in **mol**.
    The number of first neighbors, defined as all atoms within a threshold radius
    **d_inner** is then count for each atom.
    The threshold radius can be changed to a desired value **d_outer** (in angstrom)
    to obtain higher coordination numbers associated to outer coordination shells.
    The function finally groups the (1-based) indices of all atoms in **mol**
    according to their atomic symbols and coordination numbers.

    Parameters
    ----------
    mol : array-like [:class:`float`], shape :math:`(n, 3)`
        An array-like object with the Cartesian coordinates of the molecule.

    shell : :class:`str`
        The coordination shell to be considered.
        Only ``'inner'`` or ``'outer'`` values are accepted.
        The default, ``'inner'``, refers to the first coordination shell.

    d_outer : :class:`float`, optional
        The threshold radius for defining which atoms are considered as neighbors.
        The default, ``None``, is accepted only if ``shell`` is ``'inner'``

    Returns
    -------
    :class:`dict`
        A nested dictionary ``{'Cd': {8: [0, 1, 2, 3, 4, ...], ...}, ...}``
        containing lists of (1-based) indices refered to the atoms in **mol**
        having a given atomic symbol (*e.g.* ``'Cd'``) and
        coordination number (*e.g.* ``8``).

    Raises
    ------
    :exc:`TypeError`
        Raised if no threshold radius is defined for the outer coordination shell.

    :exc:`ValueError`
        Raised if a wrong value is attributed to ``shell``.

    See Also
    --------
    :func:`guess_core_core_dist()<nanoCAT.bde.guess_core_dist.guess_core_core_dist>`
        Estimate a value for **d_inner** based on the radial distribution function of **mol**.
        Can also be used to estimate **d_outer** as the distance between the atom pairs ('A', 'B').

    """  # noqa
    # Return the Cartesian coordinates of **mol**
    xyz = np.asarray(mol)
    length = len(xyz)

    # Group atom indices according to atom symbols
    idx_dict = get_indices(mol)

    # Construct the upper distance matrix
    x, y = idx_pairs(idx_dict)
    dist = upper_dist(xyz, x, y)

    # Compute the coordination number
    if shell == 'inner':
        d_inner = radius_inner(mol, idx_dict)
        coord = coordination_inner(dist, d_inner, idx_dict, length)

    elif shell == 'outer':
        if d_outer is None:
            raise TypeError("user defined threshold radius required "
                            "for the outer coordination shell")
        coord = coordination_outer(dist, d_outer, length)

    else:
        raise ValueError(f"'shell' expected to be 'inner' or 'outer'; observed value: '{shell}'")

    # Return the final dictionary
    return map_coordination(coord, idx_dict)


coordination_number = get_coordination_number
