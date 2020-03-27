import numpy as np
from itertools import combinations
from typing import Dict, Tuple, List
from scm.plams import Molecule
from FOX import group_by_values
from nanoCAT.bde.guess_core_dist import guess_core_core_dist

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
    """
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

    Construct a nested dictionary {'atom_type1': {coord1: [indices], ...}, ...}
    """
    cn_dict = {}
    for k, v in idx_dict.items():
        cn = coord[v]
        mapping = {i: (v[cn == i] + 1).tolist() for i in np.unique(cn)}
        cn_dict[k] = mapping
    return cn_dict


def coordination_number(mol: Molecule, shell: str = 'inner', d_outer: float = None) -> NestedDict:
    """Full description
    mol = Molecule('perovskite.xyz')
    d_outer = 5.1
    """
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
            raise ValueError(f"user defined threshold radius required for the outer coordination shell")
        coord = coordination_outer(dist, d_outer, length)

    else:
        raise ValueError(f"'shell' expected to be 'inner' or 'outer'; observed value: '{shell}'")

    # Return the final dictionary
    return map_coordination(coord, idx_dict)
