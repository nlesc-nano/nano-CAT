import numpy as np
from itertools import combinations
from typing import Dict, Tuple
from scm.plams import Molecule
from FOX import group_by_values
from nanoCAT.bde.guess_core_dist import guess_core_core_dist


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


def coordination_outer(dist: np.ndarray, d_outer: float) -> Dict[str, Dict[int, list]]:
    """Map atoms according to their atomic symbols and coordination number in the outer shell.

    Construct a nested dictionary with atomic symbols as keys and dictionaries {coord_outer: list of indices} as values.
    """
    # Compute the number of neighbors in the outer coordination shell of each atom
    a, b = np.where(dist <= d_outer)
    coord_outer = np.bincount(a, minlength=len(xyz)) + np.bincount(b, minlength=len(xyz))

    # Build a dictionary to class atoms according to their atomic symbols and coordination number in the outer shell
    cn_dict = {}
    for k, v in idx_dict.items():
        cn = coord_outer[v]
        mapping = {i: (v[cn == i] + 1).tolist() for i in np.unique(cn)}
        cn_dict[k] = mapping
    return cn_dict


def radius_inner(idx_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Search the threshold radius of the inner coordination shell for each atom type.

    This radius is calculated as the distance with the first neighbors:
    in case of heterogeneous coordinations, only the closest neighbors are thus considered.
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


def coordination_inner(dist: np.ndarray, d_inner: Dict[str, float]) -> Dict[str, Dict[int, list]]:
    """Map atoms according to their atomic symbols and coordination number in the inner shell.

    Construct a nested dictionary with atomic symbols as keys and dictionaries {coord_inner: list of indices} as values.
    """
    fn_dict = {}
    for k, v in idx_dict.items():
        # Compute the number of first neighbors (i.e. in the inner coordination shell) for each atom
        a, b = np.where(dist <= d_inner[k])
        coord_inner = np.bincount(a, minlength=len(xyz)) + np.bincount(b, minlength=len(xyz))

        # Build a dictionary to class atoms according to their atomic symbols and coordination number in the inner shell
        fn = coord_inner[v]
        mapping = {i: (v[fn == i] + 1).tolist() for i in np.unique(fn)}
        fn_dict[k] = mapping
    return fn_dict


mol = Molecule('perovskite.xyz')
d_outer = 5.1
xyz = np.asarray(mol)
idx_dict = get_indices(mol)
# Construct the upper distance matrix
x, y = idx_pairs(idx_dict)
shape = len(xyz), len(xyz)
dist = np.full(shape, fill_value=np.nan)
dist[x, y] = np.linalg.norm(xyz[x] - xyz[y], axis=1)
cn_dict = coordination_outer(dist, d_outer)
d_inner = radius_inner(idx_dict)
fn_dict = coordination_inner(dist, d_inner)
