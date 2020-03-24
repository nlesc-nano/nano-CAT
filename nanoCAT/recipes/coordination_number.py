import numpy as np
from itertools import combinations
from typing import Dict, Tuple
from scm.plams import Molecule
from FOX import group_by_values
from nanoCAT.bde.guess_core_dist import guess_core_core_dist

mol = Molecule('perovskite.xyz')
d_outer = 5.1 

xyz = np.asarray(mol)

# Construct a dictionary with atomic symbols as keys and arrays of indices as values
elements = [at.symbol for at in mol.atoms]
symbol_enumerator = enumerate(elements)
idx_dict = group_by_values(symbol_enumerator)
for k, v in idx_dict.items():
    idx_dict[k] = np.fromiter(v, dtype=int, count=len(v))

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

x, y =  idx_pairs(idx_dict)

# Construct the upper distance matrix
shape = len(xyz), len(xyz)
dist = np.full(shape, fill_value=np.nan)
dist[x, y]  = np.linalg.norm(xyz[x] - xyz[y], axis=1)

# Compute outer shell coordination number of each atom
a, b = np.where(dist <= d_outer)
coord_outer = np.bincount(a, minlength=len(xyz)) + np.bincount(b, minlength=len(xyz))

# Construct a nested dictionary with atomic symbols as keys and dictionaries {coord_outer: list of indices} as values
cn_dict = {}
for k, v in idx_dict.items():
     cn = coord_outer[v]
     mapping = {i: (v[cn == i] + 1).tolist() for i in np.unique(cn)}
     cn_dict[k] = mapping
     
# Search the threshold radii of the inner coordination shell for each atom type as the distance with the first neighbors
d_inner = {}
symbol_combinations = combinations(idx_dict.keys(), r=2)
for symbol1, symbol2 in symbol_combinations:
    d_pair = guess_core_core_dist(mol, (symbol1, symbol2))
    if d_pair < d_inner.get(symbol1, np.inf):
         d_inner[symbol1] = d_pair
    if d_pair < d_inner.get(symbol2, np.inf):
         d_inner[symbol2] = d_pair

# Construct a nested dictionary with atomic symbols as keys and dictionaries {coord_inner: list of indices} as values
fn_dict = {}
for k, v in idx_dict.items():
    a, b = np.where(dist <= d_inner[k])
    coord_inner = np.bincount(a, minlength=len(xyz)) + np.bincount(b, minlength=len(xyz))
    fn = coord_inner[v]
    mapping = {i: (v[fn == i] + 1).tolist() for i in np.unique(fn)}
    fn_dict[k] = mapping
