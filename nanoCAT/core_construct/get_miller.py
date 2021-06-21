from itertools import combinations
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import ase.io.cif
from FOX import MultiMolecule


def _get_perp_vecs(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Construct a unit vector perpendicular to a set of triangular polygons."""
    v1 = p2 - p1
    v2 = p3 - p1
    vec = np.cross(v1, v2)
    vec /= np.linalg.norm(vec, axis=-1)[..., None]
    return vec


def _get_unique_vecs1(vec: npt.ArrayLike) -> np.ndarray:
    ret = np.array(vec)
    ret /= ret.min(axis=-1, where=(ret != 0), initial=1)[..., None]
    return np.unique(ret, axis=0)


def _set_unique_vecs2(vec: np.ndarray) -> None:
    for i, (_, j, k) in enumerate(vec):
        frac_j = Fraction(j).limit_denominator()
        frac_k = Fraction(k).limit_denominator()
        j_int = frac_j.denominator == 1
        k_int = frac_k.denominator == 1

        if j_int and k_int:
            continue
        elif not j_int and not k_int:
            if frac_j.denominator == frac_k.denominator:
                vec[i] *= frac_j.denominator
            else:
                import pdb; pdb.set_trace()
        elif not j_int:
            vec[i] *= frac_j.denominator
        elif not k_int:
            vec[i] *= frac_k.denominator


def get_vecs(mol: MultiMolecule) -> np.ndarray:
    n = mol.shape[1]
    lst = []
    for i, j, k in combinations(range(n), 3):
        vec_i, vec_j, vec_k = mol[0, i], mol[0, j], mol[0, k]
        out = _get_perp_vecs(vec_i, vec_j, vec_k)
        if np.isnan(out).any():
            continue
        else:
            out = np.abs(out)
        out_lst = sorted(out)
        lst.append(out_lst)

    ret = _get_unique_vecs1(lst)
    _set_unique_vecs2(ret)
    return ret



cif_file = r"C:\Users\hardd\Downloads\PbS layered corrected.cif"

ase_mol = next(ase.io.cif.read_cif(cif_file, slice(None)))
mol = MultiMolecule.from_ase(ase_mol).delete_atoms("Pb")

vecs = get_vecs(mol)
print(vecs)
