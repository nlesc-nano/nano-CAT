from __future__ import annotations

import os
from collections.abc import Mapping, Iterable
from itertools import product, combinations, permutations
from typing import Union, TypeVar, Any, TYPE_CHECKING

import ase.io.cif
import numpy as np
from scipy.spatial import cKDTree
from scm.plams import Molecule, Atom, fromASE

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from numpy import (
        bool_ as b,
        int64 as i8,
        float64 as f8,
    )

_SCT = TypeVar("_SCT", bound=np.generic)

__all__ = ["parse_miller_keys", "get_surface_intersections", "get_interior_points"]


def parse_miller_keys(keys: Iterable[str | bytes] | Iterable[tuple[int, int, int]]) -> NDArray[i8]:
    """Parse and validate the passed miller indices."""
    ar = np.array([i for i in keys])
    if ar.dtype.kind in "US":
        if ar.dtype.itemsize != 3:
            raise ValueError("Miller indices must be of length 3")
        return ar.view(((ar.dtype.kind, 1), 3)).astype(np.int64)
    else:
        if ar.ndim != 2 or ar.shape[1] != 3:
            raise ValueError("Miller indices must be of length 3")
        return ar.astype(np.int64, copy=False, casting="same_kind")


def _filter_rank(coef: NDArray[f8], aug: NDArray[f8]) -> NDArray[b]:
    """Return a matrix-rank-based boolean mask for **coef** and **aug**."""
    r = np.fromiter([np.linalg.matrix_rank(i) for i in coef], np.int64, count=len(coef))
    r_prime = np.fromiter([np.linalg.matrix_rank(i) for i in aug], np.int64, count=len(aug))
    return (r == 3) & (r_prime == 3)


def _get_miller_combinations(miller: NDArray[_SCT]) -> NDArray[_SCT]:
    """Construct all combinations of three miller indices.

    Converts a ``(n, 3)`` array into a ``(n, 3, 3)`` array.
    """
    # Get all unique sign- and position-permutations of the Miller indices
    sign_perm = list(product([1, -1], repeat=3))
    pos_perm = list(permutations(range(3), 3))
    coef = (miller[:, pos_perm][..., None, :] * sign_perm).reshape(-1, 3)
    coef_unique = np.unique(coef, axis=0)

    # Create all possible combinations of 3 (quadrant) indices
    idx_comb = np.array(list(combinations(range(coef_unique.shape[0]), 3)))
    return coef_unique[idx_comb]


def get_surface_intersections(
    miller: NDArray[_SCT],
    radius: float,
) -> tuple[NDArray[f8], NDArray[_SCT]]:
    """Get the intersection points between all possible pairs of 3 Miller surfaces

    Parameters
    ----------
    miller : :class:`np.ndarray[np.int64] <numpy.ndarray>`, shape :math:`(n, 3)`
        An array of all to-be explored miller indices.
    radius : :class:`float`
        The radius.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m, 3)` & :class:`np.ndarray[np.int64] <numpy.ndarray>`shape :math:`(m, 3, 3)`
        The coordinates of all intersection points and the corresponding triplet of miller indices.

    """  # noqa: E501
    miller = np.array(miller, ndmin=2, copy=False)
    if miller.ndim != 2 or miller.shape[-1] != 3:
        raise ValueError("`miller` expected a (n, 3)-shaped array")

    # Construct all combinations of coefficient- and augmented coefficient matrices
    coef = _get_miller_combinations(miller)
    coef_norm = coef / np.linalg.norm(coef, axis=-1)[..., None]
    aug = np.empty(coef_norm.shape[:2] + (4,))
    aug[..., :3] = coef_norm
    aug[..., 3] = -np.sqrt(radius**2 / 3)

    # Evaluate the matrix ranks
    rank_eq_3 = _filter_rank(coef_norm, aug)
    coef_subset = coef_norm[rank_eq_3]
    aug_subset = aug[rank_eq_3]

    # Get the coordinates of the intersection points
    xyz = np.empty((aug_subset.shape[0], 3))
    xyz[..., 0] = np.linalg.det(aug_subset[..., [3, 1, 2]])
    xyz[..., 1] = np.linalg.det(aug_subset[..., [0, 3, 2]])
    xyz[..., 2] = np.linalg.det(aug_subset[..., [0, 1, 3]])
    xyz /= np.linalg.det(coef_subset)[..., None]

    xyz_ret, idx_ref = np.unique(xyz, axis=0, return_index=True)
    miller_ret = coef[rank_eq_3][idx_ref].astype(miller.dtype, copy=False)
    return xyz_ret, miller_ret


def get_interior_points(vertices: NDArray[f8], points: ArrayLike) -> NDArray[b]:
    """Return all **points** inside the polyhedron defined by **vertices**."""
    tree = cKDTree(vertices)
    _, idx = tree.query(points, 1)
    try:
        vec = points - vertices[idx]
    except IndexError:  # In case no nearest-neighbor is found
        return np.zeros(len(vertices), dtype=np.bool_)
    else:
        return (vec >= 0).all(axis=1)


def read_cif(file: str | bytes | os.PathLike[Any]) -> Molecule:
    """Read the passed .cif file."""
    with open(file, "r") as f:
        ase_mol = next(ase.io.cif.read_cif(f, np.s_[:]))
    return fromASE(ase_mol)


def get_supercell(mol: Molecule, r: float) -> Molecule:
    """Construct a supercell from **mol** that is at least as large as the radius **r**."""
    cell = np.asarray(mol.lattice)
    if not np.can_cast(cell, np.float64, casting="same_kind"):
        raise TypeError("`mol.lattice` expected an int or float array")
    elif cell.shape != (3, 3):
        raise ValueError("`mol.lattice` expected a (3, 3)-shaped array")

    vec_lengths = np.linalg.norm(cell, axis=1)
    n: int = np.ceil(vec_lengths / r).max().astype(np.int64).item()
    if (n / 2).is_integer():
        n += 1  # Ensure that `n` is odd

    # Translate the supercell so that it is centered on the original cell
    ret = mol.supercell(n, n, n)
    coords = np.array(ret)
    coords -= cell.sum(axis=1) * max(1, (n - 1) / 2)
    ret.from_array(coords)
    return ret


def cut_supercell(mol: Molecule, vertices: NDArray[f8]) -> Molecule:
    """Remove all atoms that lay outside the convex hull defined by **vertices**."""
    points = np.array(mol)
    mask = get_interior_points(vertices, points)

    ret = mol.copy()
    atoms = np.array(ret.atoms, dtype=np.object_)
    if not mol.bonds:
        ret.atoms = atoms[mask].tolist()
    else:
        for at in atoms[~mask]:
            ret.delete_atom(at)
    return ret
