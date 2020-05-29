"""
nanoCAT.bde.identify_surface
============================

A module for identifying which atoms are located on the surface, rather than in the bulk

Index
-----
.. currentmodule:: nanoCAT.bde.identify_surface
.. autosummary::
    identify_surface
    identify_surface_ch

API
---
.. autofunction:: identify_surface
.. autofunction:: identify_surface_ch

"""

import operator
from typing import Optional, Union, Callable

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from scm.plams import Molecule

from CAT.attachment.edge_distance import to_convex
from .guess_core_dist import guess_core_core_dist

__all__ = ['identify_surface', 'identify_surface_ch']

CompareFunc = Callable[[np.ndarray, float], np.ndarray]


def identify_surface_ch(mol: Union[Molecule, np.ndarray],
                        n: float = 0.5,
                        invert: bool = False) -> np.ndarray:
    """Identify the surface of **mol** using a convex hull-based approach.

    A convex hull represents the smallest set of points enclosing itself, thus defining a surface.

    Parameters
    ----------
    mol : array-like [:class:`float`], shape :math:`(n, 3)`
        A 2D array-like object of Cartesian coordinates representing a polyhedron.
        The supplied polyhedron should be convex in shape.

    n : :class:`float`
        Smoothing factor for constructing a convex hull.
        Should obey :math:`0 <= n <= 1`.
        Represents the degree of displacement of all atoms to a spherical surface;
        :math:`n = 1` is a complete projection while :math:`n = 0` means no displacement at all.

        A non-zero value is generally recomended here,
        as the herein utilized :class:`ConvexHull<scipy.spatial.ConvexHull>` class
        requires an adequate degree of surface-convexness,
        lest it fails to properly identify all valid surface points.

    invert : :class:`bool`
        If ``True``, return the indices of all atoms in the bulk rather than on the surface.

    Returns
    -------
    :class:`numpy.ndarray` [:class:`int`], shape :math:`(n,)`
        The (0-based) indices of all atoms in **mol** located on the surface.

    See Also
    --------
    :class:`ConvexHull<scipy.spatial.ConvexHull>`
        Convex hulls in N dimensions.

    """
    xyz = np.asarray(mol)

    # Need at least 4 points to construct a convex hull
    if len(xyz) < 4:
        return np.arange(len(xyz))

    xyz_convex = xyz if n == 0 else to_convex(xyz, n=n)
    idx = ConvexHull(xyz_convex).vertices

    if not invert:
        return idx
    else:
        bool_ar = np.ones(len(xyz), dtype=bool)
        bool_ar[idx] = False
        return np.arange(len(xyz))[bool_ar]


def identify_surface(mol: Union[Molecule, np.ndarray],
                     max_dist: Optional[float] = None,
                     tolerance: float = 0.5,
                     compare_func: CompareFunc = operator.__gt__) -> np.ndarray:
    """Take a molecule and identify which atoms are located on the surface, rather than in the bulk.

    The function compares the position of all reference atoms in **mol** with its direct neighbors,
    the latter being defined as all atoms within a radius **max_dist**.
    The distance is then calculated between the reference atoms and the mean-position of its
    direct neighbours.
    A length of 0 means that the atom is surrounded in a spherical symmetric manner,
    *i.e.* it must be located in the bulk.
    Deviations from 0 conversely imply that an atom is located on the surface.

    Parameters
    ----------
    mol : array-like [:class:`float`], shape :math:`(n, 3)`
        An array-like object with the Cartesian coordinates of the molecule.

    max_dist : :class:`float`, optional
        The radius for defining which atoms constitute as neighbors.
        If ``None``, estimate this value using the radial distribution function of **mol**.

    tolerance : :class:`float`
        The tolerance for considering atoms part of the surface.
        A higher value will impose stricter criteria,
        which might be necasary as the local symmetry of **mol** becomes less pronounced.
        Should be in the same units as the coordinates of **mol**.

    compare_func : :data:`Callable<typing.Callable>`
        The function for evaluating the direct-neighbor distance.
        The default, :func:`__gt__()<operator.__gt__>`, is equivalent to identifying the surface,
        while *e.g.* :func:`__lt__()<operator.__lt__>` identifies the bulk.

    Returns
    -------
    :class:`numpy.ndarray` [:class:`int`], shape :math:`(n,)`
        The (0-based) indices of all atoms in **mol** located on the surface.

    Raises
    ------
    :exc:`ValueError`
        Raised if no atom-pairs are found within the distance **max_dist**.
        Implies that either the user-specified or guessed value is too small.

    See Also
    --------
    :func:`guess_core_core_dist()<nanoCAT.bde.guess_core_dist.guess_core_core_dist>`
        Estimate a value for **max_dist** based on the radial distribution function of **mol**.

    """  # noqa
    xyz = np.asarray(mol)
    if max_dist is None:
        max_dist = guess_core_core_dist(mol)

    # Construct the distance matrix and fill the diagonal
    dist = cdist(xyz, xyz)
    np.fill_diagonal(dist, np.nan)
    with np.errstate(invalid='ignore'):
        x, y = np.where(dist <= max_dist)
    if not x.any():
        raise ValueError(f"No atom-pair found in 'mol' within a distance of {repr(max_dist)}"
                         " Angstrom")
    neighbour_count = np.bincount(x, minlength=len(xyz))

    # Slice xyz_array, creating arrays of reference atoms and neighbouring atoms
    neighbours = xyz[y]

    # Calculate the vector length from each reference atom to the mean position
    # of its neighbours
    # A vector length close to 0.0 implies that a reference atom is surrounded by neighbours in
    # a more or less spherical pattern:
    # i.e. the reference atom is in the bulk and not on the surface
    indices = np.zeros(len(neighbour_count), dtype=int)
    indices[1:] = np.cumsum(neighbour_count[:-1])

    with np.errstate(divide='ignore'):
        average = np.add.reduceat(neighbours, indices)
        average /= neighbour_count[:, None]
    vec = xyz - average

    vec_len = np.linalg.norm(vec, axis=1)
    return np.where(compare_func(vec_len, tolerance))[0]
