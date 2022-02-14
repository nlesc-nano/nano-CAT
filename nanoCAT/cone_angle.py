"""Function for computing ligand cone angles.

Index
-----
.. currentmodule:: nanoCAT.cone_angle
.. autosummary::
    get_cone_angle

API
---
.. autofunction:: get_cone_angle

"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scm.plams import Molecule, rotation_matrix

from CAT.workflows import WorkFlow, CONE_ANGLE
from CAT.settings_dataframe import SettingsDataFrame

if TYPE_CHECKING:
    from typing_extensions import SupportsIndex
    from numpy.typing import NDArray, ArrayLike
    from numpy import float64 as f8


def init_cone_angle(ligand_df: SettingsDataFrame) -> None:
    workflow = WorkFlow.from_template(ligand_df, name='cone_angle')
    workflow.keep_files = False
    dist = ligand_df.settings.optional.ligand.cone_angle.distance

    # Import from the database and start the calculation
    idx = workflow.from_db(ligand_df)
    if dist.ndim == 0:
        columns = pd.MultiIndex.from_tuples(
            [("cone_angle", f"dist={dist}")], names=ligand_df.columns.names
        )
    else:
        columns = pd.MultiIndex.from_tuples(
            [("cone_angle", f"dist={i}") for i in dist], names=ligand_df.columns.names
        )
    del ligand_df[CONE_ANGLE]
    for i in columns:
        ligand_df[i] = np.nan
    workflow(_start_cone_angle, ligand_df, index=idx, columns=columns, surface_dist=dist)

    # Export to the database
    workflow.to_db(ligand_df, index=idx, columns=columns)


def _start_cone_angle(
    lig_series: pd.Series,
    surface_dist: float = 0.0,
    **kwargs: Any,
) -> list[f8 | NDArray[f8]]:
    """Start the main loop for the ligand cone angle calculation."""
    ret = []
    for ligand in lig_series:
        anchor = ligand.atoms.index(ligand.properties.dummies)
        angle = get_cone_angle(ligand, anchor, surface_dist)
        ret.append(angle)
    return ret


@np.errstate(invalid="ignore")
def _get_angle(xyz: NDArray[f8]) -> NDArray[f8] | f8:
    """Return the maximum angle in ``xyz`` w.r.t. to the X-axis"""
    vecs = xyz.copy()
    vecs /= np.linalg.norm(vecs, axis=-1)[..., None]
    angles = np.arccos(vecs @ [1, 0, 0])
    return np.nanmax(angles, axis=-1)


def _minimize_func(vec: NDArray[f8], xyz: NDArray[f8], i: int) -> np.float64:
    """Rotate the X-axis in ``xyz`` to ``vec`` and \
    compute the maximum angle w.r.t. to the X-axis."""
    rotmat = rotation_matrix([1, 0, 0], vec)
    xyz_rot = xyz @ rotmat.T
    xyz_rot -= xyz_rot[i]
    return _get_angle(xyz_rot)


def _remove_anchor_hydrogens(mol: Molecule, anchor: int) -> tuple[NDArray[f8], int]:
    """Remove all hydrogen atoms connected to the anchor atom."""
    mol = mol.copy()
    anchor_at = mol.atoms[anchor]
    neighbors = anchor_at.neighbors()
    for at in neighbors:
        if at.atnum == 1:
            mol.delete_atom(at)
    return np.asarray(mol, dtype=np.float64), mol.atoms.index(anchor_at)


def get_cone_angle(
    mol: Molecule,
    anchor: SupportsIndex,
    surface_dist: ArrayLike = 0,
    *,
    remove_anchor_hydrogens: bool = False,
) -> NDArray[f8] | f8:
    r"""Compute the smallest enclosing cone angle in ``mol``.

    The smallest enclosing cone angle is herein defined as two times the largest angle
    (:math:`2 * \phi_{max}`) w.r.t. a central ligand vector, the ligand vector in turn being
    defined as the vector that minimizes :math:`\phi_{max}`.

    Note
    ----
    This function operates under one assumption:

    * The Cartesian X-axis is reasonable guess for the central ligand vector.

    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule`
        The molecule whose cone angle should be computed.
    anchor : :class:`int`
        The (0-based) index of the anchor atom.
    surface_dist : :class:`float`
        The distance of ``anchor`` w.r.t. to the surface of interest.
    remove_anchor_hydrogens : :class:`bool`
        If :data:`True`, remove all hydrogens connected to the anchor atom.

    Returns
    -------
    :class:`float` or :class:`np.ndarray[np.float64] <numpy.ndarray>`
        The cone angle(s) in degrees.

    """
    # Parse arguments
    surface_dist = np.asarray(surface_dist, dtype=np.float64)
    anchor = operator.index(anchor)
    if remove_anchor_hydrogens:
        xyz, anchor = _remove_anchor_hydrogens(mol, anchor)
    else:
        xyz = np.asarray(mol, dtype=np.float64)

    # Rotate the system such that the maximum angle w.r.t. the X-axis is minimized
    trial_vec = np.array([1, 0, 0], dtype=np.float64)
    output = minimize(_minimize_func, trial_vec, args=(xyz, anchor))
    vecs_opt = xyz @ rotation_matrix(trial_vec, output.x).T

    if surface_dist.ndim == 0:
        vecs_opt[..., 0] += surface_dist
    elif surface_dist.ndim == 1:
        shape = (len(surface_dist),) + vecs_opt.shape
        vecs_opt = np.full(shape, vecs_opt)
        vecs_opt[..., 0] += surface_dist[..., None]
    else:
        raise ValueError(f"Invalid `surface_dist` dimensionality: {surface_dist.ndim}")
    return np.degrees(2 * _get_angle(vecs_opt))
