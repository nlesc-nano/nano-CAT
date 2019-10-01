"""
nanoCAT.mol_bulk
================

A module for calculating the bulkiness of ligands.

Index
-----
.. currentmodule:: nanoCAT.mol_bulk
.. autosummary::
    init_lig_bulkiness
    get_cone_angles
    get_V
    _export_to_db
    _get_anchor_idx

API
---
.. autofunction:: init_lig_bulkiness
.. autofunction:: get_cone_angles
.. autofunction:: get_V
.. autofunction:: _export_to_db
.. autofunction:: _get_anchor_idx

"""

from typing import Tuple, Optional

import numpy as np

from scm.plams import rotation_matrix, Molecule

from CAT.logger import logger
from CAT.settings_dataframe import SettingsDataFrame

__all__ = ['init_lig_bulkiness']

# Aliases for DataFrame column keys
MOL: Tuple[str, str] = ('mol', '')
V_BULK: Tuple[str, str] = ('V_bulk', '')


def init_lig_bulkiness(ligand_df: SettingsDataFrame) -> None:
    r"""Initialize the ligand bulkiness workflow.

    Given a set of angles :math:`\phi`, the bulkiness factor :math:`V_{bulk}` is defined below.
    Angles are constructed according to :math:`\phi = \angle_{ABC}`,
    where :math:`A` represents a set of all ligand atoms,
    :math:`B` is the ligand anchor atom and
    :math:`C` is the mean position of all ligand atoms (*i.e.* the ligand center).

    .. math::
        W_{bulk} = \frac{1}{n} \sum_{i}^{n} e^{\sin \phi_{i}}

    Conceptually, the bulkiness factor :math:`V_{bulk}` is related to ligand (half-)cones angles,
    with the key difference that :math:`V_{bulk}` builds on top of it,
    representing an estimate of mean inter-ligand steric interactions.

    Parameters
    ----------
    ligand_df : |CAT.SettingsDataFrame|
        A DataFrame of ligands.

    See also
    --------
    `Ligand cone angle <https://en.wikipedia.org/wiki/Ligand_cone_angle>`_:
        The ligand cone angle is a measure of the steric bulk of a ligand in
        a transition metal complex.

    """
    write = ligand_df.settings.optional.database.write
    V_list = []
    logger.info('Starting ligand bulkiness calculations')

    for mol in ligand_df[MOL]:
        angle_ar, height_ar = get_cone_angles(mol)
        V_bulk = get_V(angle_ar, height_ar)
        V_list.append(V_bulk)

    logger.info('Finishing ligand bulkiness calculations\n')
    ligand_df[V_BULK] = V_list

    if 'ligand' in write:
        _export_to_db(ligand_df)


def _export_to_db(ligand_df: SettingsDataFrame) -> None:
    """Export the ``"V_bulk"`` column in **ligand_df** to the database."""
    settings = ligand_df.settings.optional
    overwrite = 'ligand' in settings.database.overwrite

    db = settings.database.db
    db.update_csv(
        ligand_df, database='ligand', columns=['V_bulk'], overwrite=overwrite
    )


def get_cone_angles(mol: Molecule, anchor: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    r"""Return all half-cone angles defined according to :math:`\phi = \angle_{ABC}`.

    :math:`A` represents a set of all ligand atoms,
    :math:`B` is the ligand anchor atom and
    :math:`C` is the mean position of all ligand atoms (*i.e.* the ligand center).

    Parameters
    ----------
    mol : :math:`n` |plams.Molecule|
        A PLAMS molecule with :math:`n` atoms.


    anchor : :class:`int`, optional
        The index (0-based) of the anchor atom: :math:`B`.
        If ``None``, extract it from the |Molecule.properties| ``["anchor"]`` key.

    Returns
    -------
    :math:`n` :class:`numpy.ndarray` [:class:`float`]
        A 1D array with :math:`n` half-cone angles.
        Units are in radian.

    """
    # Find and return the anchor atom
    i = _get_anchor_idx(mol) if anchor is None else anchor
    xyz = np.array(mol)

    # Reset the origin and allign the ligand with the Cartesian x-axis
    _initial_vec = xyz.mean(axis=0) - xyz[i]
    _final_vec = np.array([1, 0, 0], dtype=float)
    rotmat = rotation_matrix(_initial_vec, _final_vec)
    xyz[:] = xyz@rotmat.T
    xyz -= xyz[i]

    # Calculate distance the height (h) and radius (r)
    h = xyz[:, 0]
    r = np.linalg.norm(xyz[:, 1:], axis=1)
    r += [at.radius for at in mol]  # Correct for atomic radii
    r[i] = 0.0

    # Calculate and return the angles
    with np.errstate(divide='ignore', invalid='ignore'):
        ret = np.arctan(r / h)
    ret[np.isnan(ret)] = 0.0
    return ret, h


def _get_anchor_idx(mol: Molecule) -> int:
    """Return the index of the anchor atom specified |Molecule.properties| ``["anchor"]``."""
    anchor_str = mol.properties.anchor
    try:
        return int(anchor_str)
    except ValueError:
        pass

    for i, _ in enumerate(anchor_str):
        try:
            idx = int(anchor_str[i:])
        except ValueError:
            pass
        else:
            return idx - 1


def get_V(angle_array: np.ndarray, height_array: np.ndarray) -> float:
    r"""Calculate the "bulkiness factor", :math:`V_{bulk}`, from an array of angles, :math:`\phi`.

    .. math::
        V_{bulk} = \sum_{i}^{n} \frac {e^{\sin \phi_{i}}} {h_{i}^2}

    The bulkiness factor, :math:`V_{bulk}`, makes the following two assumptions:

    * The inter-ligand distance is inversely proportional to the angle :math:`\phi_{i}`.
    * The inter-ligand repulsion (*i.e.* Pauli repulsion) is inversely proportional to the exponent
      of the inter-ligand distance.

    .. _`array-like`: https://docs.scipy.org/doc/numpy/glossary.html#term-array-like

    Parameters
    ----------
    angle_array : array-like
        An `array-like`_ object consisting of angles between :math:`0.0` and :math:`\pi` rad.

    height_array : array-like
        An `array-like`_ object representing the height of the cone matching **angle_array**.

    Returns
    -------
    :class:`float`
        The bulkiness factor :math:`V_{bulk}`.

    """
    # Convert into an array if required and change the unit to radian
    angle = np.array(angle_array, dtype=float, copy=True)
    height = np.array(height_array, dtype=float, copy=False)

    # Ensure that all angles are between 0.0 and pi
    if angle.min() < 0.0:
        angle[:] = np.abs(angle)
    if angle.max() > np.pi:
        angle[angle > np.pi] = np.pi
    if height.min() < 0.0:
        height[:] = np.abs(height)

    ret = np.exp(np.sin(angle))
    with np.errstate(divide='ignore', invalid='ignore'):
        ret /= height**2
    ret[ret == np.inf] = 0.0
    return ret.sum()
