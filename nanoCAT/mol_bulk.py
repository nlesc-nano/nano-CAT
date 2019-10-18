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
from scipy.spatial.distance import cdist

from scm.plams import Molecule

from CAT.logger import logger
from CAT.settings_dataframe import SettingsDataFrame

__all__ = ['init_lig_bulkiness']

# Aliases for DataFrame column keys
MOL: Tuple[str, str] = ('mol', '')
V_BULK: Tuple[str, str] = ('V_bulk', '')


def init_lig_bulkiness(qd_df: SettingsDataFrame, ligand_df: SettingsDataFrame,
                       core_df: SettingsDataFrame) -> None:
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

    See Also
    --------
    `Ligand cone angle <https://en.wikipedia.org/wiki/Ligand_cone_angle>`_:
        The ligand cone angle is a measure of the steric bulk of a ligand in
        a transition metal complex.

    """
    write = ligand_df.settings.optional.database.write
    logger.info('Starting ligand bulkiness calculations')

    V_list = []
    for (i, j, k, l) in qd_df.index:
        # Extract the core and ligand
        ij = (i, j)
        kl = (k, l)
        core = core_df.at[ij, MOL]
        ligand = ligand_df.at[kl, MOL]

        # Calculate V_bulk
        angle, r_ref = get_core_angle(core)
        r, h = get_lig_radius(ligand)
        V_bulk = get_V(r, h, r_ref, angle)
        V_list.append(V_bulk)

    logger.info('Finishing ligand bulkiness calculations\n')
    qd_df[V_BULK] = V_list

    if 'ligand' in write:
        _export_to_db(qd_df)


def _export_to_db(qd_df: SettingsDataFrame) -> None:
    """Export the ``"V_bulk"`` column in **ligand_df** to the database."""
    settings = qd_df.settings.optional
    overwrite = 'qd' in settings.database.overwrite

    db = settings.database.db
    db.update_csv(
        qd_df, database='QD', columns=['V_bulk'], overwrite=overwrite
    )


def get_lig_radius(ligand: Molecule, anchor: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    r"""Return all half-cone angles defined according to :math:`\phi = \angle_{ABC}`.

    :math:`A` represents a set of all ligand atoms,
    :math:`B` is the ligand anchor atom and
    :math:`C` is the mean position of all ligand atoms (*i.e.* the ligand center).

    Parameters
    ----------
    ligand : :math:`n` |plams.Molecule|
        A ligand molecule with :math:`n` atoms.

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
    i = _get_anchor_idx(ligand) if anchor is None else anchor
    xyz = np.array(ligand)

    # Calculate distance the height (h) and radius (r)
    h = xyz[:, 0]
    r = np.linalg.norm(xyz[:, 1:], axis=1)
    r += [at.radius for at in ligand]  # Correct for atomic radii
    r[i] = 0.0
    return r, h


def get_core_angle(core: Molecule) -> Tuple[float, float]:
    """Return the mean."""
    # Find all nearest anchor neighbours
    anchors = np.array([at.coords for at in core.properties.dummies])
    dist = cdist(anchors, anchors)
    np.fill_diagonal(dist, 10000)
    idx = np.argmin(dist, axis=0)

    # Construct (and normalize) vectors from the center of mass to the anchor atoms
    center = np.array(core.get_center_of_mass())
    vec1 = anchors - center
    vec2 = anchors[idx] - center
    vec1 /= np.linalg.norm(vec1, axis=1)[..., None]
    vec2 /= np.linalg.norm(vec2, axis=1)[..., None]

    # Calculate (and average) all the anchor-center-anchor angles
    r_ref = np.linalg.norm(anchors - anchors[idx], axis=1)
    dot = np.einsum('ij,ij->i', vec1, vec2)
    return np.arccos(dot).mean(), r_ref.mean()


def _get_anchor_idx(mol: Molecule) -> int:
    """Return the index of the anchor atom specified |Molecule.properties| ``["anchor"]``."""
    anchor_str = mol.properties.anchor
    try:  # Is it an integer?
        return int(anchor_str)
    except ValueError:
        pass

    # Is it a string containing an integer?
    for i, _ in enumerate(anchor_str):
        try:
            idx = int(anchor_str[i:])
        except ValueError:
            pass
        else:
            return idx - 1
    raise ValueError   # I give up


def get_V(radius_array: np.ndarray, height_array: np.ndarray,
          d: float, angle: float) -> float:
    r"""Calculate the :math:`V_{bulk}`, a ligand- and core-sepcific descriptor of a ligands' bulkiness.

    .. math::
        V(r_{i}, h_{i}; d, h_{lim}) =
        \sum_{i=1}^{n} e^{r_{i}} (\frac{2 r_{i}}{d} - 1)^{+} (1 - \frac{h_{i}}{h_{lim}})^{+}

    :math:`r` and :math:`h`, respectively, represent the radius and height of a "cylinder" centered
    around the ligand vector, the ligand anchor being the origin.
    Due to the conal (rather than cylindrical) shape of the ligand,
    the radius is substituted for an effective height- and angle-depedant radius: :math:`r^{eff}`.
    This effective radius reduces the repulsive force of the potential
    as :math:`h` grows larger, *i.e.* when atoms are positioned further away from
    the surface of the core.
    :math:`\phi` is the angle between the vectors of two neighbouring ligands,
    an angle of 0 reverting back to a cylindrical description of the ligand.

    .. _`array-like`: https://docs.scipy.org/doc/numpy/glossary.html#term-array-like

    Parameters
    ----------
    radius_array : array-like
        An `array-like`_ object representing the distance of an atom with respect to vector
        representing the molecules' orientation.
        Conceptually equivalent to a set of cylinder radii.

    height_array : array-like
        An `array-like`_ object representing the height of an atom along a vector
        representing the molecules' orientation.
        Conceptually equivalent to a set of cylinder heights.

    angle : :class:`float`
        The angle (in radian) between two ligand vectors.

    d : class`float`
        The average distance between two neighbouring core anchor atoms.
        Equivalent to the lattice spacing of the core.

    Returns
    -------
    :class:`float`
        The bulkiness factor :math:`V_{bulk}`.

    """  # noqa
    r = np.array(radius_array, dtype=float, ndmin=1, copy=True)
    h = np.array(height_array, dtype=float, ndmin=1, copy=False)

    d = 5.878
    step1 = (2 * r) / d - 1
    step2 = 1 - (h / 10)
    step1[step1 < 0] = 0
    step2[step2 < 0] = 0
    ret = step1 * step2 * np.exp(r)

    return ret.sum()