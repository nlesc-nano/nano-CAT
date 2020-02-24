"""
nanoCAT.bde.guess_core_dist
===========================

A module for estimating ideal values for ``["optional"]["qd"]["bde"]["core_core_dist"]`` .

Index
-----
.. currentmodule:: nanoCAT.bde.guess_core_dist
.. autosummary::
    guess_core_core_dist

API
---
.. autofunction:: guess_core_core_dist

"""

from typing import Union, Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

from scm.plams import Molecule, MoleculeError

from CAT.mol_utils import to_atnum, to_symbol
from FOX.functions.rdf import get_rdf_lowmem as get_rdf

__all__ = ['guess_core_core_dist']


def guess_core_core_dist(mol: Union[Molecule, np.ndarray],
                         atom: Optional[Union[str, int]] = None,
                         dr: float = 0.1,
                         r_max: float = 8.0,
                         window_length: int = 21,
                         polyorder: int = 7) -> float:
    """Guess a value for the ``["optional"]["qd"]["bde"]["core_core_dist"]`` parameter in **CAT**.

    The estimation procedure involves finding the first minimum in the
    radial distribution function (RDF) of **mol**.
    After smoothing the RDF wth a Savitzky-Golay filer, the gradient of the RDF
    is explored (starting from the RDFs' global maximum) until a stationary point is found with
    a positive second derivative (*i.e.* a minimum).

    Parameters
    ----------
    mol : |plams.Molecule| or :class:`numpy.ndarray`
        A molecule.

    atom : :class:`str` or :class:`int`
        An atomic number or symbol for defining an atom subset within **mol**.
        The RDF is constructed for this subset.

    dr : :class:`float`
        The RDF integration step-size in Angstrom, *i.e.* the distance between concentric spheres.

    r_max : :class:`float`
        The maximum to be evaluated interatomic distance in the RDF.

    window_length : :class:`int`
        The length of the filter window (*i.e.* the number of coefficients) for
        the Savitzky-Golay filter.

    polyorder : :class:`int`
        The order of the polynomial used to fit the samples for the Savitzky-Golay filter.

    Returns
    -------
    :class:`float`
        The interatomic radius of the first RDF minimum (following the first maximum).

    Raises
    ------
    MoleculeError
        Raised if **atom** is not in **mol**.

    ValueError
        Raised if no minimum is found in the smoothed RDF.

    See Also
    --------
    :func:`savgol_filter()<scipy.signal.savgol_filter>`
        Apply a Savitzky-Golay filter to an array.

    """
    if atom is not None:
        atnum = to_atnum(atom)
        xyz = mol.as_array(atom_subset=(at for at in mol if at.atnum == atnum))
        if not xyz.any():
            raise MoleculeError(f"No atoms with atomic symbol '{to_symbol(atom)}' "
                                f"in '{mol.get_formula()}'")
    else:
        xyz = np.asarray(mol)

    # Create a disance matrix
    dist = cdist(xyz, xyz)

    # Create and smooth the RDF
    rdf = get_rdf(dist, dr=dr, r_max=r_max)
    rdf[0] = 0.0
    rdf_smooth = savgol_filter(rdf, window_length, polyorder)

    # Find the global maximum and calculate the gradient of the smoothed RDF
    idx_max = rdf_smooth.argmax()
    grad = np.gradient(rdf_smooth[idx_max:])

    # Find and return the first point after the global maximum where the gradient is ~0
    # And the second derivative is positive
    for i, (j, k) in enumerate(zip(grad, grad[1:])):
        if not (j <= 0 and k >= 0):
            continue

        # Interpolate and return
        idx = (0 - j) / (k - j)
        return dr * (idx_max + i + idx)

    raise ValueError("No minimum found in the (smoothed) radial distribution function of 'mol'")
