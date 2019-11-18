"""
nanoCAT.bde.guess_core_dist
===========================

A module for estimating ideal values for ``["optional"]["qd"]["bde"]["core_core_dist"]`` .

Index
-----
.. currentmodule:: nanoCAT.bde.guess_core_dist
.. autosummary::
    guess_core_core_dist
    get_rdf

API
---
.. autofunction:: guess_core_core_dist
.. autofunction:: get_rdf

"""

from typing import Union

import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

from scm.plams import Molecule, MoleculeError

from CAT.mol_utils import to_atnum


def guess_core_core_dist(mol: Molecule,
                         atom: Union[str, int],
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

    .. _scipy.signal.savgol_filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

    Parameters
    ----------
    mol : |plams.Molecule|
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
    scipy.signal.savgol_filter_: Apply a Savitzky-Golay filter to an array.

    """  # noqa
    atnum = to_atnum(atom)

    # Create a disance matrix
    ar = mol.as_array(atom_subset=(at for at in mol if at.atnum == atnum))
    if not ar.any():
        raise MoleculeError(f"No atoms with atomic number/symbol '{atom}' in 'mol'")
    dist = cdist(ar, ar)

    # Create and smooth the RDF
    rdf = get_rdf(dist, dr, r_max)
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


def get_rdf(dist: np.ndarray, dr: float = 0.1, r_max: float = 8.0) -> np.ndarray:
    """Calculate and return the radial distribution function (RDF).

    Implementation based on the RDF generator in Auto-FOX_.

    .. _Auto-FOX: https://github.com/nlesc-nano/auto-FOX

    Parameters
    ----------
    dist : :math:`n*k` |np.ndarray|_ [|np.float64|_]
        A 2D array representing a distance matrix of :math:`n` by :math:`k` atoms.

    dr : float
        The integration step-size in Angstrom, *i.e.* the distance between concentric spheres.

    r_max : float
        The maximum to be evaluated interatomic distance.

    Returns
    -------
    |np.ndarray|_ [|np.float64|_]
        A 1D array of length :math:`1 + r_{max} / dr`: with a radial distribution function.

    """
    dist_int = np.array(dist / dr, dtype=int).ravel()
    r = np.arange(0, r_max + dr, dr)
    r_len = len(r)

    # Calculate the average particle density N / V
    # The diameter of the spherical volume (V) is defined by the largest inter-particle distance
    dens_mean = dist.shape[-1] / ((4/3) * np.pi * (0.5 * dist.max())**3)

    # Count the number of occurances of each (rounded) distance (i.e. particle count)
    dens = np.bincount(dist_int, minlength=r_len)[:r_len].astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        # Correct for the number of reference atoms
        dens /= dist.shape[1]

        # Convert the particle count into a partical density
        dens /= (4 * np.pi * r**2 * dr)

        # Normalize and return the particle density
        dens /= dens_mean
    dens[0] = 0.0
    return dens
