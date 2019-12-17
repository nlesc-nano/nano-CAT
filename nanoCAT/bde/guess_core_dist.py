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
from FOX.functions.rdf import get_rdf_lowmem as get_rdf


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
