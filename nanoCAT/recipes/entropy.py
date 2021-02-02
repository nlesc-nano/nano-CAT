"""
nanoCAT.recipes.entropy
=======================

A recipe for calculating the rotational and translational entropy.

Index
-----
.. currentmodule:: nanoCAT.recipes
.. autosummary::
    get_entropy

API
---
.. autofunction:: get_entropy

"""

from typing import Tuple, NamedTuple

from scm.plams import Molecule, Units
import numpy as np

__all__ = ['get_entropy']


class EntropyTuple(NamedTuple):
    S_trans: float
    S_rot: float


def get_entropy(mol: Molecule, temp: float = 298.15) -> Tuple[float, float]:
    """Calculate the translational of the passsed molecule.

    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule`
        A PLAMS molecule.
    temp : :class:`float`
        The temperature in Kelvin.

    Returns
    -------
    :class:`float` & :class:`float`
        Two floats respectively representing the translational and rotational entropy.
        Units are in kcal/mol/K

    """
    # Define constants (SI units)
    pi = np.pi
    kT = 1.380648 * 10**-23 * temp  # Boltzmann constant * temperature
    h = 6.6260701 * 10**-34  # Planck constant
    R = 8.31445  # Gas constant
    # Volume(1 mol ideal gas) / Avogadro's number
    V_Na = ((R * temp) / 10**5) / Units.constants['NA']

    mass: np.ndarray = np.array([at.mass for at in mol]) * 1.6605390 * 10**-27
    x, y, z = mol.as_array().T * 10**-10

    # Calculate the rotational partition function: q_rot
    inertia = np.array([
        [sum(mass*(y**2 + z**2)), -sum(mass*x*y), -sum(mass*x*z)],
        [-sum(mass*x*y), sum(mass*(x**2 + z**2)), -sum(mass*y*z)],
        [-sum(mass*x*z), -sum(mass*y*z), sum(mass*(x**2 + y**2))]
    ])
    inertia_product = np.product(np.linalg.eig(inertia)[0])
    q_rot = pi**0.5 * ((8 * pi**2 * kT) / h**2)**1.5 * inertia_product**0.5

    # Calculate the translational and rotational entropy in j/mol
    S_trans: np.float64 = 1.5 + np.log(V_Na * ((2 * pi * sum(mass) * kT) / h**2)**1.5)
    S_rot: np.float64 = 1.5 + np.log(q_rot)

    # Apply the unit
    ret: np.ndarray = np.array([S_trans, S_rot]) * R
    ret *= Units.conversion_ratio('kj/mol', 'kcal/mol') / 1000
    return EntropyTuple(ret.item(0), ret.item(1))
