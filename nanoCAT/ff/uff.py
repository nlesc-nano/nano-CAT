"""
nanoCAT.ff.uff
==============

Functions related to the Universal force field (UFF).

Index
-----
.. currentmodule:: nanoCAT.ff.uff
.. autosummary::
    UFF_DF
    _CSV
    combine_xi
    combine_di

API
---
.. autodata:: UFF_DF
    :annotation: = <pandas.core.frame.DataFrame object>

.. autodata:: _CSV
    :annotation: = <str object>

.. autofunction:: combine_xi
.. autofunction:: combine_di

"""

from os.path import join

import pandas as pd

import nanoCAT

__all__ = ['UFF_DF', 'combine_xi', 'combine_di']

#: Absolute path to the ``nanoCAT.data.uff`` .csv file.
_CSV: str = join(nanoCAT.__path__[0], 'data', 'uff.csv')

#: A DataFrame with UFF Lennard-Jones parameters.
#: Has access to the ``"xi"``, ``"di"`` and ``"psi"`` columns.
#: See :data:`_CSV` for the path to the corresponding .csv file.
UFF_DF: pd.DataFrame = pd.read_csv(_CSV, index_col=0, skiprows=10)


def combine_xi(a: str, b: str) -> float:
    r"""Return the arithmetic mean of two UFF Lennard-Jones distances (:math:`x_{i}`).

    .. math::

        x_{ab} = \frac {x_{a} + x_{b}}{2}

    Distances are pulled from the ``"xi"`` column in :data:`UFF_DF` based on the supplied
    atomic symbols (**a** and **b**).

    Paramaters
    ----------
    a : str
        The first atomic symbol.

    b : str
        The second atomic symbol.

    Raises
    ------
    KeyError:
        Raised if **a** and/or **b** cannot be found in the index of :data:`UFF_DF`.

    """
    try:
        x_a = UFF_DF.at[a, 'xi']
        x_b = UFF_DF.at[b, 'xi']
    except KeyError as ex:
        raise KeyError(f"No UFF parameters available for atom type '{ex.args[0]}'")
    return (x_a + x_b) / 2


def combine_di(a: str, b: str) -> float:
    r"""Return the root of the product of two Lennard-Jones well depths (:math:`D_{i}`).

    .. math::

        D_{ab} = \sqrt{D_{a} D_{b}}

    Well depts are pulled from the ``"di"`` column in :data:`UFF_DF` based on the supplied
    atomic symbols (**a** and **b**).

    Paramaters
    ----------
    a : str
        The first atomic symbol.

    b : str
        The second atomic symbol.

    Raises
    ------
    KeyError:
        Raised if **a** and/or **b** cannot be found in the index of :data:`UFF_DF`.

    """
    try:
        di_a = UFF_DF.at[a, 'di']
        di_b = UFF_DF.at[b, 'di']
    except KeyError as ex:
        raise KeyError(f"No UFF parameters available for atom type '{ex.args[0]}'")
    return (di_a * di_b)**0.5
