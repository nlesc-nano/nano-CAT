"""Recipes for filtering molecules.

Index
-----
.. currentmodule:: nanoCAT.recipes
.. autosummary::
    get_mol_length
    filter_mol
    filter_data

API
---
.. autofunction:: get_mol_length
.. autofunction:: filter_mol
.. autofunction:: filter_data

"""

from typing import Union, Iterable, Dict, TypeVar, Callable

import numpy as np
from scipy.spatial.distance import cdist
from scm.plams import Molecule, Atom

__all__ = ['get_mol_length', 'filter_mol', 'filter_data']

T = TypeVar('T')


def get_mol_length(mol: Union[np.ndarray, Molecule],
                   atom: Union[np.ndarray, Atom]) -> float:
    """Return the distance between **atom** and the atom in **mol** which it is furthest removed from.

    Examples
    --------
    Use the a molecules length for filtering a list of molecules:

    .. code:: python

        >>> from CAT.recipes import get_mol_length, filter_mol
        >>> from scm.plams import Molecule

        >>> mol_list = [Molecule(...), ...]
        >>> data = [...]
        >>> filter = lambda mol: get_mol_length(mol, mol.properties.get('anchor')) < 10

        >>> mol_dict = filter_mol(mol_list, data, filter=filter)

    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule` or :class:`numpy.ndarray`
        A PLAMS molecule or a 2D numpy array with a molecules Cartesian coordinates.

    atom : :class:`~scm.plams.mol.atom.Atom` or :class:`numpy.ndarray`
        A PLAMS atom or a 1D numpy array with an atoms Cartesian coordinates.

    Returns
    -------
    :class:`float`
        The largest distance between **atom** and all other atoms **mol**.

    See Also
    --------
    :func:`filter_mol`
        Filter **mol_list** and **data** based on elements from **mol_list**.

    """  # noqa: E501
    if isinstance(atom, Atom):
        atom_xyz = np.fromiter(atom.coords, 3, dtype=float)
        atom_xyz.shape = (1, 3)
    else:
        atom_xyz = np.asarray(atom, dtype=float).reshape((1, 3))

    dist = cdist(atom_xyz, mol)
    return dist.max()


def filter_mol(mol_list: Iterable[Molecule], data: Iterable[T],
               filter: Callable[[Molecule], bool]) -> Dict[Molecule, T]:
    """Filter **mol_list** and **data** based on elements from **mol_list**.

    Examples
    --------
    .. code:: python

        >>> from scm.plams import Molecule
        >>> from CAT.recipes import filter_mol

        >>> mol_list = [Molecule(...), ...]
        >>> data = [...]
        >>> mol_dict1 = filter_mol(mol_list, data, filter=lambda n: n < 10)

        >>> prop1 = [...]
        >>> prop2 = [...]
        >>> prop3 = [...]
        >>> multi_data = zip([prop1, prop2, prop3])
        >>> mol_dict2 = filter_mol(mol_list, multi_data, filter=lambda n: n < 10)

        >>> keys = mol_dict1.keys()
        >>> values = mol_dict1.values()
        >>> mol_dict3 = filter_mol(keys, values, filter=lambda n: n < 5)

    Parameters
    ----------
    mol_list : :class:`~collections.abc.Iterable` [:class:`~scm.plams.mol.molecule.Molecule`]
        An iterable of the, to-be filtered, PLAMS molecules.

    data : :class:`Iterable[T]<collections.abc.Iterable>`
        An iterable which will be assigned as values to the to-be returned dict.
        These parameters will be filtered in conjunction with **mol_list**.
        Note that **mol_list** and **data** *should* be of the same length.

    filter : :class:`Callable[[Molecule], bool]<collections.abc.Callable>`
        A callable for filtering the distance vector.
        An example would be :code:`lambda n: max(n) < 10`.

    Returns
    -------
    :class:`dict` [:class:`~scm.plams.mol.molecule.Molecule`, :class:`T<typing.TypeVar>`]
        A dictionary with all (filtered) molecules as keys and elements from **data** as values.

    See Also
    --------
    :func:`filter_data`
        Filter **mol_list** and **data** based on elements from **data**.

    """
    return {mol: item for mol, item in zip(mol_list, data) if filter(mol)}


def filter_data(mol_list: Iterable[Molecule], data: Iterable[T],
                filter: Callable[[T], bool]) -> Dict[Molecule, T]:
    """Filter **mol_list** and **data** based on elements from **data**.

    Examples
    --------
    See :func:`filter_mol` for a number of input examples.

    Parameters
    ----------
    mol_list : :class:`~collections.abc.Iterable` [:class:`~scm.plams.mol.molecule.Molecule`]
        An iterable of the, to-be filtered, PLAMS molecules.

    data : :class:`Iterable[T]<collections.abc.Iterable>`
        An iterable which will be assigned as values to the to-be returned dict.
        These parameters will be filtered in conjunction with **mol_list**.
        Note that **mol_list** and **data** *should* be of the same length.

    filter : :class:`Callable[[T], bool]<collections.abc.Callable>`
        A callable for filtering the elements of **data**.
        An example would be :code:`lambda n: n < 10`.

    Returns
    -------
    :class:`dict` [:class:`~scm.plams.mol.molecule.Molecule`, :class:`T<typing.TypeVar>`]
        A dictionary with all (filtered) molecules as keys and elements from **data** as values.

    See Also
    --------
    :func:`filter_mol`
        Filter **mol_list** and **data** based on elements from **mol_list**.

    """
    return {mol: item for mol, item in zip(mol_list, data) if filter(item)}
