"""
nanoCAT.recipes.mark_surface
============================

A recipe for identifying surface-atom subsets.

Index
-----
.. currentmodule:: nanoCAT.recipes
.. autosummary::
    replace_surface

API
---
.. autofunction:: replace_surface

"""

from typing import Any, Union, Iterable
from collections import abc

import numpy as np
import pandas as pd

from scm.plams import Molecule, MoleculeError

from CAT.mol_utils import to_atnum, to_symbol
from CAT.attachment.distribution import distribute_idx
from nanoCAT.bde.identify_surface import identify_surface_ch

__all__ = ['replace_surface']


def replace_surface(mol: Molecule,
                    symbol: Union[str, int],
                    symbol_new: Union[str, int] = 'Cl',
                    nth_shell: Union[int, Iterable[int]] = 0,
                    f: float = 0.5,
                    mode: str = 'uniform',
                    displacement_factor: float = 0.5,
                    **kwargs: Any) -> Molecule:
    r"""A workflow for identifying all surface atoms in **mol** and replacing a subset of them.

    Consists of three distinct steps:

    1. Identifying which atoms, with a user-specified atomic **symbol**,
       are located on the surface of **mol** rather than in the bulk.
    2. Define a subset of the newly identified surface atoms using one of CAT's
       distribution algorithms.
    3. Create and return a molecule where the atom subset defined in step 2
       has its atomic symbols replaced with **symbol_new**.

    Examples
    --------
    Replace 75% of all surface ``"Cl"`` atoms with ``"I"``.

    .. code:: python

        >>> from scm.plams import Molecule
        >>> from CAT.recipes import replace_surface

        >>> mol = Molecule(...)  # Read an .xyz file
        >>> mol_new = replace_surface(mol, symbol='Cl', symbol_new='I', f=0.75)
        >>> mol_new.write(...)  # Write an .xyz file

    The same as above, except this time the new ``"I"`` atoms are all deleted.

    .. code:: python

        >>> from scm.plams import Molecule
        >>> from CAT.recipes import replace_surface

        >>> mol = Molecule(...)  # Read an .xyz file
        >>> mol_new = replace_surface(mol, symbol='Cl', symbol_new='I', f=0.75)

        >>> del_atom = [at for at in mol_new if at.symbol == 'I']
        >>> for at in del_atom:
        ...     mol_new.delete_atom(at)
        >>> mol_new.write(...)  # Write an .xyz file

    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule`
        The input molecule.

    symbol : :class:`str` or :class:`int`
        An atomic symbol or number defining the super-set of the surface atoms.

    symbol_new : :class:`str` or :class:`int`
        An atomic symbol or number which will be assigned to the new surface-atom subset.

    nth_shell : :class:`int` or :class:`~collections.abc.Iterable` [:class:`int`]
        One or more integers denoting along which shell-surface(s) to search.
        For example, if ``symbol = "Cd"`` then ``nth_shell = 0`` represents the surface,
        ``nth_shell = 1`` is the first sub-surface ``"Cd"`` shell and
        ``nth_shell = 2`` is the second sub-surface ``"Cd"`` shell.
        Using ``nth_shell = [1, 2]`` will search along both the first and second
        ``"Cd"`` sub-surface shells.
        Note that a :exc:`Zscm.plams.core.errors.MoleculeError` will be raised if
        the specified **nth_shell** is larger than the actual number of available
        sub-surface shells.

    f : :class:`float`
        The fraction of surface atoms whose atom types will be replaced with **symbol_new**.
        Must obey the following condition: :math:`0 < f \le 1`.

    mode : :class:`str`
        How the subset of surface atoms will be generated.
        Accepts one of the following values:

        * ``"random"``: A random distribution.
        * ``"uniform"``: A uniform distribution; maximizes the nearest-neighbor distance.
        * ``"cluster"``: A clustered distribution; minimizes the nearest-neighbor distance.

    displacement_factor : :class:`float`
        The smoothing factor :math:`n` for constructing a convex hull;
        should obey :math:`0 <= n <= 1`.
        Represents the degree of displacement of all atoms with respect to a spherical surface;
        :math:`n = 1` is a complete projection while :math:`n = 0` means no displacement at all.

        A non-zero value is generally recomended here,
        as the herein utilized :class:`~scipy.spatial.ConvexHull` class
        requires an adequate degree of surface-convexness,
        lest it fails to properly identify all valid surface points.

    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for
        :func:`~CAT.attachment.distribution.distribute_idx`.

    Returns
    -------
    :class:`~scm.plams.mol.molecule.Molecule`
        A new Molecule with a subset of its surface atoms replaced with **symbol_new**.

    See Also
    --------
    :func:`~CAT.attachment.distribution.distribute_idx`
        Create a new distribution of atomic indices from **idx** of length :code:`f * len(idx)`.

    :func:`~nanoCAT.bde.identify_surface.identify_surface`
        Take a molecule and identify which atoms are located on the surface,
        rather than in the bulk.

    :func:`~nanoCAT.bde.identify_surface.identify_surface_ch`
        Identify the surface of a molecule using a convex hull-based approach.

    """
    # Parse input arguments
    xyz = np.array(mol, dtype=float, ndmin=2, copy=False)
    atnum = to_atnum(symbol)
    atnum_new = to_atnum(symbol_new)

    # Define the surface-atom subset
    idx = np.fromiter((i for i, at in enumerate(mol) if at.atnum == atnum), dtype=int)
    try:
        idx_surface = idx[_collect_surface(xyz[idx], displacement_factor, nth_shell)]
    except MoleculeError as exc:
        nth = exc.args[0]
        if nth == 0:
            exc = MoleculeError(f"No atoms with atomic symbol {to_symbol(symbol)!r} available in "
                                f"{mol.get_formula()!r}")
        else:
            exc = MoleculeError(f"No subsurface shell >= {nth!r} of atom {to_symbol(symbol)!r} "
                                f"available in {mol.get_formula()!r}")
        exc.__cause__ = None
        raise exc

    try:
        idx_surface_subset = distribute_idx(xyz, idx_surface, f=f, mode=mode, **kwargs)
    except ValueError as ex:
        raise MoleculeError("Failed to identify any surface atoms with atomic symbol "
                            f"{to_symbol(symbol)!r} in {mol.get_formula()!r}") from ex
    else:
        idx_surface_subset += 1

    # Update atomic symbols and return
    ret = mol.copy()
    for i in idx_surface_subset:
        ret[i].atnum = atnum_new
    return ret


def _collect_surface(xyz: np.ndarray, displacement_factor: float,
                     nth_shell: Union[int, Iterable[int]] = 0) -> np.ndarray:
    """Collect all user-specified surface and/or sub-surface shells of **xyz**."""
    n = -1
    xyz_df = pd.DataFrame(xyz)
    n_set = {nth_shell} if not isinstance(nth_shell, abc.Iterable) else set(nth_shell)

    ret = []
    while True:
        n += 1

        # Check if the DataFrame is empty
        if not xyz_df.size:
            raise MoleculeError(n)

        idx = identify_surface_ch(xyz_df, n=displacement_factor)
        bool_ar = np.ones(len(xyz_df), dtype=bool)
        bool_ar[idx] = False

        if n not in n_set:
            xyz_df = xyz_df.loc[bool_ar, :]
            continue
        else:
            index = xyz_df.index[~bool_ar]
            xyz_df = xyz_df.loc[bool_ar, :]

        n_set.remove(n)
        ret += index.tolist()
        if not n_set:
            return np.fromiter(ret, count=len(ret), dtype=int)
