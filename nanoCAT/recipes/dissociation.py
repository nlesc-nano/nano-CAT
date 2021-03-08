"""
nanoCAT.recipes.surface_dissociation
====================================

A recipe for dissociation specific sets of surface atoms.

Index
-----
.. currentmodule:: nanoCAT.recipes
.. autosummary::
    dissociate_surface
    dissociate_bulk
    row_accumulator

API
---
.. autofunction:: dissociate_surface
.. autofunction:: dissociate_bulk
.. autofunction:: row_accumulator

"""

from __future__ import annotations

import sys
from typing import (
    Iterable,
    Generator,
    Any,
    TypeVar,
    cast,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np
from scm.plams import Molecule, MoleculeError
from CAT.utils import get_nearest_neighbors
from CAT.mol_utils import to_atnum, to_symbol
from CAT.distribution import distribute_idx
from CAT.attachment.distribution_brute import brute_uniform_idx
from nanoCAT.bde.dissociate_xyn import dissociate_ligand
from nanoCAT.bde.identify_surface import identify_surface_ch

if TYPE_CHECKING:
    import numpy.typing as npt
    if sys.version_info >= (3, 8):
        from typing import Literal
    else:
        from typing_extensions import Literal

    _IT1 = TypeVar("_IT1", bound=np.integer[Any])
    _IT2 = TypeVar("_IT2", bound=np.integer[Any])
    _SCT = TypeVar("_SCT", bound=np.generic)
    _NDArray = np.ndarray[Any, np.dtype[_SCT]]
    _ModeKind = Literal['uniform', 'random', 'cluster']

__all__ = ['dissociate_surface', 'row_accumulator', 'dissociate_bulk']


def _parse_idx(idx: npt.ArrayLike, ndim: int, **kwargs: Any) -> _NDArray[np.int64]:
    _idx_ar = np.array(idx, ndmin=ndim, **kwargs)
    if _idx_ar.ndim != ndim:
        raise ValueError
    elif _idx_ar.dtype.kind in 'USO':
        return _idx_ar.astype(np.int64)
    else:
        return _idx_ar.astype(np.int64, casting='same_kind')


def dissociate_surface(mol: Molecule,
                       idx: npt.ArrayLike,
                       symbol: str = 'Cl',
                       lig_count: int = 1,
                       k: int = 4,
                       displacement_factor: float = 0.5,
                       **kwargs: Any) -> Generator[Molecule, None, None]:
    r"""A workflow for dissociating :math:`(XY_{n})_{\le m}` compounds from the surface of **mol**.

    The workflow consists of four distinct steps:

    1. Identify which atoms :math:`Y`, as specified by **symbol**,
       are located on the surface of **mol**.
    2. Identify which surface atoms are neighbors of :math:`X`, the latter being defined by **idx**.
    3. Identify which pairs of :math:`n*m` neighboring surface atoms are furthest removed from
       each other.
       :math:`n` is defined by **lig_count** and :math:`m`, if applicable, by the index along axis 1
       of **idx**.
    4. Yield :math:`(XY_{n})_{\le m}` molecules constructed from **mol**.

    Note
    ----
    The indices supplied in **idx** will, when applicable, be sorted along its last axis.

    Examples
    --------
    .. code:: python

        >>> from pathlib import Path

        >>> import numpy as np

        >>> from scm.plams import Molecule
        >>> from CAT.recipes import dissociate_surface, row_accumulator

        >>> base_path = Path(...)
        >>> mol = Molecule(base_path / 'mol.xyz')

        # The indices of, e.g., Cs-pairs
        >>> idx = np.array([
        ...     [1, 3],
        ...     [4, 5],
        ...     [6, 10],
        ...     [15, 12],
        ...     [99, 105],
        ...     [20, 4]
        ... ])

        # Convert 1- to 0-based indices by substracting 1 from idx
        >>> mol_generator = dissociate_surface(mol, idx-1, symbol='Cl', lig_count=1)

        # Note: The indices in idx are (always) be sorted along axis 1
        >>> iterator = zip(row_accumulator(np.sort(idx, axis=1)), mol_generator)
        >>> for i, mol in iterator:
        ...     mol.write(base_path / f'output{i}.xyz')


    Parameters
    ----------
    mol : :class:`Molecule<scm.plams.mol.molecule.Molecule>`
        The input molecule.

    idx : array-like, dimensions: :math:`\le 2`
        An array of indices denoting to-be dissociated atoms (*i.e.* :math:`X`);
        its elements will, if applicable, be sorted along the last axis.
        If a 2D array is provided then all elements along axis 1 will be dissociated
        in a cumulative manner.
        :math:`m` is herein defined as the index along axis 1.

    symbol : :class:`str` or :class:`int`
        An atomic symbol or number defining the super-set of the atoms to-be dissociated in
        combination with **idx** (*i.e.* :math:`Y`).

    lig_count : :class:`int`
        The number of atoms specified in **symbol** to-be dissociated in combination
        with a single atom from **idx** (*i.e.* :math:`n`).

    k : :class:`int`
        The number of atoms specified in **symbol** which are surrounding a single atom in **idx**.
        Must obey the following condition: :math:`k \ge 1`.

    displacement_factor : :class:`float`
        The smoothing factor :math:`n` for constructing a convex hull;
        should obey :math:`0 <= n <= 1`.
        Represents the degree of displacement of all atoms with respect to a spherical surface;
        :math:`n = 1` is a complete projection while :math:`n = 0` means no displacement at all.

        A non-zero value is generally recomended here,
        as the herein utilized :class:`ConvexHull<scipy.spatial.ConvexHull>` class
        requires an adequate degree of surface-convexness,
        lest it fails to properly identify all valid surface points.

    \**kwargs : :data:`Any<typing.Any>`
        Further keyword arguments for
        :func:`brute_uniform_idx()<CAT.attachment.distribution_brute.brute_uniform_idx>`.

    Yields
    ------
    :class:`Molecule<scm.plams.mol.molecule.Molecule>`
        Yields new :math:`(XY_{n})_{m}`-dissociated molecules.

    See Also
    --------
    :func:`brute_uniform_idx()<CAT.attachment.distribution_brute.brute_uniform_idx>`
        Brute force approach to creating uniform or clustered distributions.

    :func:`identify_surface()<nanoCAT.bde.identify_surface.identify_surface>`
        Take a molecule and identify which atoms are located on the surface,
        rather than in the bulk.

    :func:`identify_surface_ch()<nanoCAT.bde.identify_surface.identify_surface_ch>`
        Identify the surface of a molecule using a convex hull-based approach.

    :func:`dissociate_ligand()<nanoCAT.bde.dissociate_xyn.dissociate_ligand>`
        Remove :math:`XY_{n}` from **mol** with the help of the
        :class:`MolDissociater<nanoCAT.bde.dissociate_xyn.MolDissociater>` class.

    """
    idx_ar = _parse_idx(idx, ndim=2, copy=False)
    if idx_ar.ndim > 2:
        raise ValueError("'idx' expected a 2D array-like object; "
                         f"observed number of dimensions: {idx_ar.ndim}")
    idx_ar.sort(axis=1)
    idx_ar = idx_ar[:, ::-1]

    # Identify all atoms in **idx** located on the surface
    idx_surface_superset = _get_surface(
        mol, symbol=symbol, displacement_factor=displacement_factor
    )

    # Construct an array with the indices of opposing surface-atoms
    n = lig_count * idx_ar.shape[1]
    idx_surface = _get_opposite_neighbor(
        np.asarray(mol), idx_ar, idx_surface_superset, n=n, k=k, **kwargs
    )

    # Dissociate and yield new molecules
    idx_ar += 1
    idx_surface += 1
    for idx_pair, idx_pair_surface in zip(idx_ar, idx_surface):
        mol_tmp = mol.copy()
        _mark_atoms(mol_tmp, idx_pair_surface)

        for i in idx_pair:
            mol_tmp = next(dissociate_ligand(mol_tmp, lig_count=lig_count,
                                             core_index=i, lig_core_pairs=1,
                                             **kwargs))
            yield mol_tmp


def row_accumulator(iterable: Iterable[Iterable[object]]) -> Generator[str, None, None]:
    """Return a generator which accumulates elements along the nested elements of **iterable**.

    Examples
    --------
    .. code:: python

        >>> iterable = [[1, 3],
        ...             [4, 5],
        ...             [6, 10]]

        >>> for i in row_accumulator(iterable):
        ...     print(repr(i))
        '_1'
        '_1_3'
        '_4'
        '_4_5'
        '_6'
        '_6_10'

    Parameters
    ----------
    iterable : :class:`Iterable<collections.abc.Iterable>` [:class:`Iterable<collections.abc.Iterable>` [:data:`Any<typing.Any>`]]
        A nested iterable.

    Yields
    ------
    :class:`str`
        The accumulated nested elements of **iterable** as strings.

    """  # noqa
    for i in iterable:
        ret = ''
        for j in i:
            ret += f'_{j}'
            yield ret


def dissociate_bulk(
    mol: Molecule,
    symbol_x: str,
    symbol_y: None | str = None,
    count_x: int = 1,
    count_y: int = 1,
    n_pairs: int = 1,
    k: None | int = 4,
    r_max: None | float = None,
    mode: _ModeKind = 'uniform',
    **kwargs: Any,
) -> Molecule:
    r"""A workflow for removing :math:`XY`-based compounds from the bulk of **mol**.

    Examples
    --------
    .. code-block:: python

        >>> from scm.plams import Molecule
        >>> from CAT.recipes import dissociate_bulk

        >>> mol: Molecule = ...

        # Remove two PbBr2 pairs in a system where
        # each lead atom is surrounded by 6 bromides
        >>> mol_out1 = dissociate_bulk(
        ...     mol, symbol_x="Pb", symbol_y="Br", count_y=2, n_pairs=2, k=6
        ... )

        # The same as before, expect all potential bromides are
        # identified based on a radius, rather than a fixed number
        >>> mol_out2 = dissociate_bulk(
        ...     mol, symbol_x="Pb", symbol_y="Br", count_y=2, n_pairs=2, r_max=5.0
        ... )

        # Convert a fraction to a number of pairs
        >>> f = 0.5
        >>> count_x = 2
        >>> symbol_x = "Pb"
        >>> n_pairs = int(f * sum(at.symbol == symbol_x for at in mol) / count_x)

        >>> mol_out3 = dissociate_bulk(
        ...     mol, symbol_x="Pb", symbol_y="Br", count_y=2, n_pairs=n_pairs, k=6
        ... )

    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule`
        The input molecule.
    symbol_x : :class:`str` or :class:`int`
        The atomic symbol or number of the central (to-be dissociated) atom(s) :math:`X`.
    symbol_y : :class:`str` or :class:`int`, optional
        The atomic symbol or number of the surrounding (to-be dissociated) atom(s) :math:`Y`.
        If :data:`None`, do not dissociate any atoms :math:`Y`.
    count_x : :class:`int`
        The number of central atoms :math:`X` per individual to-be dissociated cluster.
    count_y : :class:`int`
        The number of surrounding atoms :math:`Y` per individual to-be dissociated cluster.
    n_pairs : :class:`int`
        The number of to-be removed :math:`XY` fragments.
    k : :class:`int`, optional
        The total number of :math:`Y` candidates surrounding each atom :math:`X`.
        This value should be smaller than or equal to **count_y**.
        See the **r_max** parameter for a radius-based approach;
        note that both parameters are not mutually exclusive.
    r_max : :class:`int`, optional
        The radius used for searching for :math:`Y` candidates surrounding each atom :math:`X`.
        See **k** parameter to use a fixed number of nearest neighbors;
        note that both parameters are not mutually exclusive.
    mode : :class:`str`
        How the subset of to-be removed atoms :math:`X` should be generated.
        Accepts one of the following values:

        * ``"random"``: A random distribution.
        * ``"uniform"``: A uniform distribution; the distance between each successive atom and
          all previous points is maximized.
        * ``"cluster"``: A clustered distribution; the distance between each successive atom and
          all previous points is minmized.

    Keyword Arguments
    -----------------
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :func:`CAT.distribution.distribute_idx`.

    Returns
    -------
    :class:`~scm.plams.mol.molecule.Molecule`
        The molecule with :math:`n_{\text{pair}} * XY` fragments removed.

    """
    if n_pairs < 0:
        raise ValueError("`n_pairs` must be larger than or equal to 0; "
                         f"observed value: {n_pairs!r}")
    elif n_pairs == 0:
        return mol.copy()

    # Validate the input args
    if count_x <= 0:
        raise ValueError("`count_x` expected a an integer larger than 0; "
                         f"observed value: {count_x!r}")
    elif count_y < 0:
        raise ValueError("`count_y` expected a an integer larger than 0; "
                         f"observed value: {count_y!r}")
    elif count_y == 0:
        symbol_y = None

    # Parse `symbol_x`
    atnum_x = to_atnum(symbol_x)
    idx_x = np.fromiter([i for i, at in enumerate(mol) if at.atnum == atnum_x], dtype=np.intp)
    if len(idx_x) == 0:
        raise MoleculeError(f"No atoms {to_symbol(atnum_x)!r} in {mol.get_formula()}")

    # Parse `symbol_y`
    if symbol_y is not None:
        atnum_y = to_atnum(symbol_y)
        idx_y = np.fromiter([i for i, at in enumerate(mol) if at.atnum == atnum_y], dtype=np.intp)
        if len(idx_y) == 0:
            raise MoleculeError(f"No atoms {to_symbol(atnum_y)!r} in {mol.get_formula()}")

    # Parse `k` and `r_max`
    if k is None and r_max is None:
        raise TypeError("`k` and `r_max` cannot be both set to None")
    if k is None:
        k = cast(int, len(idx_y))
    if r_max is None:
        r_max = cast(float, np.inf)

    # Validate `n_pairs`
    if (n_pairs * count_x) > len(idx_x):
        x = n_pairs * count_x
        raise ValueError(
            f"Insufficient atoms {to_symbol(atnum_x)!r} in {mol.get_formula()}; "
            f"atleast {x} required with `n_pairs={n_pairs!r}` and `count_x={count_x!r}"
        )
    elif symbol_y is not None and (n_pairs * count_y) > len(idx_y):
        y = n_pairs * count_y
        raise ValueError(
            f"Insufficient atoms {to_symbol(atnum_y)!r} in {mol.get_formula()}; "
            f"atleast {y} required with `n_pairs={n_pairs!r}` and `count_y={count_y!r}"
        )

    # Identify a subset of atoms `x`
    if count_x != 1:
        if "cluster_size" in kwargs:
            raise TypeError("Specifying both `cluster_size` and `count_x` is not supported")
        kwargs["cluster_size"] = count_x
    idx_x_sorted = distribute_idx(mol, idx_x, f=1, **kwargs)
    idx_x_sorted.shape = -1, count_x

    if symbol_y is not None:
        # Identify a subset of matching atoms `y`
        idx_x_subset, idx_y_subset, = _get_opposite_neighbor2(
            np.asarray(mol), idx_x_sorted, idx_y, n_pairs, n=count_y, k=k, r_max=r_max
        )

        # Concatenate the subsets
        idx_xy_subset = np.concatenate([idx_x_subset.ravel(), idx_y_subset.ravel()])
    else:
        idx_xy_subset = idx_x[:(n_pairs * count_x)]
    idx_xy_subset.sort()
    idx_xy_subset += 1

    # Create and return the new molecule
    ret = mol.copy()
    atom_set = {ret[i] for i in idx_xy_subset}
    for at in atom_set:
        ret.delete_atom(at)
    return ret


def _get_opposite_neighbor(
    mol: _NDArray[np.number],
    idx_center: _NDArray[np.integer[Any]],
    idx_neighbor: _NDArray[_IT1],
    k: int = 4,
    n: int = 2,
    **kwargs: Any,
) -> _NDArray[_IT1]:
    """Identify the **k** nearest neighbors of **idx_center** and return those furthest removed from each other."""  # noqa
    # Indices of the **k** nearest neighbors in **neighbor** with respect to **center**
    xyz1 = mol[idx_neighbor]
    xyz2 = mol[idx_center.ravel()]
    try:
        idx_nn = idx_neighbor[get_nearest_neighbors(xyz2, xyz1, k=k)]
    except IndexError:
        raise ValueError("'k' should be smaller than the total number of surface atoms "
                         f"(len(idx_neighbor)); observed value: {k!r}") from None
    idx_nn.shape = -1, idx_nn.shape[1] * idx_center.shape[1]

    # Find the **n** atoms in **idx_nn** furthest removed from each other
    return brute_uniform_idx(mol, idx_nn, n=n, **kwargs)


def _get_opposite_neighbor2(
    mol: _NDArray[np.number],
    idx_center: _NDArray[_IT1],
    idx_neighbor: _NDArray[_IT2],
    n_pairs: int,
    k: int = 4,
    n: int = 2,
    r_max: float = np.inf,
    **kwargs: Any,
) -> Tuple[_NDArray[_IT1], _NDArray[_IT2]]:
    """Identify the **k** nearest neighbors of **idx_center** and return those furthest removed from each other."""  # noqa
    # Indices of the **k** nearest neighbors in **neighbor** with respect to **center**
    xyz1 = mol[idx_neighbor]
    xyz2 = mol[idx_center.ravel()]

    idx = get_nearest_neighbors(xyz2, xyz1, k=k, distance_upper_bound=r_max)
    if len(idx_neighbor) in idx:
        is_overflow = (idx != len(idx_neighbor)).sum(dtype=np.int64, axis=1) < n
        if np.count_nonzero(~is_overflow) < n_pairs:
            raise ValueError("Insufficient number of `XY` pairs with "
                             f"`k={k!r}` and `r_max={r_max!r}`")
        i_ar = np.arange(len(idx), dtype=np.intp)[~is_overflow][:n_pairs]
        i: np.intp = idx[i_ar].argmax(axis=1).min()
        idx = idx[i_ar][:, :i]
        idx_center_subset = idx_center[i_ar]
    else:
        idx = idx[:n_pairs]
        idx_center_subset = idx_center[:n_pairs]
    idx_nn = idx_neighbor[idx]

    # Find the **n** atoms in **idx_nn** furthest removed from each other
    idx_neighbor_subset = brute_uniform_idx(mol, idx_nn, n=n, **kwargs)
    return idx_center_subset, idx_neighbor_subset


def _get_surface(
    mol: Molecule,
    symbol: str,
    displacement_factor: float = 0.5,
) -> _NDArray[np.int64]:
    """Return the indices of all atoms, whose atomic symbol is equal to **atom_symbol**, located on the surface."""  # noqa
    # Identify all atom with atomic symbol **atom_symbol**
    atnum = to_atnum(symbol)
    idx = np.array([i for i, atom in enumerate(mol) if atom.atnum == atnum], dtype=np.int64)
    xyz = np.asarray(mol)

    # Identify all atoms on the surface
    try:
        return idx[identify_surface_ch(xyz[idx], n=displacement_factor)]
    except ValueError as ex:
        raise MoleculeError(f"No atoms with atomic symbol/number {repr(symbol)} available in "
                            f"{mol.get_formula()!r}") from ex


def _mark_atoms(mol: Molecule, idx: Iterable[int]) -> None:
    """Mark all atoms in **mol** whose index is in **idx**; indices should be 1-based."""
    for i in idx:
        mol[i].properties.anchor = True
