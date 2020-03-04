import textwrap
from os import PathLike
from io import TextIOBase
from types import MappingProxyType
from typing import (ClassVar, FrozenSet, Mapping, Optional, Any, Callable, Union, AnyStr, TypeVar,
                    Iterator, KeysView, ItemsView, ValuesView, Dict, SupportsFloat)
from collections import abc
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from scm.plams import Settings
from assertionlib.dataclass import AbstractDataClass
from FOX import (MultiMolecule, PSFContainer, PRMContainer,
                 get_non_bonded, get_intra_non_bonded, get_bonded)

__all__ = ['EnergyGatherer']

# TypeVars for keys and values
KT = TypeVar('KT', bound=str)
KV = TypeVar('KV', None, pd.DataFrame)


def _return_true(value: Any) -> bool: return True


class EnergyGatherer(AbstractDataClass, abc.Mapping):
    r"""A :class:`Mapping<collections.abc.Mapping>` for calculating, collecting and storing forcefield energy terms.

    The class has three methods for calculating forcefield energy terms:

    * :meth:`EnergyGatherer.inter_nonbonded`:
      Collect, assign and return all inter-ligand non-bonded interactions.
    * :meth:`EnergyGatherer.intra_nonbonded`:
      Collect, assign and return all intra-ligand non-bonded interactions.
    * :meth:`EnergyGatherer.intra_bonded`:
      Collect, assign and return all intra-ligand bonded interactions.

    The resulting DataFrames can be exported to and from .csv files or concatenated with,
    respectivelly, the :meth:`EnergyGatherer.write_csv` and :meth:`EnergyGatherer.read_csv` methods.

    All units are in atomic units.

    Parameters
    ----------
    inter_elstat : :class:`pandas.DataFrame`, optional
        A DataFrame with inter-ligand electrostatic interactions.
        See :attr:`EnergyGatherer.inter_elstat`.

    inter_lj : :class:`pandas.DataFrame`, optional
        A DataFrame with inter-ligand Lennard-Jones interactions.
        See :attr:`EnergyGatherer.inter_lj`.

    intra_elstat : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand electrostatic interactions.
        See :attr:`EnergyGatherer.intra_elstat`.

    intra_lj : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand Lennard-Jones interactions.
        See :attr:`EnergyGatherer.intra_lj`.

    bonds : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand bonded interactions.
        See :attr:`EnergyGatherer.bonds`.

    angles : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand angle interactions.
        See :attr:`EnergyGatherer.angles`.

    urey_bradley : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand Urey-Bradley interactions.
        See :attr:`EnergyGatherer.urey_bradley`.

    dihedrals : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand proper dihedral interactions.
        See :attr:`EnergyGatherer.dihedrals`.

    impropers : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand improper dihedral interactions.
        See :attr:`EnergyGatherer.impropers`.

    Attributes
    ----------
    inter_elstat : :class:`pandas.DataFrame`, optional
        A DataFrame with inter-ligand electrostatic interactions:
        :math:`V_{elstat} = \frac{1}{4 \pi \varepsilon_{0}} \frac{q_{i} q_{j}}{r_{ij}}`.

    inter_lj : :class:`pandas.DataFrame`, optional
        A DataFrame with inter-ligand Lennard-Jones interactions:
        :math:`V_{LJ} = 4 \varepsilon \left( \left( \frac{\sigma}{r} \right )^{12} - \left(\frac{\sigma}{r}\right )^6 \right )`.

    intra_elstat : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand electrostatic interactions:
        :math:`V_{elstat} = \frac{1}{4 \pi \varepsilon_{0}} \frac{q_{i} q_{j}}{r_{ij}}`.

    intra_lj : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand Lennard-Jones interactions:
        :math:`V_{LJ} = 4 \varepsilon \left( \left( \frac{\sigma}{r} \right )^{12} - \left(\frac{\sigma}{r}\right )^6 \right )`.

    bonds : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand bonded interactions:
        :math:`V_{bonds} = k_{r} (r - r_{0})^2`.

    angles : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand angle interactions:
        :math:`V_{angles} = k_{\theta} (\theta - \theta_{0})^2`.

    urey_bradley : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand Urey-Bradley interactions:
        :math:`V_{Urey-Bradley} = k_{\hat{r}} (\hat{r} - \hat{r}_{0})^2`.

    dihedrals : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand proper dihedral interactions:
        :math:`V_{dihedrals} = k_{\phi} [1 + \cos(n \phi - \delta)]`.

    impropers : :class:`pandas.DataFrame`, optional
        A DataFrame with intra-ligand improper dihedral interactions:
        :math:`V_{impropers} = k_{\omega} (\omega - \omega_{0})^2`.

    vars_view : :class:`MappingProxyType<types.MappingProxyType>`
        A read-only view of this instances' instance variables.

    """  # noqa

    #: A :class:`frozenset` of potenial energy terms.
    E_SET: ClassVar[FrozenSet[str]] = frozenset({
        'inter_elstat', 'inter_lj', 'intra_elstat', 'intra_lj',
        'bonds', 'angles', 'urey_bradley', 'dihedrals', 'impropers'
    })

    #: A :class:`frozenset` with the names of private instance variables.
    #: These attributes will be excluded whenever calling :meth:`EnergyGatherer.as_dict`,
    #: printing or comparing objects.
    _PRIVATE_ATTR: ClassVar[FrozenSet[str]] = frozenset({'_vars_view'})

    @property
    def vars_view(self) -> Mapping[KT, KV]:
        """Return a read-only view of all attributes specified in :attr:`EnergyGatherer.E_SET`."""
        return MappingProxyType(self._vars_view)

    def __init__(self, **kwargs: Optional[pd.DataFrame]) -> None:
        """Initialize the :class:`EnergyGatherer` instance."""
        super().__init__()
        self._vars_view: Dict[str, Optional[pd.DataFrame]] = {}

        self.inter_elstat = None
        self.inter_lj = None
        self.intra_elstat = None
        self.intra_lj = None

        self.bonds = None
        self.angles = None
        self.urey_bradley = None
        self.dihedrals = None
        self.impropers = None

        self._validate_keys(kwargs)
        for k, df in kwargs.items():
            setattr(self, k, df)

    @classmethod
    def _validate_keys(cls, kwargs: Mapping[KT, KV]) -> None:
        """Validate the variable keyword arguments of :meth:`EnergyGatherer.__init__`."""
        difference = set(kwargs.keys()).difference(cls.E_SET)
        if len(difference) == 1:
            raise AttributeError(f"Invalid attribute name {repr(difference.pop())}; "
                                 f"accepted names: {tuple(sorted(cls.E_SET))}")
        elif len(difference) > 1:
            raise AttributeError(f"Multiple invalid attribute names {tuple(sorted(difference))}; "
                                 f"accepted names: {tuple(sorted(cls.E_SET))}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Implement :code:`setattr(self, name, value)`."""
        if name in self.E_SET:
            self._vars_view[name] = value
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Implement :code:`delattr(self, name)`."""
        if name in self.E_SET:
            del self._vars_view[name]
        super().__delattr__(name)

    @property
    def __getitem__(self) -> Callable[[KT], KV]:
        """Get the :meth:`__getitem__<types.MappingProxyType.__getitem__>` method of :attr:`EnergyGatherer.vars_view`."""  # noqa
        return self.vars_view.__getitem__

    @property
    def __iter__(self) -> Callable[[], Iterator[KT]]:
        """Get the :meth:`__iter__<types.MappingProxyType.__iter__>` method of :attr:`EnergyGatherer.vars_view`."""  # noqa
        return self.vars_view.__iter__

    @property
    def __len__(self) -> Callable[[], int]:
        """Get the :meth:`__len__<types.MappingProxyType.__len__>` method of :attr:`EnergyGatherer.vars_view`."""  # noqa
        return self.vars_view.__len__

    @property
    def __contains__(self) -> Callable[[KT], bool]:
        """Get the :meth:`__contains__<types.MappingProxyType.__contains__>` method of :attr:`EnergyGatherer.vars_view`."""  # noqa
        return self.vars_view.__contains__

    @property
    def get(self) -> Callable[[KT, Optional[Any]], KV]:
        """Get the :meth:`get<types.MappingProxyType.get>` method of :attr:`EnergyGatherer.vars_view`."""  # noqa
        return self.vars_view.get

    @property
    def keys(self) -> Callable[[], KeysView[KT]]:
        """Get the :meth:`keys<types.MappingProxyType.keys>` method of :attr:`EnergyGatherer.vars_view`."""  # noqa
        return self.vars_view.keys

    @property
    def items(self) -> Callable[[], ItemsView[KT, KV]]:
        """Get the :meth:`items<types.MappingProxyType.items>` method of :attr:`EnergyGatherer.vars_view`."""  # noqa
        return self.vars_view.items

    @property
    def values(self) -> Callable[[], ValuesView[KV]]:
        """Get the :meth:`values<types.MappingProxyType.values>` method of :attr:`EnergyGatherer.vars_view`."""  # noqa
        return self.vars_view.values

    @staticmethod
    @AbstractDataClass.inherit_annotations()
    def _str(key: str, value: Any,
             width: Optional[int] = None,
             indent: Optional[int] = None) -> str:
        """Return a string representation of a single **key**/**value** pair."""
        key_str = f'{key} = ' if width is None else f'{key:{width}} = '
        val = repr(value) if not isinstance(value, pd.DataFrame) else f'{value.__class__.__name__}(..., shape={value.shape})'  # noqa
        value_str = textwrap.indent(val, ' ' * indent)[indent:]
        return f'{key_str}{value_str}'  # e.g.: "key   =     'value'"

    @staticmethod
    @AbstractDataClass.inherit_annotations()
    def _eq(v1, v2): np.testing.assert_array_equal(v1, v2)

    @AbstractDataClass.inherit_annotations()
    def __copy__(self): return self.copy(deep=True)

    def inter_nonbonded(self, multi_mol: MultiMolecule, s: Settings, psf: PSFContainer,
                        prm: PRMContainer, **kwargs) -> float:
        """Collect, assign and return all inter-ligand non-bonded interactions."""
        atom_set = set(psf.atom_type[psf.residue_name != 'COR'])
        atom_pairs = combinations_with_replacement(sorted(atom_set), r=2)

        # Manually calculate all inter-ligand, ligand/core & core/core interactions
        elstat_df, lj_df = get_non_bonded(multi_mol, psf=psf, prm=prm, cp2k_settings=s,
                                          atom_pairs=atom_pairs, **kwargs)
        self.inter_elstat = elstat_df
        self.inter_lj = lj_df
        return elstat_df.mean().sum() + lj_df.mean().sum()

    def intra_nonbonded(self, multi_mol: MultiMolecule, psf: PSFContainer,
                        prm: PRMContainer, **kwargs: Any) -> float:
        """Collect, assign and return all intra-ligand non-bonded interactions."""
        elstat_df, lj_df = get_intra_non_bonded(multi_mol, psf, prm, **kwargs)
        self.intra_elstat = elstat_df
        self.intra_lj = lj_df
        return elstat_df.mean().sum() + lj_df.mean().sum()

    def intra_bonded(self, multi_mol: MultiMolecule, psf: PSFContainer,
                     prm: PRMContainer) -> float:
        """Collect, assign and return all intra-ligand bonded interactions."""
        E_tup = get_bonded(multi_mol, psf, prm)
        names = ('bonds', 'angles', 'urey_bradley', 'dihedrals', 'impropers')
        for name, df in zip(names, E_tup):
            setattr(self, name, df)

        return sum((df.mean().sum() if df is not None else 0.0) for df in E_tup)

    def write_csv(self, path_or_buf: Union[None, AnyStr, PathLike, TextIOBase] = None,
                  drop_zero: bool = True, **kwargs: Any) -> Optional[str]:
        r"""Export this instances :meth:`values<EnergyGatherer.values>` to a .csv file.

        Serves as wrapper arround the :meth:`pandas.DataFrame.to_csv` method.

        Parameters
        ----------
        path_or_buf : :class:`str` or file handle
            File path or object, if ``None`` is provided the result is returned as a string.
            If a file object is passed it should be opened with ``newline=''``,
            disabling universal newlines.

        drop_zero : :class:`bool`
            Drop all columns whose values consist exclusively of ``0.0``.

        \**kwargs : :data:`Any<typing.Any>`
            Further keyword arguments to-be passed to :meth:`pandas.DataFrame.to_csv`.

        Returns
        -------
        :class:`str`, optional
            If ``path_or_buf = None``, return the .csv file's context as :class:`str`.
            Return ``None`` otherwise.

        Raises
        ------
        :exc:`TypeError`
            Raised if all :meth:`values<EnergyGatherer.values>` in this instance are ``None``.

        See Also
        --------
        :meth:`EnergyGatherer.read_csv`
            Update this instance's :meth:`values<EnergyGatherer.values>` with data from a .csv file.

        """
        df = self.concatenate(drop_zero=drop_zero)
        if df is not None:
            return df.to_csv(path_or_buf=path_or_buf, **kwargs)
        else:
            raise TypeError("No DataFrames available in this instance; all values are None")

    @classmethod
    def read_csv(cls, filepath_or_buffer: Union[AnyStr, PathLike, TextIOBase],
                 **kwargs: Any) -> Optional['EnergyGatherer']:
        r"""Update this instance's :meth:`values<EnergyGatherer.values>` with data from a .csv file.

        When calling this method on a class (:code:`EnergyGatherer.read_csv(...)`) create
        and return a new :class:`EnergyGatherer` instance instead.

        Parameters
        ----------
        filepath_or_buffer : :class:`str` or file handle
            Any valid string path is acceptable.
            The string could be a URL.
            Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is expected.
            A local file could be: file://localhost/path/to/table.csv.

        \**kwargs : :data:`Any<typing.Any>`
            Further keyword arguments to-be passed to :func:`pandas.read_csv`.

        Returns
        -------
        :class:`EnergyGatherer`, optional
            Return a new EnergyGatherer instance when called as a classmethod
            (:code:`EnergyGatherer.read_csv(...)`).
            If this method is called on an existing EnergyGatherer instance,
            then perform an inplace update and return ``None``.

        See Also
        --------
        :meth:`EnergyGatherer.write_csv`
            Export this instances :meth:`values<EnergyGatherer.values>` to a .csv file.

        """
        if 'header' not in kwargs:
            kwargs['header'] = [0, 1]
        if 'index_col' not in kwargs:
            kwargs['index_col'] = 0

        df = pd.read_csv(filepath_or_buffer, **kwargs)
        df_dict = cls.separate(df)

        if isinstance(cls, type):
            self = cls()
            for key, value in df_dict.items():
                setattr(self, key, value)
            return self

        else:
            self = cls
            self._validate_keys(df_dict)
            for key, value in df_dict.items():
                setattr(self, key, value)
            return None

    @staticmethod
    def separate(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Separate a :meth:`concatenated<EnergyGatherer.concatenate>` DataFrame into a dictionary of DataFrames.

        The keys on the first level of its columns will be used as the new dictionary keys.
        Keys on the second level will be split and,
        after converting them into a :class:`MultiIndex<pandas.MultiIndex>`,
        used as new DataFrame columns.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A DataFrame with a 2-level :class:`MultiIndex<pandas.MultiIndex>` as its columns.

        Returns
        -------
        :class:`dict` [:class:`str`, :class:`pandas.DataFrame`]
            A dictionary of DataFrames.

        """  # noqa
        ret: Dict[str, pd.DataFrame] = {}

        try:
            assert len(df.columns.levels) == 2
        except AttributeError as ex:
            raise TypeError("'df' expected a 2-level MultiIndex as columns; observed type: "
                            f"{df.columns.__class__.__name__!r}") from ex
        except AssertionError as ex:
            raise ValueError("'df' expected a 2-level MultiIndex as columns; observed number of "
                             f"levels: {len(df.columns.levels)}") from ex

        for key in df.columns.levels[0]:
            sub_df = ret[key] = df[key].copy()
            sub_df.columns = pd.MultiIndex.from_tuples(i.split() for i in sub_df.columns)
        return ret

    def concatenate(self, drop_zero: bool = True) -> Optional[pd.DataFrame]:
        """Concatenate all DataFrames stored in this instances' :meth:`values<EnergyGatherer.values>` into a single DataFrame.

        Parameters
        ----------
        drop_zero : :class:`bool`
            Drop all columns whose values consist exclusively of ``0.0``.

        Returns
        -------
        :class:`pandas.DataFrame`, optional
            A new DataFrame constructed from concatenating all DataFrames in this instance's
            :meth:`items<EnergyGatherer.items>`.
            ``None`` is returned if all :meth:`values<EnergyGatherer.values>`
            in this instance are ``None``.

        """  # noqa
        values_iter = iter(self.values())
        try:
            df = None
            while df is None:
                df = next(values_iter)
        except StopIteration:
            return None

        index = df.index.copy()
        index.name = 'MD Iteration'
        columns = pd.MultiIndex(levels=([], []), codes=([], []), names=('Energy (au)', 'atoms'))
        ret = pd.DataFrame(index=index, columns=columns)

        evaluate: Callable[[pd.Series], bool] = pd.Series.any if drop_zero else _return_true

        for key, df in self.items():
            if df is None:
                continue

            for column, value in df.items():
                if evaluate(value):
                    column_new = (key, ' '.join(str(i) for i in column))
                    ret[column_new] = value.copy()
        return ret

    def drop_zero(self) -> None:
        """Remove all DataFrame columns whose values consist exclusively of ``0.0``."""
        for df in self.values():
            if df is not None:
                is_zero = ~df.any(axis=0)
                labels = is_zero[is_zero].index
                df.drop(labels, inplace=True, axis=1)

    def _imap(self, value: SupportsFloat,
              func: Callable[[pd.DataFrame, float], Any]) -> None:
        """Apply **func** to all DataFrames in this instance."""
        try:
            value_ = float(value)
        except TypeError as ex:
            raise TypeError(f"{self.__class__.__name__!r} instances only support arithmetic "
                            "operations with scalars; observed type: "
                            f"{value.__class__.__name__!r}") from ex

        for df in self.values():
            if df is not None:
                func(df, value_)

    def __iadd__(self, value: SupportsFloat) -> 'EnergyGatherer':
        self._imap(value, func=pd.DataFrame.__iadd__)
        return self

    def __isub__(self, value: SupportsFloat) -> 'EnergyGatherer':
        self._imap(value, func=pd.DataFrame.__isub__)
        return self

    def __imul__(self, value: SupportsFloat) -> 'EnergyGatherer':
        self._imap(value, func=pd.DataFrame.__imul__)
        return self

    def __ifloordiv__(self, value: SupportsFloat) -> 'EnergyGatherer':
        self._imap(value, func=pd.DataFrame.__ifloordiv__)
        return self

    def __itruediv__(self, value: SupportsFloat) -> 'EnergyGatherer':
        self._imap(value, func=pd.DataFrame.__itruediv__)
        return self

    def __ipow__(self, value: SupportsFloat) -> 'EnergyGatherer':
        self._imap(value, func=pd.DataFrame.__ipow__)
        return self

    def __imod__(self, value: SupportsFloat) -> 'EnergyGatherer':
        self._imap(value, func=pd.DataFrame.__imod__)
        return self

    def __add__(self, value: SupportsFloat) -> 'EnergyGatherer':
        ret = self.copy()
        ret += value
        return ret

    def __sub__(self, value: SupportsFloat) -> 'EnergyGatherer':
        ret = self.copy()
        ret -= value
        return ret

    def __mul__(self, value: SupportsFloat) -> 'EnergyGatherer':
        ret = self.copy()
        ret *= value
        return ret

    def __floordiv__(self, value: SupportsFloat) -> 'EnergyGatherer':
        ret = self.copy()
        ret /= value
        return ret

    def __truediv__(self, value: SupportsFloat) -> 'EnergyGatherer':
        ret = self.copy()
        ret //= value
        return ret

    def __pow__(self, value: SupportsFloat) -> 'EnergyGatherer':
        ret = self.copy()
        ret **= value
        return ret

    def __mod__(self, value: SupportsFloat) -> 'EnergyGatherer':
        ret = self.copy()
        ret %= value
        return ret
