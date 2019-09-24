"""
nanoCAT.ff.prm
==============

A class for reading and generating .prm parameter files.

Index
-----
.. currentmodule:: nanoCAT.ff.prm
.. autosummary::
    PRMContainer

API
---
.. autoclass:: PRMContainer
    :members:
    :private-members:
    :special-members:

"""

import inspect
from typing import (Any, Iterator, Dict, Tuple, FrozenSet)
from itertools import chain

import pandas as pd

from CAT.abc.dataclass import AbstractDataClass
from CAT.abc.file_container import AbstractFileContainer

__all__ = ['PRMContainer']


class PRMContainer(AbstractDataClass, AbstractFileContainer):
    """A container for managing prm files.

    Attributes
    ----------
    pd_printoptions : :class:`dict` [:class:`str`, :class:`object`], private
        A dictionary with Pandas print options.
        See `Options and settings <https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html>`_.

    """  # noqa

    #: A :class:`frozenset` with the names of private instance attributes.
    #: These attributes will be excluded whenever calling :meth:`PRMContainer.as_dict`.
    _PRIVATE_ATTR: FrozenSet[str] = frozenset({'_pd_printoptions'})

    #: A tuple of supported .psf headers.
    HEADERS: Tuple[str] = (
        'ATOMS', 'BONDS', 'ANGLES', 'DIHEDRALS', 'NBFIX', 'HBOND', 'NONBONDED', 'IMPROPERS', 'END'
    )

    def __init__(self, filename=None, atoms=None, bonds=None, angles=None, dihedrals=None,
                 impropers=None, nonbonded=None, nonbonded_header=None, nbfix=None,
                 hbond=None) -> None:
        """Initialize a :class:`PRMContainer` instance."""
        self.filename: str = filename
        self.atoms: pd.DataFrame = atoms
        self.bonds: pd.DataFrame = bonds
        self.angles: pd.DataFrame = angles
        self.dihedrals: pd.DataFrame = dihedrals
        self.impropers: pd.DataFrame = impropers
        self.nonbonded_header: str = nonbonded_header
        self.nonbonded: pd.DataFrame = nonbonded
        self.nbfix: pd.DataFrame = nbfix
        self.hbond: str = hbond

        # Print options for Pandas DataFrames
        self.pd_printoptions: Dict[str, Any] = {'display.max_rows': 20}

    @property
    def pd_printoptions(self) -> Iterator: return chain.from_iterable(self._pd_printoptions.items())

    @pd_printoptions.setter
    def pd_printoptions(self, value: dict) -> None: self._pd_printoptions = self._is_dict(value)

    @staticmethod
    def _is_dict(value: Any) -> dict:
        """Check if **value** is a :class:`dict` instance; raise a :exc:`TypeError` if not."""
        if not isinstance(value, dict):
            caller_name: str = inspect.stack()[1][3]
            raise TypeError(f"The {repr(caller_name)} parameter expects an instance of 'dict'; "
                            f"observed type: {repr(type(value))}")
        return value

    @AbstractDataClass.inherit_annotations()
    def __str__(self):
        with pd.option_context(*self.pd_printoptions):
            return super().__str__()

    __repr__ = __str__

    # Ensure that a deepcopy is returned unless explictly specified

    @AbstractDataClass.inherit_annotations()
    def copy(self, deep=True, copy_private=False):
        kwargs = self.as_dict(copy=deep, return_private=copy_private)
        return self.from_dict(kwargs)

    @AbstractDataClass.inherit_annotations()
    def __copy__(self): return self.copy()

    """########################### methods for reading .prm files. ##############################"""

    @classmethod
    @AbstractFileContainer.inherit_annotations()
    def _read_iterate(cls, iterator):
        ret = {}
        value = None

        for i in iterator:
            j = i.rstrip('\n')
            if j.startswith('!') or j.startswith('*') or j.isspace() or not j:
                continue  # Ignore comment lines and empty lines

            key = j.split(maxsplit=1)[0]
            if key in cls.HEADERS:
                ret[key.lower()] = value = []
                ret[key.lower() + '_comment'] = value_comment = []
                if key in ('HBOND', 'NONBONDED'):
                    value.append(i.split()[1:])
                continue

            v, _, comment = j.partition('!')
            value.append(v.split())
            value_comment.append(comment.strip())

        cls._read_post_iterate(ret)
        return ret

    @staticmethod
    def _read_post_iterate(kwargs: dict) -> None:
        """Post process the dictionary produced by :meth:`PRMContainer._read_iterate`."""
        if 'end' in kwargs:
            del kwargs['end']
        if 'end_comment' in kwargs:
            del kwargs['end_comment']

        comment_dict = {}
        for k, v in kwargs.items():
            if k.endswith('_comment'):
                comment_dict[k] = v
            elif k == 'hbond':
                kwargs[k] = ' '.join(chain.from_iterable(v)).split('!')[0].rstrip()
            elif k == 'nonbonded':
                nonbonded_header = ' '.join(chain.from_iterable(v[0:2])).rstrip()
                kwargs[k] = pd.DataFrame(v[2:])
            else:
                kwargs[k] = pd.DataFrame(v)

        try:
            kwargs['nonbonded_header'] = nonbonded_header
        except NameError:
            pass

        for k, v in comment_dict.items():
            if k == 'nonbonded_comment':
                v = v[1:]
            del kwargs[k]
            if k == 'hbond_comment':
                continue
            kwargs[k.rstrip('_comment')]['comment'] = v

    @AbstractFileContainer.inherit_annotations()
    def _read_postprocess(self, filename, encoding=None, **kwargs):
        if isinstance(filename, str):
            self.filename = filename

    """########################### methods for writing .prm files. ##############################"""

    @AbstractFileContainer.inherit_annotations()
    def write(self, filename=None, encoding=None, **kwargs):
        _filename = filename if filename is not None else self.filename
        if not _filename:
            raise TypeError("The 'filename' parameter is missing")
        super().write(_filename, encoding, **kwargs)

    @AbstractFileContainer.inherit_annotations()
    def _write_iterate(self, write, **kwargs) -> None:
        for key in self.HEADERS[:-1]:
            key_low = key.lower()
            df = getattr(self, key_low)
            if key_low == 'hbond':
                write(f'\n{key} {df}\n')
                continue
            elif not isinstance(df, pd.DataFrame):
                continue

            iterator = range(df.shape[1] - 1)
            df_str = ' '.join('{:8}' for _ in iterator) + ' ! {}\n'

            if key_low != 'nonbonded':
                write(f'\n{key}\n')
            else:
                header = '-\n'.join(i for i in self.nonbonded_header.split('-'))
                write(f'\n{key} {header}\n')
            for _, value in df.iterrows():
                write(df_str.format(*(('' if i is None else i) for i in value)))

        write('\nEND\n')
