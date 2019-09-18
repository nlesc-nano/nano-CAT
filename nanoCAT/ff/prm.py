"""
nanoCAT.ff.prm
==============

A class for reading and generating .prm parameter files.

Index
-----
.. currentmodule:: nanoCAT.ff.prm
.. autosummary::
    PSF

API
---
.. autoclass:: PRM
    :members:
    :private-members:
    :special-members:

"""
from copy import deepcopy
import textwrap
from codecs import iterdecode
from typing import (Dict, Optional, Any, Iterator, Union)
from itertools import chain
from collections.abc import Container

import numpy as np
import pandas as pd

__all__ = ['PRM']


class PRM(Container):

    HEADERS = (
        'ATOMS', 'BONDS', 'ANGLES', 'DIHEDRALS', 'NBFIX', 'HBOND', 'NONBONDED', 'IMPROPERS', 'END'
    )

    def __init__(self, filename=None, atoms=None, bonds=None, angles=None, dihedrals=None,
                 impropers=None, nonbonded=None, nonbonded_header=None, nbfix=None,
                 hbond=None) -> None:
        """Initialize a :class:`PRM` instance."""
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

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        def _str(k: str, v: Any) -> str:
            return f'{k:{width}} = ' + textwrap.indent(repr(v), indent2)[len(indent2):]

        width = max(len(k) for k in vars(self))
        indent1 = ' ' * 4
        indent2 = ' ' * (3 + width)
        with pd.option_context('display.max_rows', 20):
            ret = ',\n'.join(_str(k, v) for k, v in vars(self).items())

        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent1)}\n)'

    def __eq__(self, value: Any) -> bool:
        """Check if this instance is equivalent to **value**."""
        try:
            return type(self) is type(value) and vars(self) == vars(value)
        except TypeError:
            return False

    def copy(self, deep: bool = True) -> 'PRM':
        """Return a deep or shallow copy of this instance.

        Parameters
        ----------
        deep : bool
            If ``True``, return a deep copy instead of a shallow one.

        Returns
        -------
        :meth:`PRM.from_dict`:
            A :class:`PRM` instance copied from this instance.

        """
        kwargs = self.as_dict()
        if deep:
            return self.from_dict(deepcopy(kwargs))
        return self.from_dict(kwargs)

    def __copy__(self) -> 'PRM':
        """Return a deep copy of this instance."""
        return self.copy(deep=False)

    def __deepcopy__(self, memo: Any = None) -> 'PRM':
        """Return a deep copy of this instance."""
        return self.copy(deep=True)

    def __contains__(self, value: Any) -> bool:
        """Check if this instance contains **value**."""
        return value in vars(self).values()

    def as_dict(self) -> Dict[str, Any]:
        """Construct a :class:`dict` from this instance.

        Returns
        -------
        |dict|_
            A dictionary with keyword arguments for :meth:`PRM.__init__`.

        See also
        --------
        :meth:`PRM.from_dict`:
            Construct a :class:`PRM` instance from a dictionary.

        """
        return vars(self)

    @classmethod
    def from_dict(cls, dct: Dict[str, np.ndarray]) -> 'PRM':
        """Construct a :class:`PRM` instance from **dct**.

        Returns
        -------
        |nanoCAT.PRM|_
            A :class:`PRM` instance constructed from **psf_dict**.

        See also
        --------
        :meth:`PRM.as_dict`:
            Construct a dictionary from a :class:`PRM` instance.

        """
        return cls(**dct)

    """########################### methods for reading .prm files. ##############################"""

    @classmethod
    def read(cls, filename: Union[str, Iterator[Union[str, bytes]]],
             encoding: Optional[str] = None) -> 'PRM':
        """Construct :class:`PRM` instance from a .prm parameter file.

        Parameters
        ----------
        filename : |str|_ or `file object`_
            The path+filename or a file object of the to-be read .prm file.

        encoding : str
            Optional: Encoding used to decode the input (*e.g.* ``"utf-8"``).
            Only relevant when a file object is supplied to **filename** and
            the datastream is *not* in text mode.

        Returns
        -------
        |nanoCAT.PRM|_:
            A :class:`.PRM` instance holding the content of **filename**.

        """

        def iterate(stream: Iterator[Union[str, bytes]]) -> None:
            """Iterate over **stream** and parse its content."""
            iterator = iter(stream) if encoding is None else iterdecode(stream, encoding)
            value = None

            for i in iterator:
                j = i.rstrip('\n')
                if j.startswith('!') or j.startswith('*') or not j:
                    continue  # Ignore comment lines and empty lines

                key = j.split(maxsplit=1)[0]
                if key in cls.HEADERS:
                    kwargs[key.lower()] = value = []
                    kwargs[key.lower() + '_comment'] = value_comment = []
                    if key in ('HBOND', 'NONBONDED'):
                        value.append(i.split()[1:])
                    continue

                v, _, comment = j.partition('!')
                value.append(v.split())
                value_comment.append(comment.strip())

        kwargs = {}

        # filename is an actual filename
        if isinstance(filename, str):
            with open(filename, 'r') as f:
                iterate(f)
        else:  # filename is a data stream
            iterate(filename)

        cls._post_process(kwargs)
        kwargs['filename'] = filename if isinstance(filename, str) else None
        return cls(**kwargs)

    @staticmethod
    def _post_process(kwargs: dict) -> None:
        """Post process the dictionary produced by :meth:`PRM.read`."""
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

    """########################### methods for writing .prm files. ##############################"""

    def write(self, filename: Union[str, None] = None, encoding: Optional[str] = None) -> None:
        """Create a .prm parameter file from this :class:`PRM` instance.

        .. _`file object`: https://docs.python.org/3/glossary.html#term-file-object

        Parameters
        ----------
        filename : |str|_ or `file object`_
            The path+filename or a file object of the output .prm file.
            If ``None``, attempt to pull the filename from :attr:`PRM.filename`.

        encoding : str
            Optional: Encoding used to encode the output (*e.g.* ``"utf-8"``).
            Only relevant when a file object is supplied to **filename** and
            the datastream is *not* in text mode.

        Raises
        ------
        TypeError
            Raised if the filename is specified in neither **filename** nor :attr:`PRM.filename`.

        """
        def serialize() -> str:
            ret = ''
            for key in self.HEADERS[:-1]:
                _key = key.lower()
                df = getattr(self, _key)
                if _key == 'hbond':
                    ret += f'\n{key} {df}\n'
                    continue
                elif not isinstance(df, pd.DataFrame):
                    continue

                iterator = range(df.shape[1] - 1)
                df_str = ' '.join('{:8}' for _ in iterator) + ' ! {}\n'

                if _key != 'nonbonded':
                    ret += f'\n{key}\n'
                else:
                    header = '-\n'.join(i for i in self.nonbonded_header.split('-'))
                    ret += f'\n{key} {header}\n'
                for _, value in df.iterrows():
                    ret += df_str.format(*(('' if i is None else i) for i in value))
            return ret

        _filename = filename if filename is not None else self.filename
        if not _filename:
            raise TypeError("The 'filename' parameter is missing")

        output = serialize()
        output += '\nEND\n'
        output = output[1:]

        if isinstance(_filename, str):
            with open(_filename, 'w') as f:
                f.write(output)

        else:  # _filename is a data stream
            if encoding is None:
                _filename.write(output)
            else:
                _filename.write(output.encode(encoding))
