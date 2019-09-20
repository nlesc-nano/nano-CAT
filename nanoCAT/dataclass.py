import textwrap
from abc import ABC
from copy import deepcopy
from typing import (Any, Dict, FrozenSet)

__all__ = ['AbstractDataClass']


class AbstractDataClass(ABC):
    """A generic dataclass."""

    #: A :class:`frozenset` with the names of private instance attributes.
    #: These attributes will be excluded whenever calling :meth:`AbstractDataClass.as_dict`.
    _PRIVATE_ATTR: FrozenSet[str] = frozenset({})

    def __str__(self) -> str:
        """Return a human-readable string representation of this instance."""
        def _str(k: str, v: Any) -> str:
            return f'{k:{width}} = ' + textwrap.indent(repr(v), indent2)[len(indent2):]

        width = max(len(k) for k in vars(self))
        indent1 = ' ' * 4
        indent2 = ' ' * (3 + width)
        ret = ',\n'.join(_str(k, v) for k, v in self.as_dict().items())

        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent1)}\n)'

    def __repr__(self) -> str:
        """Return a machine-readable string representation of this instance."""
        return str(self)

    def __eq__(self, value: Any) -> bool:
        """Check if this instance is equivalent to **value**."""
        if type(self) is not type(value):
            return False
        return vars(self) == vars(value)

    def copy(self, deep: bool = False) -> 'AbstractDataClass':
        """Return a deep or shallow copy of this instance.

        Returns
        -------
        :class:`AbstractDataClass`
            A new instance constructed from this instance.

        """
        kwargs = deepcopy(self.as_dict()) if deep else self.as_dict()
        return self.from_dict(kwargs)

    def __copy__(self) -> 'AbstractDataClass':
        """Return a shallow copy of this instance."""
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None) -> 'AbstractDataClass':
        """Return a deep copy of this instance."""
        return self.copy(deep=True)

    def as_dict(self) -> Dict[str, Any]:
        """Construct a dictionary from this instance with all non-private instance attributes.

        No attributes specified in :data:`_PRIVATE_ATTR` will be included in the to-be
        returned dictionary.

        Returns
        -------
        :class:`dict` [:class:`str`, :class:`.Any`]
            A dictionary of arrays with keyword arguments for initializing a new
            :class:`AbstractDataClass` instance.

        See also
        --------
        :func:`vars`:
            Construct a dictionary from this instance with *all* instance attributes.

        :meth:`AbstractDataClass.from_dict`:
            Construct a :class:`AbstractDataClass` instance from
            a dictionary with keyword arguments.

        """
        ret = vars(self)
        for key in self._PRIVATE_ATTR:
            del ret[key]
        return ret

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> 'AbstractDataClass':
        """Construct a :class:`AbstractDataClass` instance from a dictionary with keyword arguments.

        Parameters
        ----------
        dct : :class:`dict` [:class:`str`, :class:`.Any`]
            A dictionary with keyword arguments for constructing a new
            :class:`AbstractDataClass` instance.

        Returns
        -------
        :class:`AbstractDataClass`
            A new :class:`AbstractDataClass` instance constructed from **dct**.

        See also
        --------
        :meth:`AbstractDataClass.as_dict`:
            Construct a dictionary from a :class:`AbstractDataClass` instance.

        """
        return cls(**dct)
