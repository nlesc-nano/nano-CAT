import textwrap
from abc import ABC
from copy import deepcopy
from typing import (Any, Dict, FrozenSet, Iterator, Tuple)

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
        iterator = self._str_iterator()
        ret = ',\n'.join(_str(k, v) for k, v in iterator)

        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent1)}\n)'

    def _str_iterator(self) -> Iterator[Tuple[str, Any]]:
        """Return an iterator for this instances' :meth:`.__str__` method."""
        return self.as_dict().items()

    def __repr__(self) -> str:
        """Return a machine-readable string representation of this instance."""
        return str(self)

    def __eq__(self, value: Any) -> bool:
        """Check if this instance is equivalent to **value**."""
        if type(self) is not type(value):
            return False
        return self.as_dict() == self.as_dict()

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

        No attributes specified in :data:`._PRIVATE_ATTR` will be included in the to-be
        returned dictionary.

        Returns
        -------
        :class:`dict` [:class:`str`, :class:`.Any`]
            A dictionary of arrays with keyword arguments for initializing a new
            instance of this class.

        See also
        --------
        :func:`vars`:
            Construct a dictionary from this instance with *all* instance attributes.

        :meth:`AbstractDataClass.from_dict`:
            Construct a instance of this objects' class from a dictionary with keyword arguments.

        """
        if not self._PRIVATE_ATTR:
            return vars(self)
        return {k: v for k, v in vars(self).items() if k not in self._PRIVATE_ATTR}

    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> 'AbstractDataClass':
        """Construct a instance of this objects' class from a dictionary with keyword arguments.

        Parameters
        ----------
        dct : :class:`dict` [:class:`str`, :class:`.Any`]
            A dictionary with keyword arguments for constructing a new
            :class:`AbstractDataClass` instance.

        Returns
        -------
        :class:`AbstractDataClass`
            A new instance of this object's class constructed from **dct**.

        See also
        --------
        :meth:`AbstractDataClass.as_dict`:
            Construct a dictionary from this instance with all non-private instance attributes.

        """
        return cls(**dct)

    @classmethod
    def inherit_annotations(cls) -> type:
        def decorator(type_: type) -> type:
            cls_meth = getattr(cls, type_.__name__)
            annotations = cls_meth.__annotations__.copy()
            dct = type_.__annotations__ = annotations.update(type_.__annotations__)
            if 'return' in dct and dct['return'] == cls.__name__:
                dct['return'] = type_.__self__.__name__
            return type_
        return decorator
