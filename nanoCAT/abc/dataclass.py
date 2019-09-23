import textwrap
from abc import ABC
from copy import deepcopy
from typing import (Any, Dict, FrozenSet, Iterable, Tuple)

__all__ = ['AbstractDataClass']


class AbstractDataClass(ABC):
    """A generic dataclass."""

    #: A :class:`frozenset` with the names of private instance attributes.
    #: These attributes will be excluded whenever calling :meth:`AbstractDataClass.as_dict`,
    #: printing an instance or comparing objects.
    _PRIVATE_ATTR: FrozenSet[str] = frozenset({})

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        def _str(k: str, v: Any) -> str:
            return f'{k:{width}} = ' + textwrap.indent(repr(v), indent2)[len(indent2):]

        width = max(len(k) for k in vars(self))
        indent1 = ' ' * 4
        indent2 = ' ' * (3 + width)
        iterator = self._str_iterator()
        ret = ',\n'.join(_str(k, v) for k, v in iterator)

        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent1)}\n)'

    __repr__ = __str__

    def _str_iterator(self) -> Iterable[Tuple[str, Any]]:
        """Return an iterable for this instances' :meth:`.__str__` method."""
        return self.as_dict().items()

    def __eq__(self, value: Any) -> bool:
        """Check if this instance is equivalent to **value**."""
        if type(self) is not type(value):
            return False
        return self.as_dict() == self.as_dict()

    def copy(self, deep: bool = False, copy_private: bool = False) -> 'AbstractDataClass':
        """Return a deep or shallow copy of this instance.

        Parameters
        ----------
        return_private : :class:`bool`
            If ``True``, copy both public and private instance variables.
            Private instance variables are defined in :data:`AbstractDataClass._PRIVATE_ATTR`.

        Returns
        -------
        :class:`AbstractDataClass`
            A new instance constructed from this instance.

        """
        kwargs = deepcopy(self.as_dict(copy_private)) if deep else self.as_dict(copy_private)
        return self.from_dict(kwargs)

    def __copy__(self) -> 'AbstractDataClass':
        """Return a shallow copy of this instance."""
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None) -> 'AbstractDataClass':
        """Return a deep copy of this instance."""
        return self.copy(deep=True)

    def as_dict(self, return_private: bool = False) -> Dict[str, Any]:
        """Construct a dictionary from this instance with all non-private instance variables.

        No attributes specified in :data:`AbstractDataClass._PRIVATE_ATTR` will be included in
        the to-be returned dictionary.

        Parameters
        ----------
        return_private : :class:`bool`
            If ``True``, return both public and private instance variables.
            Private instance variables are defined in :data:`AbstractDataClass._PRIVATE_ATTR`.

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
        ret = vars(self).copy()
        if not self._PRIVATE_ATTR:
            return ret
        for key in self._PRIVATE_ATTR:
            del ret[key]

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
        """A decorator for inheriting annotations and docstrings.

        Can be applied to methods of :class:`AbstractDataClass` subclasses to automatically
        inherit the docstring and annotations of identical-named functions of its superclass.

        Examples
        --------
        .. code:: python

            >>> class sub_class(AbstractDataClass)
            ...
            ...     @AbstractDataClass.inherit_annotations()
            ...     def as_dict(self, return_private=False):
            ...         pass

            >>> sub_class.as_dict.__doc__ == AbstractDataClass.as_dict.__doc__
            True

            >>> sub_class.as_dict.__annotations__ == AbstractDataClass.as_dict.__annotations__
            True

        """
        def decorator(sub_attr: type) -> type:
            super_attr = getattr(cls, sub_attr.__name__)
            if not sub_attr.__annotations__:
                sub_attr.__annotations__ = super_attr.__annotations__.copy()
            if sub_attr.__doc__ is None:
                sub_attr.__doc__ = super_attr.__doc__
            return sub_attr
        return decorator
