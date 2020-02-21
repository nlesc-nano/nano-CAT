"""
CAT.recipes
===========

A number of recipes constructed using the CAT and Nano-CAT packages.

Examples
--------
.. code:: python

    >>> from nanoCAT.recipes import bulk_workflow
    >>> from nanoCAT.recipes import replace_surface
    >>> from nanoCAT.recipes import dissociate_surface, row_accumulator
    ...

"""

from .bulk import bulk_workflow
from .mark_surface import replace_surface
from .surface_dissociation import dissociate_surface, row_accumulator

__all__ = ['bulk_workflow', 'replace_surface', 'dissociate_surface', 'row_accumulator']
