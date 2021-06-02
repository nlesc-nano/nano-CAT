"""
Nano-CAT
========

A collection of tools for the analysis of nanocrystals.

"""

# flake8: noqa: E402

from nanoutils import VersionInfo

from .__version__ import __version__

__author__ = 'Bas van Beek'
__email__ = 'b.f.van.beek@vu.nl'

version_info = VersionInfo.from_str(__version__)
del VersionInfo

__all__ = []
