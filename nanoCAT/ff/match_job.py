"""
nanoCAT.ff.match_job
====================

A module containing a :class:`Job` subclass for interfacing with MATCH_: Multipurpose Atom-Typer for CHARMM.

.. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

Index
-----
.. currentmodule:: nanoCAT.ff.match_job
.. autosummary::
    MatchJob
    MatchResults

API
---
.. autoclass:: MatchJob
    :members:
    :private-members:
    :special-members:

.. autoclass:: MatchResults
    :members:
    :private-members:
    :special-members:

"""  # noqa

import stat
import os
from typing import (Optional, Any, List)
from itertools import chain

from scm.plams.core.basejob import SingleJob
from scm.plams import JobError, Results, Settings, Molecule, FileError

try:
    from scm.plams.interfaces.molecule.rdkit import (writepdb, readpdb)
except ImportError:
    writepdb = Molecule.write
    readpdb = Molecule

__all__ = ['MatchJob', 'MatchResults']


class MatchResults(Results):
    """The :class:`Results` subclass of :class:`MatchJob`."""

    def recreate_molecule(self) -> Molecule:
        """Create a |Molecule| instance from ``"$JN.pdb"``."""
        # Read $JN.pdb
        try:
            return readpdb(self['$JN.pdb'])
        except FileError:
            pass

        # No .pdb file is present; check for of .mol and .mol2 files
        for extension, read_func in Molecule._readformat:
            filename = f'{self.name}.{extension}'
            if filename in self.files:
                return read_func(self[filename])

    def recreate_settings(self) -> Settings:
        """Construct a |Settings| instance from ``"$JN.run"``."""
        runfile = self['$JN.run']

        # Ignore the first 2 lines
        with open(runfile, 'r') as f:
            args = None
            for i in f:
                if 'MATCH.pl' in i:
                    args = next(f).split()
                    break

        if args is None:
            raise FileError(f"recreate_settings: Failed to parse the content of '{runfile}'")

        # Delete the executable and pop the .pdb filename
        del args[0]
        pdb_file = args.pop(-1)

        s = Settings()
        for k, v in zip(args[0::2], args[1::2]):
            k = k[1:]
            s.input[k] = v
        s.input.filename = pdb_file

        return s

    def get_atom_names(self) -> List[str]:
        """Return a list of atom names extracted from ``"$JN.rtf"``."""
        return self._parse_rtf(1)

    def get_atom_types(self) -> List[str]:
        """Return a list of atom types extracted from ``"$JN.rtf"``."""
        return self._parse_rtf(2)

    def get_charges(self) -> List[float]:
        """Return a list of atomic charges extracted from ``"$JN.rtf"``."""
        return self._parse_rtf(3, as_type=float)

    def _parse_rtf(self, column: int, block: str = 'ATOM', as_type: type = str) -> list:
        """Extract values from the **block** in ``"$JN.rtf"``."""
        filename = self['$JN.rtf']
        ret = []
        i, j = len(block), column

        with open(filename, 'r') as f:
            for item in f:
                if item[:i] == block:
                    item_list = item.split()
                    ret.append(as_type(item_list[j]))

        return ret


class MatchJob(SingleJob):
    """A :class:`Job` subclass for interfacing with MATCH_: Multipurpose Atom-Typer for CHARMM.

    .. _MATCH:
        http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

    Examples
    --------
    An example :class:`MatchJob` job:

    .. code:: python

        >>> s = Settings()
        >>> s.input.forcefield = 'top_all36_cgenff'
        >>> s.input.filename = 'ala.pdb'

        # Command line equivalent: MATCH.pl -forcefield top_all36_cgenff ala.pdb
        >>> job = MatchJob(settings=s)
        >>> results = job.run()

    The same example while with a :class:`Molecule` instance:

    .. code:: python

        >>> mol = Molecule('ala.pdb')
        >>> s = Settings()
        >>> s.input.forcefield = 'top_all36_cgenff'

        # Command line equivalent: MATCH.pl -forcefield top_all36_cgenff ala.pdb
        >>> job = MatchJob(molecule=mol, settings=s)
        >>> results = job.run()

    See Also
    --------
    Publication:
        `MATCH: An atom-typing toolset for molecular mechanics force fields,
        J.D. Yesselman, D.J. Price, J.L. Knight and C.L. Brooks III,
        J. Comput. Chem., 2011 <https://doi.org/10.1002/jcc.21963>`_

    """

    _result_type = MatchResults

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a :class:`MathcJob` instance."""
        if self.MATCH is None:
            raise JobError("The 'MATCH' environment variables has not been set")
        super().__init__(**kwargs)

    def _get_ready(self) -> None:
        """Create the runfile."""
        runfile = os.path.join(self.path, self._filename('run'))
        with open(runfile, 'w') as run:
            run.write(self.full_runscript())
        os.chmod(runfile, os.stat(runfile).st_mode | stat.S_IEXEC)

    def get_input(self) -> None: return None
    def hash_input(self) -> str: return self.hash_runscript()

    def get_runscript(self) -> str:
        """Run a MACTH runscript."""
        # Create a .pdb file from self.molecule
        if self.molecule:
            self._writepdb()

        kwargs = {self._sanitize_key(k): str(v) for k, v in self.settings.input.items()}
        filename = kwargs.pop('-filename')

        kwargs_iter = chain.from_iterable(kwargs.items())
        args = ' '.join(i for i in kwargs_iter)
        return f'{self.MATCH} {args} {filename}'

    def check(self) -> bool:
        files = (f'{self.name}.prm', f'{self.name}.rtf', f'top_{self.name}.rtf')
        return all(i in self.results.files for i in files)

    """###################################### New methods ######################################"""

    try:
        #: The path to ``"$MATCH/scripts/MATCH.pl"`` executable.
        MATCH: str = repr(os.path.join(os.environ['MATCH'], 'scripts', 'MATCH.pl'))
    except KeyError:
        #: The path to ``"$MATCH/scripts/MATCH.pl"`` executable.
        MATCH: Optional[str] = None

    @staticmethod
    def _sanitize_key(key: str) -> str:
        """Lower *key* and prepended it with the ``"-"`` character."""
        ret = key if key.startswith('-') else f'-{key}'
        return ret.lower()

    def _writepdb(self) -> None:
        """Convert :attr:`MatchJob.molecule` into a pdb file."""
        filename = os.path.join(self.path, self.name + '.pdb')
        writepdb(self.molecule, filename)
        self.settings.input.filename = repr(filename)
