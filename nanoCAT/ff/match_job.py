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

import os
import io
import stat
from typing import Generator, Any, List, Tuple, Hashable, Optional
from os.path import join, isfile
from itertools import chain

from scm.plams.core.private import sha256
from scm.plams import SingleJob, JobError, Results, Settings, Molecule, FileError

try:
    from scm.plams import writepdb, readpdb
    RDKIT_EX: Optional[ImportError] = None
except ImportError as ex:
    RDKIT_EX = ex

__all__ = ['MatchJob', 'MatchResults']


class MatchResults(Results):
    """The :class:`Results` subclass of :class:`MatchJob`."""

    def recreate_molecule(self) -> Molecule:
        """Create a |Molecule| instance from ``"$JN.pdb"``."""
        pdb_file = self['$JN.pdb']
        try:
            return readpdb(self['$JN.pdb'])
        except AttributeError as ex:
            # readpdb() will pass None to from_rdmol(), resulting in an AttributeError down the line
            raise FileError(f"Failed to parse the content of ...{os.sep}{pdb_file!r}") from ex

    def recreate_settings(self) -> Settings:
        """Construct a |Settings| instance from ``"$JN.run"``."""
        runfile = self['$JN.run']

        # Ignore the first 2 lines
        with open(runfile, 'r') as f:
            for i in f:
                if 'MATCH.pl' in i:
                    args = next(f).split()
                    break
            else:
                raise FileError(f"Failed to parse the content of ...{os.sep}{runfile!r}")

        # Delete the executable and pop the .pdb filename
        del args[0]
        pdb_file = args.pop(-1)

        s = Settings()
        for k, v in zip(args[0::2], args[1::2]):
            k = k[1:].lower()
            s.input[k] = v
        s.input.filename = pdb_file

        return s

    def get_atom_names(self) -> List[str]:
        """Return a list of atom names extracted from ``"$JN.rtf"``."""
        return self._parse_rtf(1)

    def get_atom_types(self) -> List[str]:
        """Return a list of atom types extracted from ``"$JN.rtf"``."""
        return self._parse_rtf(2)

    def get_atom_charges(self) -> List[float]:
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
        >>> s.input.forcefield = 'top_all36_cgenff_new'
        >>> s.input.filename = 'ala.pdb'

        # Command line equivalent: MATCH.pl -forcefield top_all36_cgenff_new ala.pdb
        >>> job = MatchJob(settings=s)
        >>> results = job.run()

    The same example while with a :class:`Molecule` instance:

    .. code:: python

        >>> mol = Molecule('ala.pdb')
        >>> s = Settings()
        >>> s.input.forcefield = 'top_all36_cgenff_new'

        # Command line equivalent: MATCH.pl -forcefield top_all36_cgenff_new ala.pdb
        >>> job = MatchJob(molecule=mol, settings=s)
        >>> results = job.run()

    See Also
    --------
    `10.1002/jcc.21963<https://doi.org/10.1002/jcc.21963>`_
        MATCH: An atom-typing toolset for molecular mechanics force fields,
        J.D. Yesselman, D.J. Price, J.L. Knight and C.L. Brooks III,
        J. Comput. Chem., 2011.

    """

    _result_type = MatchResults

    MATCH: str = os.path.join('$MATCH', 'scripts', 'MATCH.pl')
    pdb: str

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialize a :class:`MatchJob` instance."""
        if RDKIT_EX is not None:
            raise ImportError(f"{RDKIT_EX}; usage of {self.__class__.__name__!r} requires "
                              "the 'rdkit' package") from RDKIT_EX
        super().__init__(**kwargs)
        self._prepare_settings()
        self._prepare_pdb(io.StringIO())

    def _get_ready(self) -> None:
        """Create the runfile."""
        runfile = os.path.join(self.path, self._filename('run'))
        with open(runfile, 'w') as run:
            run.write(self.full_runscript())
        os.chmod(runfile, os.stat(runfile).st_mode | stat.S_IEXEC)

    def get_input(self) -> None:
        """Not implemented; see :meth:`MatchJob.get_runscript`."""
        cls_name = self.__class__.__name__
        raise NotImplementedError(f"`{cls_name}.get_input()` is not implemented; "
                                  f"see `{cls_name}.get_runscript()`")

    def hash_input(self) -> str:
        def get_2tups() -> Generator[Tuple[str, Hashable], None, None]:
            for k, v in self.settings.input.items():
                if isinstance(v, list):
                    v = tuple(v)
                yield k, v
            yield 'pdb', self.pdb
            yield 'type', type(self)

        return sha256(frozenset(get_2tups()))

    def get_runscript(self) -> str:
        """Run a MACTH runscript."""
        self._writepdb()  # Write the .pdb file stored in MatchJob.pdb

        kv_iterator = ((k.strip('-'), str(v)) for k, v in self.settings.input.items())
        args = ' '.join(i for i in chain.from_iterable(kv_iterator))
        return f'"{self.MATCH}" {args} ".{os.sep}{self.name}.pdb"'

    def hash_runscript(self) -> str:
        """Alias for :meth:`MatchJob.hash_input`."""
        return self.hash_input()

    def check(self) -> bool:
        """Check if the .prm, .rtf and top_...rtf files are present."""
        files = {f'{self.name}.prm', f'{self.name}.rtf', f'top_{self.name}.rtf'}
        return files.issubset(self.results.files)

    """###################################### New methods ######################################"""

    def _prepare_settings(self) -> None:
        """Take :attr:`MatchJob.settings` and lower all its keys and strip ``"-"`` characters."""
        s = self.settings.input
        self.settings.input = Settings({k.lower().strip('-'): v for k, v in s.items()})

    def _prepare_pdb(self, stream):
        """Fill :attr:`MatchJob.pdb` with a string-representation of the .pdb file."""
        conitions = {'filename' in self.settings.input, bool(self.molecule)}
        if not any(conitions):
            raise JobError("Ambiguous input: either `molecule` or "
                           "`settings.input.filename` must be specified")
        if all(conitions):
            raise JobError("Ambiguous input: `molecule` and "
                           "`settings.input.filename` cannot be both specified")

        if self.molecule:
            writepdb(self.molecule, stream)
        else:
            filename = self.settings.input.pop('filename')
            writepdb(readpdb(filename), stream)

        self.pdb: str = stream.getvalue()

    def _writepdb(self) -> None:
        """Convert :attr:`MatchJob.pdb` into a pdb file."""
        filename = join(self.path, f'{self.name}.pdb')
        if not isfile(filename):
            writepdb(self.molecule, filename)
