"""
nanoCAT.ff.psf
===============

A class for reading and generating protein structure (.psf) files.

Index
-----
.. currentmodule:: nanoCAT.ff.psf
.. autosummary::
    PSF

API
---
.. autoclass:: nanoCAT.ff.psf
    :members:
    :private-members:
    :special-members:

"""

import textwrap
from typing import (Dict, Optional, Any, Union)
from itertools import chain

import numpy as np
import pandas as pd

from CAT.frozen_settings import FrozenSettings

__all__ = ['PSF']

_SHAPE_DICT = FrozenSettings({
    'filename': {'shape': 1},
    'title': {'shape': 1},
    'atoms': {'shape': 8},
    'bonds': {'shape': 2, 'row_len': 4, 'header': '{:>10d} !NBOND: bonds'},
    'angles': {'shape': 3, 'row_len': 3, 'header': '{:>10d} !NTHETA: angles'},
    'dihedrals': {'shape': 4, 'row_len': 2, 'header': '{:>10d} !NPHI: dihedrals'},
    'impropers': {'shape': 4, 'row_len': 2, 'header': '{:>10d} !NIMPHI: impropers'},
    'donors': {'shape': 1, 'row_len': 8, 'header': '{:>10d} !NDON: donors'},
    'acceptors': {'shape': 1, 'row_len': 8, 'header': '{:>10d} !NACC: acceptors'},
    'no_nonbonded': {'shape': 2, 'row_len': 4, 'header': '{:>10d} !NNB'}
})


class PSF:
    """A class for storing and parsing protein structure files (.psf).

    Attributes
    ----------
    _filename : :math:`1` |np.ndarray|_ [|np.str_|_]
        A 1D array with a single string as filename.
        Seel also :meth:`PSF.filename`

    title : :math:`a` |np.ndarray|_ [|np.str_|_]
        A 1D array of strings holding the ``"title"`` block.

    atoms : :math:`b*8` |pd.DataFrame|_
        A Pandas DataFrame holding the ``"atoms"`` block.

    bonds : :math:`c*2` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-pairs defining bonds.

    angles : :math:`d*3` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atoms defining angles.

    dihedrals : :math:`e*4` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atoms defining proper dihedral angles.

    impropers : :math:`f*4` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atoms defining improper dihedral angles.

    donors : :math:`g*1` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the atomic indices of all hydrogen-bond donors.

    acceptors : :math:`h*1` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the atomic indices of all hydrogen-bond acceptors.

    no_nonbonded : :math:`i*2` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-pairs whose nonbonded
        interactions should be ignored.

    """

    def __init__(self, filename=None, title=None, atoms=None, bonds=None, angles=None,
                 dihedrals=None, impropers=None, donors=None, acceptors=None, no_nonbonded=None):
        """Initialize a :class:`PSF` instance."""
        self.filename = filename
        self.title = np.array(title, dtype=str, ndmin=1, copy=False)
        self.atoms = atoms
        self.bonds = np.array(bonds, dtype=int, ndmin=2, copy=False)
        self.angles = np.array(angles, dtype=int, ndmin=2, copy=False)
        self.dihedrals = np.array(dihedrals, dtype=int, ndmin=2, copy=False)
        self.impropers = np.array(impropers, dtype=int, ndmin=2, copy=False)
        self.donors = np.array(donors, dtype=int, ndmin=2, copy=False)
        self.acceptors = np.array(acceptors, dtype=int, ndmin=2, copy=False)
        self.no_nonbonded = np.array(no_nonbonded, dtype=int, ndmin=2, copy=False)

    def __str__(self) -> str:
        """Return the canonical string representation of this instance."""
        def _str(k: str, v: Any) -> str:
            return f'{k:{width}} = ' + textwrap.indent(str(v), indent2)[len(indent2):]

        width = max(len(k) for k in vars(self))
        indent1 = ' ' * 4
        indent2 = ' ' * (3 + width)
        with np.printoptions(threshold=20, edgeitems=10):
            with pd.option_context('display.max_rows', 20):
                ret = ',\n'.join(_str(k, v) for k, v in vars(self).items())

        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent1)}\n)'

    def __repr__(self) -> str:
        """Return the canonical string representation of this instance."""
        return str(self)

    def __eq__(self, value: Any) -> bool:
        """Check if this instance is equivalent to **value**."""
        if self.__class__ is not value.__class__:
            return False

        try:  # Check if the object attribute values are identical
            for k, v1 in vars(self).items():
                v2 = getattr(value, k)
                assert (v1 == v2).all()
        except (AttributeError, AssertionError):
            return False  # An attribute is missing or not equivalent

        return True

    def copy(self) -> 'PSF':
        """Return a copy of this instance."""
        kwargs = {k: v.copy() for k, v in vars(self).items()}
        return self.from_dict(kwargs)

    def __copy__(self) -> 'PSF':
        """Return a copy of this instance."""
        return self.copy()

    def __deepcopy__(self, memo: Any = None) -> 'PSF':
        """Return a copy of this instance."""
        return self.copy()

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Construct a :class:`dict` from this instance"""
        return vars(self)

    @classmethod
    def from_dict(cls, psf_dict: dict) -> 'PSF':
        """Construct a :class:`PSF` instance from **psf_dict**."""
        return cls(**psf_dict)

    @property
    def filename(self) -> str:
        """Return the :class:`str` stored in the :attr:`PSF._filename` attribute."""
        return self._filename[0]

    @filename.setter
    def filename(self, filename: Union[str, np.ndarray]) -> None:
        """Set the :attr:`PSF._filename` attribute."""
        _filename = '' if filename is None else filename
        self._filename = np.array(_filename, dtype=str, ndmin=1, copy=False)

    """########################### methods for reading .psf files. ##############################"""

    @classmethod
    def read(cls, filename: str) -> 'PSF':
        """Read a protein structure file (.psf) and return the various .psf blocks as a dictionary.

        Depending on the content of the .psf file, the dictionary can contain
        the following keys and values:

            * *title*: list of remarks (str)
            * *atoms*: A dataframe of atoms
            * *bonds*: A :math:`i*2` array of atomic indices defining bonds
            * *angles*: A :math:`j*3` array of atomic indices defining angles
            * *dihedrals*: A :math:`k*4` array of atomic indices defining proper dihedral angles
            * *impropers*: A :math:`l*4` array of atomic indices defining improper dihedral angles
            * *donors*: A :math:`m*1` array of atomic indices defining hydrogen-bond donors
            * *acceptors*: A :math:`n*1` array of atomic indices defining hydrogen-bond acceptors
            * *no_nonbonded*: A :math:`o*2` array of atomic indices defining to-be ignore nonbonded
              interactions

        Examples
        --------
        The dictionary produced by this function be fed into :func:`.write_psf` to create a new .psf
        file:

        .. code:: python

            >>> psf_dict = read_psf('old_file.psf')
            >>> write_psf('new_file.psf', **psf_dict)

        Parameters
        ----------
        str filename:
            The path + filename of a .psf file.

        Returns
        -------
        |FOX.PSF|_:
            A :class:`.PSF` instance holding the content of a .psf file.

        """
        header_dict = {
            '!NTITLE': 'title',
            '!NATOM': 'atoms',
            '!NBOND': 'bonds',
            '!NTHETA': 'angles',
            '!NPHI': 'dihedrals',
            '!NIMPHI': 'impropers',
            '!NDON': 'donors',
            '!NACC': 'acceptors',
            '!NNB': 'no_nonbonded'
        }

        ret: dict = {}
        with open(filename, 'r') as f:
            next(f)  # Skip the first line
            for i in f:
                # Search for .psf blocks
                if i == '\n':  # Empty line
                    continue

                # Read the .psf block header
                key = header_dict[i.split()[1].rstrip(':')]
                ret[key] = value = []

                # Read .psf blocks
                try:
                    j = next(f)
                except StopIteration:
                    break

                while j != '\n':
                    value.append(j.split())
                    try:
                        j = next(f)
                    except StopIteration:
                        break

        return cls(**PSF._post_process_psf(ret))

    @staticmethod
    def _post_process_psf(psf_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Post-process the output of :func:`.read_psf`, casting the values into appropiate
        objects:

        * The title block is converted into an (un-nested) list of strings.
        * The atoms block is converted into a Pandas DataFrame.
        * All other blocks are converted into a flattened array of integers.

        Parameters
        ----------
        psf_dict : dict [str, |np.ndarray|_]
            A dictionary holding the content of a .psf file (see :func:`.read_psf`).

        Returns
        -------
        |dict|_ [|str|_, |np.ndarray|_]:
            The .psf output, **psf_dict**, with properly formatted values.

        """
        for key, value in psf_dict.items():  # Post-process the output
            # Cast into a flattened array of indices
            if key not in ('title', 'atoms'):
                ar = np.fromiter(chain.from_iterable(value), dtype=int)
                ar.shape = len(ar) // _SHAPE_DICT[key].shape, _SHAPE_DICT[key].shape
                psf_dict[key] = ar

            # Cast the atoms block into a dataframe
            elif key == 'atoms':
                df = pd.DataFrame(value)
                df[0] = df[0].astype(int, copy=False)
                df.set_index(0, inplace=True)
                df.index.name = 'ID'
                df.columns = ['segment name', 'residue ID', 'residue name',
                              'atom name', 'atom type', 'charge', 'mass', '0']
                df['residue ID'] = df['residue ID'].astype(int, copy=False)
                df['charge'] = df['charge'].astype(float, copy=False)
                df['mass'] = df['mass'].astype(float, copy=False)
                df['0'] = df['0'].astype(int, copy=False)
                psf_dict[key] = df

            # Cast the title in a list of strings
            elif key == 'title':
                psf_dict[key] = np.array([' '.join(i).strip('REMARKS ') for i in value])

        return psf_dict

    """########################### methods for writing .psf files. ##############################"""

    def write(self, filename: Optional[str] = None) -> None:
        """Create a protein structure file (.psf) out of a :class:`.PSF` instance.

        Parameters
        ----------
        filename : str
            The path+filename of the .psf file.
            If ``None``, attempt to pull the name from :attr:`.PSF._filename`.

        Raises
        ------
        TypeError
            Raised if the filename is specified in neither **filename** nor :attr:`.PSF._filename`.

        """
        filename = filename or self.filename
        if not filename:
            raise TypeError("The 'filename' argument is missing")

        top = self._serialize_top()
        bottom = self._serialize_bottom()

        # Write the .psf file
        with open(filename, 'w') as f:
            f.write(top)
            f.write(bottom[1:])

    def _serialize_top(self) -> str:
        """Serialize the top-most section of the to-be create .psf file.

        The following blocks are seralized:

            * :attr:`.PSF.title`
            * :attr:`.PSF.atoms`

        Returns
        -------
        |str|_
            A string constructed from the above-mentioned psf blocks.

        See Also
        --------
        :meth:`.PSF.write_psf`
            The main method for writing .psf files.

        """
        # Prepare the !NTITLE block
        ret = 'PSF EXT\n'
        if not any(self.title):
            ret += ('         2 !NTITLE'
                    '   REMARKS PSF file generated with Nano-CAT'
                    '   REMARKS https://github.com/nlesc-nano/nano-CAT')
        else:
            ret += '\n{:>10d} !NTITLE'.format(self.title.shape[0])
            for i in self.title:
                ret += f'   REMARKS {i}'

        # Prepare the !NATOM block
        if self.atoms.shape[1] != 0:
            ret += '\n\n{:>10d} !NATOM\n'.format(self.atoms.shape[0])
            string = '{:>10d} {:8.8} {:<8d} {:8.8} {:8.8} {:6.6} {:>9f} {:>15f} {:>8d}'
            for i, j in self.atoms.iterrows():
                ret += string.format(*[i]+j.values.tolist()) + '\n'
        else:
            ret += '         0 !NATOM\n'.format(0)

        return ret

    def _serialize_bottom(self) -> str:
        """Serialize the bottom-most section of the to-be create .psf file.

        The following blocks are seralized:

            * :attr:`.PSF.bonds`
            * :attr:`.PSF.angles`
            * :attr:`.PSF.dihedrals`
            * :attr:`.PSF.impropers`
            * :attr:`.PSF.donors`
            * :attr:`.PSF.acceptors`
            * :attr:`.PSF.no_nonbonded`

        Returns
        -------
        |str|_
            A string constructed from the above-mentioned psf blocks.

        See Also
        --------
        :meth:`.PSF.write_psf`
            The main method for writing .psf files.

        """
        sections = ('bonds', 'angles', 'dihedrals', 'impropers',
                    'donors', 'acceptors', 'no_nonbonded')

        ret = ''
        for attr in sections:
            if attr in ('title', 'atoms'):
                continue

            header = _SHAPE_DICT[attr].header
            row_len = _SHAPE_DICT[attr].row_len
            value = getattr(self, attr)
            if value is None:
                ret += '\n\n' + header.format(0)
            else:
                ret += '\n\n' + header.format(value.shape[0])
                ret += '\n' + self.serialize_array(value, row_len)

        return ret

    @staticmethod
    def serialize_array(array: np.ndarray, items_per_row: int = 4) -> str:
        """Serialize an array into a single string.

        Newlines are placed for every **items_per_row** rows in **array**.

        Parameters
        ----------
        array : |np.ndarray|_
            A 2D array.

        items_per_row : int
            The number of values per row before switching to a new line.

        Returns
        -------
        |str|_:
            A serialized array.
        """
        if len(array) == 0:
            return ''

        ret = ''
        k = 0
        for i in array:
            for j in i:
                ret += '{:>10d}'.format(j)
            k += 1
            if k == items_per_row:
                k = 0
                ret += '\n'

        return ret

    """################### methods for altering atomic/molecular information. ###################"""

    def update_atom_charge(self, atom_type: str, charge: float) -> None:
        """Change the charge of a specific atom type to **charge**.

        Performs an inplace update of the ``"charge"`` column in :attr:`PSF.atoms`.

        Parameters
        ----------
        atom_type : str
            An atom type in the ``"atom type"`` column in :attr:`.PSF.atoms`.

        charge : float
            A new atomic charge to-be associated with **atom_type**.

        """
        condition = self.atoms['atom type'] == atom_type
        self.atoms.loc[condition, 'charge'] = charge
