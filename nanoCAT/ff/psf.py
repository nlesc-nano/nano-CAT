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
.. autoclass:: nanoCAT.ff.psf.PSF
    :members:
    :private-members:
    :special-members:

"""

import textwrap
from typing import (Dict, Optional, Any, Iterable)
from itertools import chain
from collections.abc import Container

import numpy as np
import pandas as pd

from scm.plams import Molecule

from CAT.frozen_settings import FrozenSettings
from nanoCAT.ff.mol_topology import (get_bonds, get_angles, get_dihedrals, get_impropers)

__all__ = ['PSF']


class PSF(Container):
    """A container for managing protein structure files.

    The :class:`PSF` class has access to three general sets of methods:

    * Methods for reading & constructing .psf files: :meth:`PSF.read` and :meth:`PSF.write`.
    * Methods for updating atom types: :meth:`PSF.update_atom_charge`
      and :meth:`PSF.update_atom_type`
    * Methods for extracting bond, angle and dihedral-pairs from :class:`Molecule` instances:
      :meth:`PSF.get_bonds`, :meth:`PSF.get_angles`, :meth:`PSF.get_dihedrals`
      and :meth:`PSF.get_impropers`.

    Attributes
    ----------
    _filename : :math:`1` |np.ndarray|_ [|np.str_|_]
        A 1D array with a single string as filename.
        This attribute should be accessed via the :meth:`PSF.filename` property.

    _title : :math:`a` |np.ndarray|_ [|np.str_|_]
        A 1D array of strings holding the ``"title"`` block.
        This attribute should be accessed via the :meth:`PSF.title` property.

    atoms : :math:`b*8` |pd.DataFrame|_
        A Pandas DataFrame holding the ``"atoms"`` block.
        The DataFrame should possess the following collumn keys:

        * ``"segment name"``
        * ``"residue ID"``
        * ``"residue name"``
        * ``"atom name"``
        * ``"atom type"``
        * ``"charge"``
        * ``"mass"``
        * ``"0"``

    _bonds : :math:`c*2` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-pairs defining bonds.
        Indices are expected to be 1-based.
        This attribute should be accessed via the :meth:`PSF.bonds` property.

    _angles : :math:`d*3` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-triplets defining angles.
        Indices are expected to be 1-based.
        This attribute should be accessed via the :meth:`PSF.angles` property.

    _dihedrals : :math:`e*4` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-quartets defining proper dihedral angles.
        Indices are expected to be 1-based.
        This attribute should be accessed via the :meth:`PSF.dihedrals` property.

    _impropers : :math:`f*4` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-quartets defining improper dihedral angles.
        Indices are expected to be 1-based.
        This attribute should be accessed via the :meth:`PSF.impropers` property.

    _donors : :math:`g*1` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the atomic indices of all hydrogen-bond donors.
        Indices are expected to be 1-based.
        This attribute should be accessed via the :meth:`PSF.donors` property.

    _acceptors : :math:`h*1` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the atomic indices of all hydrogen-bond acceptors.
        Indices are expected to be 1-based.
        This attribute should be accessed via the :meth:`PSF.acceptors` property.

    _no_nonbonded : :math:`i*2` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-pairs whose nonbonded
        interactions should be ignored.
        Indices are expected to be 1-based.
        This attribute should be accessed via the :meth:`PSF.no_nonbonded` property.

    """

    #: A dictionary containg array shapes among other things
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

    #: A dictionary mapping .psf headers to :class:`PSF` attribute names
    _HEADER_DICT = FrozenSettings({
        '!NTITLE': 'title',
        '!NATOM': 'atoms',
        '!NBOND': 'bonds',
        '!NTHETA': 'angles',
        '!NPHI': 'dihedrals',
        '!NIMPHI': 'impropers',
        '!NDON': 'donors',
        '!NACC': 'acceptors',
        '!NNB': 'no_nonbonded'
    })

    def __init__(self, filename=None, title=None, atoms=None, bonds=None,
                 angles=None, dihedrals=None, impropers=None, donors=None,
                 acceptors=None, no_nonbonded=None) -> None:
        """Initialize a :class:`PSF` instance."""
        self.filename = filename
        self.title = title
        self.atoms = atoms if atoms is not None else pd.DataFrame()
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.impropers = impropers
        self.donors = donors
        self.acceptors = acceptors
        self.no_nonbonded = no_nonbonded

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        def _str(k: str, v: Any) -> str:
            _k = k[1:] if k.startswith('_') else k
            return f'{_k:{width}} = ' + textwrap.indent(str(v), indent2)[len(indent2):]

        width = max(len(k) for k in vars(self))
        indent1 = ' ' * 4
        indent2 = ' ' * (3 + width)
        with np.printoptions(threshold=20, edgeitems=10):
            with pd.option_context('display.max_rows', 20):
                ret = ',\n'.join(_str(k, v) for k, v in vars(self).items())

        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent1)}\n)'

    def __eq__(self, value: Any) -> bool:
        """Check if this instance is equivalent to **value**."""
        if self.__class__ is not value.__class__:
            return False

        try:  # Check if the object attribute values are identical
            for k, v1 in vars(self).items():
                v1 = np.asarray(v1)
                v2 = np.asarray(getattr(value, k))
                assert (v1 == v2).all()
        except (AttributeError, AssertionError):
            return False  # An attribute is missing or not equivalent

        return True

    def copy(self) -> 'PSF':
        """Return a copy of this instance."""
        kwargs = {(k[1:] if k.startswith('_') else k): v.copy() for k, v in vars(self).items()}
        return self.from_dict(kwargs)

    def __copy__(self) -> 'PSF':
        """Return a copy of this instance."""
        return self.copy()

    def __deepcopy__(self, memo: Any = None) -> 'PSF':
        """Return a copy of this instance."""
        return self.copy()

    def __contains__(self, value: Any) -> bool:
        """Check if this instance contains **value**."""
        return value in vars(self).values()

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Construct a :class:`dict` from this instance."""
        return {(k[1:] if k.startswith('_') else k): v for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, psf_dict: Dict[str, np.ndarray]) -> 'PSF':
        """Construct a :class:`PSF` instance from **psf_dict**."""
        return cls(**psf_dict)

    """###################################### Properties ########################################"""

    @property
    def filename(self) -> str: return self._filename[0]

    @filename.setter
    def filename(self, value: Iterable): self._set_nd_array('_filename', value, 1, str)

    @property
    def title(self) -> np.ndarray: return self._title

    @title.setter
    def title(self, value: Iterable): self._set_nd_array('_title', value, 1, str)

    @property
    def bonds(self) -> np.ndarray: return self._bonds

    @bonds.setter
    def bonds(self, value: Iterable): self._set_nd_array('_bonds', value, 2, int)

    @property
    def angles(self) -> np.ndarray: return self._angles

    @angles.setter
    def angles(self, value: Iterable): self._set_nd_array('_angles', value, 2, int)

    @property
    def dihedrals(self) -> np.ndarray: return self._dihedrals

    @dihedrals.setter
    def dihedrals(self, value: Iterable): self._set_nd_array('_dihedrals', value, 2, int)

    @property
    def impropers(self) -> np.ndarray: return self._impropers

    @impropers.setter
    def impropers(self, value: Iterable): self._set_nd_array('_impropers', value, 2, int)

    @property
    def donors(self) -> np.ndarray: return self._donors

    @donors.setter
    def donors(self, value: Iterable): self._set_nd_array('_donors', value, 2, int)

    @property
    def acceptors(self) -> np.ndarray: return self._acceptors

    @acceptors.setter
    def acceptors(self, value: Iterable): self._set_nd_array('_acceptors', value, 2, int)

    @property
    def no_nonbonded(self) -> np.ndarray: return self._no_nonbonded

    @no_nonbonded.setter
    def no_nonbonded(self, value: Iterable): self._set_nd_array('_no_nonbonded', value, 2, int)

    def _set_nd_array(self, name: str, value: Optional[np.ndarray],
                      ndmin: int, dtype: type) -> None:
        """Assign an array-like object, **value**, to the **name** attribute as ndarray.

        Performs an inplace update of this instance.

        Parameters
        ----------
        name : str
            The name of the to-be set attribute.

        value : array-like
            The array-like object to-be assigned to **name**.
            The supplied object is converted into into an array beforehand.

        ndmin : int
            The minimum number of dimensions of the to-be assigned array.

        dtype : |type|_ or |np.dtype|_
            The desired datatype of the to-be assigned array.

        Exceptions
        ----------
        ValueError:
            Raised if value array construction was unsuccessful.

        """
        _value = [] if value is None else value
        try:
            array = np.array(_value, dtype=dtype, ndmin=ndmin, copy=False)
            setattr(self, name, array)
        except ValueError as ex:
            ex.args = (f"The parameter '{name}' expects a '{ndmin}'d array-like object consisting "
                       f"of '{dtype}'; observed type: '{value.__class__.__name__}'",)
            raise ex

    """########################### methods for reading .psf files. ##############################"""

    @classmethod
    def read(cls, filename: str) -> 'PSF':
        """Construct :class:`PSF` instance from a protein structure file (.psf).

        Parameters
        ----------
        filename : str
            The path+filename of a .psf file.

        Returns
        -------
        |nanoCAT.PSF|_:
            A :class:`.PSF` instance holding the content of **filename**.

        """
        ret = {}
        with open(filename, 'r') as f:
            next(f)  # Skip the first line
            for i in f:
                # Search for .psf blocks
                if i == '\n':  # Empty line
                    continue

                # Read the .psf block header
                key = cls._HEADER_DICT[i.split()[1].rstrip(':')]
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

        ret = cls(**PSF._post_process_psf(ret))
        ret.filename = filename
        return ret

    @classmethod
    def _post_process_psf(cls, psf_dict: dict) -> Dict[str, np.ndarray]:
        """Post-process the output of :meth:`PSF.read`, casting the values into appropiat objects.

        * The title block is converted into a 1D array of strings.
        * The atoms block is converted into a Pandas DataFrame.
        * All other blocks are converted into 2D arrays of integers.

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
            # Cast the atoms block into a dataframe
            if key == 'atoms':
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

            # Cast into a flattened array of indices
            else:
                ar = np.fromiter(chain.from_iterable(value), dtype=int)
                ar.shape = len(ar) // cls._SHAPE_DICT[key].shape, cls._SHAPE_DICT[key].shape
                psf_dict[key] = ar

        return psf_dict

    """########################### methods for writing .psf files. ##############################"""

    def write(self, filename: Optional[str] = None) -> None:
        """Create a protein structure file (.psf) from this :class:`.PSF` instance.

        Parameters
        ----------
        filename : str
            The path+filename of the .psf file.
            If ``None``, attempt to pull the filename from :meth:`.PSF.filename`.

        Raises
        ------
        TypeError
            Raised if the filename is specified in neither **filename** nor :meth:`.PSF.filename`.

        """
        filename = filename if filename is None else self.filename
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
        :meth:`.PSF.write`
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
        :meth:`.PSF.write`
            The main method for writing .psf files.

        """
        sections = ('bonds', 'angles', 'dihedrals', 'impropers',
                    'donors', 'acceptors', 'no_nonbonded')

        ret = ''
        for attr in sections:
            if attr in ('title', 'atoms'):
                continue

            header = self._SHAPE_DICT[attr].header
            row_len = self._SHAPE_DICT[attr].row_len
            value = getattr(self, attr)
            if value is None:
                ret += '\n\n' + header.format(0)
            else:
                ret += '\n\n' + header.format(value.shape[0])
                ret += '\n' + self.serialize_array(value, row_len)

        return ret

    @staticmethod
    def serialize_array(array: np.ndarray, items_per_row: int = 4) -> str:
        """Serialize an array into a single string; used for creating .psf files.

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

        See Also
        --------
        :meth:`.PSF.write`
            The main method for writing .psf files.

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
        """Change the charge of **atom_type** to **charge**."""
        condition = self.atoms['atom type'] == atom_type
        self.atoms.loc[condition, 'charge'] = charge

    def update_atom_type(self, atom_type_old: str, atom_type_new: str) -> None:
        """Change the atom type of a **atom_type_old** to **atom_type_new**."""
        condition = self.atoms['atom type'] == atom_type_old
        self.atoms.loc[condition, 'atom type'] = atom_type_new

    def update_bonds(self, mol: Molecule) -> None:
        """Update :attr:`PSF.bonds` with the indices of all bond-forming atoms from **mol**."""
        self.bonds = get_bonds(mol)

    def update_angles(self, mol: Molecule) -> None:
        """Update :attr:`PSF.angles` with the indices of all angle-defining atoms from **mol**."""
        self.angles = get_angles(mol)

    def update_dihedrals(self, mol: Molecule) -> None:
        """Update :attr:`PSF.dihedrals` with the indices all proper dihedral angle-defining atoms from **mol**."""  # noqa
        self.dihedrals = get_dihedrals(mol)

    def update_impropers(self, mol: Molecule) -> None:
        """Update :attr:`PSF.impropers` with the indices of all improper dihedral angle-defining atoms from **mol**."""  # noqa
        self.impropers = get_impropers(mol)
