"""
nanoCAT.ff.psf
==============

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
from codecs import iterdecode
from typing import (Dict, Optional, Any, Iterable, Iterator, Union, List)
from itertools import chain
from collections.abc import Container

import numpy as np
import pandas as pd

from scm.plams import Molecule, Atom

from CAT.frozen_settings import FrozenSettings
from nanoCAT.ff.mol_topology import (get_bonds, get_angles, get_dihedrals, get_impropers)

__all__ = ['PSF']


class PSF(Container):
    """A container for managing protein structure files.

    The :class:`PSF` class has access to three general sets of methods:

    * Methods for reading & constructing .psf files: :meth:`PSF.read` and :meth:`PSF.write`.
    * Methods for updating atom types: :meth:`PSF.update_atom_charge`
      and :meth:`PSF.update_atom_type`.
    * Methods for extracting bond, angle and dihedral-pairs from :class:`Molecule` instances:
      :meth:`PSF.generate_bonds`, :meth:`PSF.generate_angles`, :meth:`PSF.generate_dihedrals`,
      :meth:`PSF.generate_impropers` and :meth:`PSF.generate_atoms`.

    Parameters
    ----------
    filename : :math:`1` |np.ndarray|_ [|np.str_|_]
        Optional: A 1D array-like object containing a single filename.
        See also :attr:`PSF.filename`.

    title : :math:`n` |np.ndarray|_ [|np.str_|_]
        Optional: A 1D array of strings holding the title block.
        See also :attr:`PSF.title`.

    atoms : :math:`n*8` |pd.DataFrame|_
        Optional: A Pandas DataFrame holding the atoms block.
        See also :attr:`PSF.atoms`.

    bonds : :math:`n*2` |np.ndarray|_ [|np.int64|_]
        Optional: A 2D array-like object holding the indices of all atom-pairs defining bonds.
        See also :attr:`PSF.bonds`.

    angles : :math:`n*3` |np.ndarray|_ [|np.int64|_]
        Optional: A 2D array-like object holding the indices of all atom-triplets defining angles.
        See also :attr:`PSF.angles`.

    dihedrals : :math:`n*4` |np.ndarray|_ [|np.int64|_]
        Optional: A 2D array-like object holding the indices of
        all atom-quartets defining proper dihedral angles.
        See also :attr:`PSF.dihedrals`.

    impropers : :math:`n*4` |np.ndarray|_ [|np.int64|_]
        Optional: A 2D array-like object holding the indices of
        all atom-quartets defining improper dihedral angles.
        See also :attr:`PSF.impropers`.

    donors : :math:`n*1` |np.ndarray|_ [|np.int64|_]
        Optional: A 2D array-like object holding the atomic indices of all hydrogen-bond donors.
        See also :attr:`PSF.donors`.

    acceptors : :math:`n*1` |np.ndarray|_ [|np.int64|_]
        Optional: A 2D array-like object holding the atomic indices of all hydrogen-bond acceptors.
        See also :attr:`PSF.acceptors`.

    no_nonbonded : :math:`n*2` |np.ndarray|_ [|np.int64|_]
        Optional: A 2D array-like object holding the indices of all atom-pairs whose nonbonded
        interactions should be ignored.
        See also :attr:`PSF.no_nonbonded`.

    Attributes
    ----------
    filename : :math:`1` |np.ndarray|_ [|np.str_|_]
        A 1D array with a single string as filename.

    title : :math:`n` |np.ndarray|_ [|np.str_|_]
        A 1D array of strings holding the title block.

    atoms : :math:`n*8` |pd.DataFrame|_
        A Pandas DataFrame holding the atoms block.
        The DataFrame should possess the following collumn keys:

        * ``"segment name"``
        * ``"residue ID"``
        * ``"residue name"``
        * ``"atom name"``
        * ``"atom type"``
        * ``"charge"``
        * ``"mass"``
        * ``"0"``

    bonds : :math:`n*2` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-pairs defining bonds.
        Indices are expected to be 1-based.

    angles : :math:`n*3` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-triplets defining angles.
        Indices are expected to be 1-based.

    dihedrals : :math:`n*4` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-quartets defining proper dihedral angles.
        Indices are expected to be 1-based.

    impropers : :math:`n*4` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-quartets defining improper dihedral angles.
        Indices are expected to be 1-based.

    donors : :math:`n*1` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the atomic indices of all hydrogen-bond donors.
        Indices are expected to be 1-based.

    acceptors : :math:`n*1` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the atomic indices of all hydrogen-bond acceptors.
        Indices are expected to be 1-based.

    no_nonbonded : :math:`n*2` |np.ndarray|_ [|np.int64|_]
        A 2D array holding the indices of all atom-pairs whose nonbonded
        interactions should be ignored.
        Indices are expected to be 1-based.

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
        self.atoms = atoms
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
        """Return a deep copy of this instance.

        Returns
        -------
        |nanoCAT.PSF|_
            A new :class:`PSF` object constructed from this instance.

        """
        kwargs = {(k[1:] if k.startswith('_') else k): v.copy() for k, v in vars(self).items()}
        return self.from_dict(kwargs)

    def __copy__(self) -> 'PSF':
        """Return a deep copy of this instance."""
        return self.copy()

    def __deepcopy__(self, memo: Any = None) -> 'PSF':
        """Return a deep copy of this instance."""
        return self.copy()

    def __contains__(self, value: Any) -> bool:
        """Check if this instance contains **value**."""
        return value in vars(self).values()

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Construct a :class:`dict` from this instance.

        Returns
        -------
        |dict|_ [|str|_, |np.ndarray|_]
            A dictionary of arrays with keyword arguments for :meth:`PSF.__init__`.

        See also
        --------
        :meth:`PSF.from_dict`:
            Construct a :class:`PSF` instance from a dictionary.

        """
        return {(k[1:] if k.startswith('_') else k): v for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, psf_dict: Dict[str, np.ndarray]) -> 'PSF':
        """Construct a :class:`PSF` instance from **psf_dict**.

        Returns
        -------
        |nanoCAT.PSF|_
            A :class:`PSF` instance constructed from **psf_dict**.

        See also
        --------
        :meth:`PSF.as_dict`:
            Construct a dictionary from a :class:`PSF` instance.

        """
        return cls(**psf_dict)

    """###################################### Properties ########################################"""

    @property
    def filename(self) -> str: return str(self._filename[0])

    @filename.setter
    def filename(self, value: Iterable): self._set_nd_array('_filename', value, 1, str)

    @property
    def title(self) -> np.ndarray: return self._title

    @title.setter
    def title(self, value: Iterable):
        if value is not None:
            self._set_nd_array('_title', value, 1, str)
        else:
            self._title = np.array(['PSF file generated with Nano-CAT',
                                    'https://github.com/nlesc-nano/nano-CAT'])

    @property
    def atoms(self) -> pd.DataFrame: return self._atoms

    @atoms.setter
    def atoms(self, value: Iterable): self._atoms = value if value is not None else pd.DataFrame()

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
        """Assign an array-like object (**value**) to the **name** attribute as ndarray.

        Performs an inplace update of this instance.

        .. _`array-like`: https://docs.scipy.org/doc/numpy/glossary.html#term-array-like

        Parameters
        ----------
        name : str
            The name of the to-be set attribute.

        value : `array-like`_
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
    def read(cls, filename: Union[str, Iterator[Union[str, bytes]]],
             encoding: Optional[str] = None) -> 'PSF':
        """Construct :class:`PSF` instance from a protein structure file (.psf).

        Parameters
        ----------
        filename : |str|_ or `file object`_
            The path+filename or a file object of the to-be read .psf file.

        encoding : str
            Optional: Encoding used to decode the input (*e.g.* ``"utf-8"``).
            Only relevant when a file object is supplied to **filename** and
            the datastream is *not* in text mode.

        Returns
        -------
        |nanoCAT.PSF|_:
            A :class:`.PSF` instance holding the content of **filename**.

        """
        def iterate(stream: Iterator[Union[str, bytes]]) -> None:
            iterator = iter(stream) if encoding is None else iterdecode(stream, encoding)
            next(iterator)  # Skip the first line
            for i in iterator:
                # Search for psf blocks
                if i == '\n':
                    continue

                # Read the psf block header
                key = cls._HEADER_DICT[i.split()[1].rstrip(':')]
                ret[key] = value = []

                # Read the actual psf blocks
                try:
                    j = next(iterator)
                except StopIteration:
                    break

                while j != '\n':
                    value.append(j.split())
                    try:
                        j = next(iterator)
                    except StopIteration:
                        break

        ret = {}

        # filename is an actual filename
        if isinstance(filename, str):
            with open(filename, 'r') as f:
                iterate(f)
        else:  # filename is a data stream
            iterate(filename)

        # Post-process and return
        _ret = cls(**PSF._post_process_psf(ret))
        _ret.filename = filename if isinstance(filename, str) else None
        return _ret

    @classmethod
    def _post_process_psf(cls, psf_dict: dict) -> Dict[str, np.ndarray]:
        """Post-process the output of :meth:`PSF.read`, casting the values into appropiat objects.

        * The title block is converted into a 1D array of strings.
        * The atoms block is converted into a Pandas DataFrame.
        * All other blocks are converted into 2D arrays of integers.

        Parameters
        ----------
        psf_dict : |dict|_ [|str|_, |np.ndarray|_]
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

    def write(self, filename: Union[str, None, Iterator] = None,
              encoding: Optional[str] = None) -> None:
        """Create a protein structure file (.psf) from this :class:`.PSF` instance.

        .. _`file object`: https://docs.python.org/3/glossary.html#term-file-object

        Parameters
        ----------
        filename : |str|_ or `file object`_
            The path+filename or a file object of the output .psf file.
            If ``None``, attempt to pull the filename from :attr:`PSF.filename`.

        encoding : str
            Optional: Encoding used to encode the output (*e.g.* ``"utf-8"``).
            Only relevant when a file object is supplied to **filename** and
            the datastream is *not* in text mode.

        Raises
        ------
        TypeError
            Raised if the filename is specified in neither **filename** nor :attr:`PSF.filename`.

        """

        _filename = filename if filename is not None else self.filename
        if not _filename:
            raise TypeError("The 'filename' parameter is missing")

        output = self._serialize_top()
        output += self._serialize_bottom()

        # _filename is an actual filename
        if isinstance(_filename, str):
            with open(_filename, 'w') as f:
                f.write(output)

        else:  # _filename is a data stream
            if encoding is None:
                _filename.write(output)
            else:
                _filename.write(output.encode(encoding))

    def _serialize_top(self) -> str:
        """Serialize the top-most section of the to-be create .psf file.

        The following blocks are seralized:

            * :attr:`PSF.title`
            * :attr:`PSF.atoms`

        Returns
        -------
        |str|_
            A string constructed from the above-mentioned psf blocks.

        See Also
        --------
        :meth:`PSF.write`
            The main method for writing .psf files.

        """
        # Prepare the !NTITLE block
        ret = ('PSF EXT\n'
               '\n{:>10d} !NTITLE'.format(self.title.shape[0]))
        for i in self.title:
            ret += f'   REMARKS {i}'

        # Prepare the !NATOM block
        ret += '\n\n{:>10d} !NATOM\n'.format(self.atoms.shape[0])
        string = '{:>10d} {:8.8} {:<8d} {:8.8} {:8.8} {:6.6} {:>9f} {:>15f} {:>8d}\n'
        for i, j in self.atoms.iterrows():
            args = [i] + j.values.tolist()
            ret += string.format(*args)

        if len(self.atoms) == 0:
            return ret
        else:  # Clip the last newline character
            return ret[:-1]

    def _serialize_bottom(self) -> str:
        """Serialize the bottom-most section of the to-be create .psf file.

        The following blocks are seralized:

            * :attr:`PSF.bonds`
            * :attr:`PSF.angles`
            * :attr:`PSF.dihedrals`
            * :attr:`PSF.impropers`
            * :attr:`PSF.donors`
            * :attr:`PSF.acceptors`
            * :attr:`PSF.no_nonbonded`

        Returns
        -------
        |str|_
            A string constructed from the above-mentioned psf blocks.

        See Also
        --------
        :meth:`PSF.write`
            The main method for writing .psf files.

        """
        sections = ('bonds', 'angles', 'dihedrals', 'impropers',
                    'donors', 'acceptors', 'no_nonbonded')

        ret = ''
        for attr in sections:
            header = self._SHAPE_DICT[attr].header
            row_len = self._SHAPE_DICT[attr].row_len

            value = getattr(self, attr)
            item_count = len(value) if value.shape[-1] != 0 else 0
            ret += ('\n\n' + header.format(item_count) +
                    '\n' + self._serialize_array(value, row_len))
        return ret

    @staticmethod
    def _serialize_array(array: np.ndarray, items_per_row: int = 4) -> str:
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
        :meth:`PSF.write`
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
        """Change the charge of **atom_type** to **charge**.

        Parameters
        ----------
        atom_type : str
            An atom type in :attr:`self.atoms` ``["atom type"]``.

        charge : float
            The new atomic charge to-be assigned to **atom_type**.
            See :attr:`self.atoms` ``["charge"]``.

        """
        condition = self.atoms['atom type'] == atom_type
        self.atoms.loc[condition, 'charge'] = charge

    def update_atom_type(self, atom_type_old: str, atom_type_new: str) -> None:
        """Change the atom type of a **atom_type_old** to **atom_type_new**.

        Parameters
        ----------
        atom_type_old : str
            An atom type in :attr:`self.atoms` ``["atom type"]``.

        atom_type_new : float
            The new atom type to-be assigned to **atom_type**.
            See :attr:`self.atoms` ``["atom type"]``.

        """
        condition = self.atoms['atom type'] == atom_type_old
        self.atoms.loc[condition, 'atom type'] = atom_type_new

    def generate_bonds(self, mol: Molecule) -> None:
        """Update :attr:`PSF.bonds` with the indices of all bond-forming atoms from **mol**.

        Parameters
        ----------
        mol : |plams.Molecule|_
            A PLAMS Molecule.

        """
        self.bonds = get_bonds(mol)

    def generate_angles(self, mol: Molecule) -> None:
        """Update :attr:`PSF.angles` with the indices of all angle-defining atoms from **mol**.

        Parameters
        ----------
        mol : |plams.Molecule|_
            A PLAMS Molecule.

        """
        self.angles = get_angles(mol)

    def generate_dihedrals(self, mol: Molecule) -> None:
        """Update :attr:`PSF.dihedrals` with the indices of all proper dihedral angle-defining atoms from **mol**.

        Parameters
        ----------
        mol : |plams.Molecule|_
            A PLAMS Molecule.

        """  # noqa
        self.dihedrals = get_dihedrals(mol)

    def generate_impropers(self, mol: Molecule) -> None:
        """Update :attr:`PSF.impropers` with the indices of all improper dihedral angle-defining atoms from **mol**.

        Parameters
        ----------
        mol : |plams.Molecule|_
            A PLAMS Molecule.

        """  # noqa
        self.impropers = get_impropers(mol)

    def generate_atoms(self, mol: Molecule) -> None:
        """Update :attr:`PSF.atoms` with the all properties from **mol**.

        DataFrame keys in :attr:`PSF.atoms` are set based on the following values in **mol**:

        ================== ========================================================= ===================
        DataFrame column   Value                                                     Backup value
        ================== ========================================================= ===================
        ``"segment name"`` ``"MOL{:d}"``; See ``"atom type"`` and ``"residue name"``
        ``"residue ID"``   :attr:`Atom.properties` ``["pdb_info"]["ResidueNumber"]``  ``1``
        ``"residue name"`` :attr:`Atom.properties` ``["pdb_info"]["ResidueName"]``    ``"COR"``
        ``"atom name"``    :attr:`Atom.symbol`
        ``"atom type"``    :attr:`Atom.properties` ``["atom_symbol"]``                :attr:`Atom.symbol`
        ``"charge"``       :attr:`Atom.properties` ``["charge"]``                     ``0.0``
        ``"mass"``         :attr:`Atom.mass`
        ``"0"``            ``0``
        ================== ========================================================= ===================

        If a value is not available in a particular :attr:`Atom.properties` instance then
        a backup value will be set.

        Parameters
        ----------
        mol : |plams.Molecule|_
            A PLAMS Molecule.

        """  # noqa
        def get_res_id(at: Atom) -> int:
            return at.properties.pdb_info.ResidueNumber if 'ResidueNumber' in at.properties.pdb_info else 1  # noqa

        def get_res_name(at: Atom) -> str:
            return at.properties.pdb_info.ResidueName if 'ResidueName' in at.properties.pdb_info else 'COR'  # noqa

        def get_at_type(at: Atom) -> str:
            return at.properties.atom_type if 'atom_type' in at.properties else at.symbol

        def get_charge(at: Atom) -> float:
            return float(at.properties.charge) if 'charge' in at.properties else 0.0

        index = pd.RangeIndex(1, 1 + len(mol))
        self.atoms = df = pd.DataFrame(index=index, dtype=str)

        df['segment name'] = None
        df['residue ID'] = [get_res_id(at) for at in mol]
        df['residue name'] = [get_res_name(at) for at in mol]
        df['atom name'] = [at.symbol for at in mol]
        df['atom type'] = [get_at_type(at) for at in mol]
        df['charge'] = [get_charge(at) for at in mol]
        df['mass'] = [at.mass for at in mol]
        df['0'] = 0

        df['segment name'] = self._construct_segment_name()

    def _construct_segment_name(self) -> List[str]:
        """Generate a list for the :attr:`PSF.atoms` ``["segment name"]`` column."""
        ret = []
        type_dict = {}

        for at_type, res_name in zip(self.atoms['atom type'], self.atoms['residue name']):
            if res_name == 'LIG':
                at_type = 'LIG'

            try:
                value = type_dict[at_type]
            except KeyError:
                type_dict[at_type] = value = 'MOL{:d}'.format(1 + len(type_dict))

            ret.append(value)

        return ret
