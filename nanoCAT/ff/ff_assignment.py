"""
nanoCAT.ff.ff_assignment
========================

A worklflow for assigning CHARMM forcefield parameters to molecules.

Index
-----
.. currentmodule:: nanoCAT.ff.ff_assignment
.. autosummary::
    init_ff_assignment
    run_match_job

API
---
.. autofunction:: init_ff_assignment
.. autofunction:: run_match_job

"""

from typing import Tuple, NoReturn

from scm.plams import Molecule, Settings, ResultsError, finish, Results

from CAT.logger import logger
from CAT.utils import restart_init
from CAT.settings_dataframe import SettingsDataFrame

from .match_job import MatchJob
from .prm import PRM

__all__ = ['init_ff_assignment', 'run_match_job']

# Aliases for pd.MultiIndex columns
MOL: Tuple[str, str] = ('mol', '')


def init_ff_assignment(df: SettingsDataFrame, mol_type: str,
                       forcefield: str = 'top_all36_cgenff') -> None:
    """Initialize the forcefield assignment procedure using MATCH_.

    .. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

    Parameters
    ----------
    df : |CAT.SettingsDataFrame|_
        A DataFrame of molecules.

    mol_type : str
        The type of molecules in **df**.
        Accepted values are ``"core"``, ``"ligand"`` and ``"qd"``.

    forcefield : str
        The type of to-be assigned forcefield atom types.
        See the ``-Forcefield`` paramater in the MATCH_ user guide for more details.
        By default the allowed values are:

        * ``"top_all22_prot"``
        * ``"top_all27_na"``
        * ``"top_all35_carb"``
        * ``"top_all35_ether"``
        * ``"top_all36_cgenff"``
        * ``"top_all36_cgenff_new"``
        * ``"top_all36_lipid"``

    See also
    --------
    MATCH publication:
        `MATCH: An atom-typing toolset for molecular mechanics force fields,
        J.D. Yesselman, D.J. Price, J.L. Knight and C.L. Brooks III,
        J. Comput. Chem., 2011 <https://doi.org/10.1002/jcc.21963>`_

    :func:`.run_match_job`:
        Assign atom types and charges to **mol** based on the results of MATCH_.

    :class:`.MatchJob`:
        A :class:`Job` subclass for interfacing with MATCH_: Multipurpose Atom-Typer for CHARMM.

    """  # noqa
    path = df.settings.optional[mol_type].dirname

    s = Settings()
    s.input.forcefield = forcefield

    logger.info(f'Starting {mol_type} forcefield parameter assignment ({forcefield})')
    restart_init(path, 'MATCH')
    for mol in df[MOL]:
        run_match_job(mol, s)
    finish()
    logger.info(f'Finishing {mol_type} forcefield parameter assignment ({forcefield})\n')


def run_match_job(mol: Molecule, s: Settings) -> None:
    """Assign atom types and charges to **mol** based on the results of MATCH_.

    Performs an inplace update of :attr:`Atom.properties` ``["symbol"]``,
    :attr:`Atom.properties` ``["charge"]`` and :attr:`Molecule.properties` ``["prm"]``.

    .. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    s : |plams.Settings|_
        Job settings for the to-be constructed :class:`.MatchJob` instance.

    See also
    --------
    :class:`.MatchJob`
        A :class:`Job` subclass for interfacing with MATCH_: Multipurpose Atom-Typer for CHARMM.

    """  # noqa
    job = MatchJob(molecule=mol, settings=s)

    # Run the job
    try:
        results = job.run()
        if job.status != 'successful':
            _raise_results_error(results)

        symbol_list = results.get_atom_types()
        charge_list = results.get_atom_charges()
        logger.info(f'{job.__class__.__name__}: {mol.properties.name} parameter assignment '
                    f'({job.name}) is successful')
    except Exception as ex:
        logger.info(f'{job.__class__.__name__}: {mol.properties.name} parameter assignment '
                    f'({job.name}) has failed')
        logger.debug(f'{ex.__class__.__name__}: {ex}', exc_info=True)
        return None

    # Update properties with new symbols, charges and the consntructed parameter (.prm) file
    mol.properties.prm = prm = results['$JN.prm']
    for at, symbol, charge in zip(mol, symbol_list, charge_list):
        at.properties.symbol = symbol
        at.properties.charge_float = charge

    post_proccess_prm(prm)
    return None


def post_proccess_prm(filename: str) -> None:
    """Move the ``"IMPROPERS"`` block to the bottom of the .prm file so CP2K doesnt complain."""
    prm = PRM.read(filename)
    prm.write(filename)


def _raise_results_error(results: Results) -> NoReturn:
    """Raise a :exc:`ResultsError` with the content of ``results['$JN.err']`` as error mesage."""
    filename = results['$JN.err']
    with open(filename, 'r') as f:
        raise ResultsError(f.read().rstrip('\n'))
