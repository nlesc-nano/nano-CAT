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

from typing import Tuple, Iterable, Any, Type

from scm.plams import Molecule, Settings
from scm.plams.core.basejob import Job

from CAT.jobs import _get_results_error
from CAT.logger import logger
from CAT.workflows import WorkFlow
from CAT.settings_dataframe import SettingsDataFrame
from FOX import PRMContainer

from .match_job import MatchJob

__all__ = ['init_ff_assignment', 'run_match_job']


def init_ff_assignment(ligand_df: SettingsDataFrame,
                       forcefield: str = 'top_all36_cgenff_new') -> None:
    """Initialize the forcefield assignment procedure using MATCH_.

    .. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

    Parameters
    ----------
    ligand_df : |CAT.SettingsDataFrame|_
        A DataFrame of ligands.

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

    See Also
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
    workflow = WorkFlow.from_template(ligand_df, name='forcefield')
    workflow.jobs = (MatchJob, )
    workflow.settings = (Settings({'input': {'forcefield': forcefield}}), )
    workflow(start_ff_assignment, ligand_df, columns=[])


def start_ff_assignment(mol_list: Iterable[Molecule], jobs: Tuple[Type[Job], ...],
                        settings: Tuple[Settings, ...], **kwargs: Any) -> None:
    """Start the forcefield assignment."""
    job = jobs[0]
    s = settings[0]
    for mol in mol_list:
        run_match_job(mol, s, job)


def run_match_job(mol: Molecule, s: Settings,
                  job_type: Type[Job] = MatchJob,
                  action: str = 'warn') -> None:
    """Assign atom types and charges to **mol** based on the results of MATCH_.

    Performs an inplace update of :attr:`Atom.properties` ``["symbol"]``,
    :attr:`Atom.properties` ``["charge"]`` and :attr:`Molecule.properties` ``["prm"]``.

    .. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

    Parameters
    ----------
    mol : |plams.Molecule|
        A PLAMS molecule.

    s : |plams.Settings|
        Job settings for the to-be constructed :class:`.MatchJob` instance.

    job_type : :class:`type` [|plams.Job|]
        The type of Job.

    action : :class:`str`
        The to-be undertaken action when the Job crashes.
        Accepted values are ``"raise"``, ``"warn"`` and ``"ignore"``.

    See Also
    --------
    :class:`.MatchJob`
        A :class:`Job` subclass for interfacing with MATCH_: Multipurpose Atom-Typer for CHARMM.

    """  # noqa
    job = job_type(molecule=mol, settings=s, name='ff_assignment')

    # Run the job
    try:
        results = job.run()
        if job.status != 'successful':
            raise _get_results_error(results)

        symbol_list = results.get_atom_types()
        charge_list = results.get_atom_charges()
        logger.info(f'{job.__class__.__name__}: {mol.properties.name} parameter assignment '
                    f'({job.name}) is successful')
    except Exception as ex:
        if action == 'raise':
            raise ex
        elif 'action' == 'warm':
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
    prm = PRMContainer.read(filename)
    prm.write(filename)
