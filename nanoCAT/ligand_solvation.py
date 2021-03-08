"""
nanoCAT.ligand_solvation
========================

A module designed for calculating solvation energies.

Index
-----
.. currentmodule:: nanoCAT.ligand_solvation
.. autosummary::
    init_solv
    start_crs_jobs
    update_columns
    get_solvent_list
    get_job_settings
    _ligand_to_db
    get_surface_charge
    get_solv
    get_surface_charge_adf
    get_coskf

API
---
.. autofunction:: init_asa
.. autofunction:: start_crs_jobs
.. autofunction:: update_columns
.. autofunction:: get_solvent_list
.. autofunction:: get_job_settings
.. autofunction:: _ligand_to_db
.. autofunction:: get_surface_charge
.. autofunction:: get_solv
.. autofunction:: get_surface_charge_adf
.. autofunction:: get_coskf

"""

import os
import math
import logging
from itertools import product
from os.path import join
from typing import (
    Optional, Sequence, Collection, Tuple, List, Iterable, Any, Type, Generator
)

import numpy as np
from scipy import constants

from scm.plams import (
    Settings, Molecule, Results, CRSJob, CRSResults, JobRunner, ADFJob, Units, add_Hs
)
from scm.plams.core.basejob import Job

import CAT
if CAT.version_info < (0, 9, 10):
    raise RuntimeError("Nano-CAT requires CAT 0.9.10 or later;"
                       f"observed version: {CAT.__version__}")

from CAT.jobs import job_single_point, _get_name  # noqa: F401
from CAT.utils import get_template
from CAT.logger import logger
from CAT.workflows import WorkFlow, MOL, JOB_SETTINGS_CRS
from CAT.settings_dataframe import SettingsDataFrame

__all__ = ['init_solv']


def init_solv(ligand_df: SettingsDataFrame,
              solvent_list: Optional[Sequence[str]] = None) -> None:
    """Initialize the ligand solvation energy calculation.

    Performs an inplace update of **ligand_df**, creating 2 sets of columns (*E_solv* & *gamma*)
    to hold all solvation energies and activity coefficients, respectively.

    Parameters
    ----------
    ligand_df : |CAT.SettingsDataFrame|_
        A dataframe of ligands.

    solvent_list : |list|_ [|str|_]
        Optional: A list of paths to the .t21 or .coskf files of solvents.
        If ``None``, use the default .coskf files distributed with CAT (see :mod:`CAT.data.coskf`).

    """
    workflow = WorkFlow.from_template(ligand_df, name='crs')

    # Create column slices
    solvent_list = get_solvent_list(solvent_list)
    columns = get_solvent_columns(solvent_list)
    columns += [
        ('LogP', 'Solute'),
        ('pKa', 'E_solute_acid'),
        ('pKa', 'E_solute_base'),
        ('pKa', 'E_solvent_base'),
        ('pKa', 'E_solvent_acid'),
        ('pKa', 'Solute'),
    ]

    # Create new import and export columns
    import_columns = {k: np.nan for k in columns}
    import_columns.update(workflow.import_columns)
    export_columns = columns + list(workflow.import_columns)

    # Create index slices and run the workflow
    idx = workflow.from_db(ligand_df, columns=import_columns)
    workflow(start_crs_jobs, ligand_df, index=idx, columns=columns, solvent_list=solvent_list)

    # Export results back to the database
    job_recipe = workflow.get_recipe()
    ligand_df[JOB_SETTINGS_CRS] = workflow.pop_job_settings(ligand_df[MOL])
    workflow.to_db(ligand_df, index=idx, columns=export_columns, job_recipe=job_recipe)


def start_crs_jobs(mol_list: Iterable[Molecule],
                   jobs: Tuple[Type[Job], ...], settings: Tuple[Settings, ...],
                   solvent_list: Sequence[str] = (), **kwargs: Any) -> List[List[float]]:
    """Loop over all molecules in **mol_list** and perform COSMO-RS calculations."""
    j1, j2 = jobs
    s1, s2 = settings

    water = join(CAT.__path__[0], 'data', 'coskf', 'Water.coskf')
    hydronium = join(CAT.__path__[0], 'data', 'coskf', 'misc', 'Hydronium.coskf')

    # Start the main loop
    ret = []
    for mol in mol_list:
        mol.round_coords()
        mol.properties.job_path = []

        # Calculate the COSMO surface
        mol_conj = _protonate_mol(mol)
        coskf = get_surface_charge(mol, job=j1, s=s1)
        coskf_conj = get_surface_charge(mol_conj, job=j1, s=s1)

        # Perform the actual COSMO-RS calculation
        lst = get_solv(mol, solvent_list, coskf, job=j2, s=s2)
        lst += get_pka(mol, coskf, coskf_conj, water, hydronium, job=j2, s=s2)
        ret.append(lst)
    return ret


def get_solvent_columns(solvent_list: Iterable[str]) -> List[Tuple[str, str]]:
    """Create a list of column names from an iterable containing .coskf names.

    Parameters
    ----------
    solvent_list : :data:`Iterable<typing.Iterable>` [:class:`str`]
        An iterable of strings representing solvent .coskf files.

    Returns
    -------
    :class:`list` [:class:`tuple` [:class:`str`, :class:`str`]]
        A list of 2-tuples twice as long as **solvent_list**.
        The first element of each tuple is either ``"E_solv"`` or ``"gamma"``;
        the second element is taken from **solvent_list**.

    """
    # Use filenames without extensions are absolute paths
    clm_tups = [os.path.basename(i).rsplit('.', maxsplit=1)[0] for i in solvent_list]
    return list(product(('E_solv', 'gamma'), clm_tups))


def get_solvent_list(solvent_list: Optional[Sequence[str]] = None) -> Sequence[str]:
    """Construct a sorted list of solvents; pull them from ``CAT.data.coskf`` if ``None``."""
    if solvent_list is None:
        base = join(CAT.__path__[0], 'data', 'coskf')
        solvent_list = [join(base, solv) for solv in os.listdir(base) if solv.endswith('coskf')]

    try:
        solvent_list.sort()
    except AttributeError:  # It's not a list but a generic iterable
        return sorted(solvent_list)
    else:
        return solvent_list


def _protonate_mol(mol: Molecule) -> Molecule:
    mol_cp = mol.copy()

    i = mol.index(mol.properties.dummies)
    anchor = mol_cp[i]
    if anchor.properties.charge != -1:
        raise NotImplementedError("Non-anionic anchors are not supported")

    anchor.properties.charge = 0
    return add_Hs(mol_cp, forcefield="uff")


def get_surface_charge(mol: Molecule, job: Type[Job], s: Settings) -> Optional[str]:
    """Construct the COSMO surface of the **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS Molecule.

    job : |Callable|_
        A type Callable of a class derived from :class:`Job`, e.g. :class:`AMSJob`
        or :class:`Cp2kJob`.

    s : |plams.Settings|_
        The settings for **job**.

    Returns
    -------
    |plams.Settings|_
        Optional: The path+filename of a file containing COSMO surface charges.

    """
    s = Settings(s)
    # Special procedure for ADF jobs
    # Use the gas-phase electronic structure as a fragment for the COSMO single point
    if job is ADFJob:
        s = get_surface_charge_adf(mol, job, s)

    s.runscript.post = '$ADFBIN/cosmo2kf "mopac.cos" "mopac.coskf"'
    results = mol.job_single_point(job, s, ret_results=True)
    return get_coskf(results)


def _crs_run(job: CRSJob, name: str, calc_type: str = 'activity coefficient') -> CRSResults:
    """Call the :meth:`CRSJob.run` on **job**, returning a :class:`CRSResults` instance."""
    _name = _get_name(job.name)
    logger.info(f'{job.__class__.__name__}: {name} {calc_type} calculation '
                f'({_name}) has started')
    return job.run(jobrunner=JobRunner(parallel=True))


def _iter_coskf(
    acid: str,
    base: str,
    solvent: str,
    solvent_conj: str
) -> Generator[Tuple[str, str], None, None]:
    yield "acid", acid
    yield "base", base
    yield "solvent", solvent
    yield "solvent_conj", solvent_conj


def get_pka(mol: Molecule, coskf_mol: Optional[str], coskf_mol_conj: Optional[str],
            water: str, hydronium: str,
            job: Type[Job], s: Settings) -> List[float]:
    if coskf_mol is None:
        return 5 * [np.nan]
    elif coskf_mol_conj is None:
        return 5 * [np.nan]

    s = Settings(s)
    s.input.compound[1]._h = water
    s.ignore_molecule = True

    s_dict = {}
    for name, coskf in _iter_coskf(coskf_mol_conj, coskf_mol, water, hydronium):
        _s = s.copy()
        _s.name = name
        _s.input.compound[0]._h = coskf
        s_dict[name] = _s

    # Run the job
    mol_name = mol.properties.name
    job_list = [CRSJob(settings=s, name=name) for name, s in s_dict.items()]
    results_list = [_crs_run(job, mol_name) for job in job_list]

    # Extract solvation energies and activity coefficients
    E_solv = {}
    for name, results in zip(("acid", "base", "solvent", "solvent_conj"), results_list):
        results.wait()
        try:
            E_solv[name] = _E = results.get_energy()
            assert _E is not None
            logger.info(f'{results.job.__class__.__name__}: {mol_name} pKa '
                        f'calculation ({results.job.name}) is successful')
        except Exception:
            logger.error(f'{results.job.__class__.__name__}: {mol_name} pKa '
                         f'calculation ({results.job.name}) has failed')
            E_solv[name] = np.nan

    try:
        mol.properties.job_path += [join(job.path, job.name + '.in') for job in job_list]
    except IndexError:  # The 'job_path' key is not available
        mol.properties.job_path = [join(job.path, job.name + '.in') for job in job_list]

    ret = [E_solv[k] for k in ("acid", "base", "solvent", "solvent_conj")]
    ret.append(_get_pka(**E_solv))
    return ret


#: The gas constant in kcal/mol
R: float = Units.convert(constants.R, 'kj/mol', 'kcal/mol') / 1000


def _get_pka(acid: float, base: float, solvent: float, solvent_conj: float,
             T: float = 298.15) -> float:
    """Calculate the pKa at the temperature **T**.

    See Also
    --------
    `Molecular Physics 108, 229 (2010) <https://doi.org/10.1080/00268970903313667>`_
        F. Eckert, M. Diedenhofen, and A. Klamt, Towards a first principles prediction of
        pKa : COSMO-RS and the cluster-continuum approach.

    """
    # See Eq 5.1.5 in
    # https://www.scm.com/doc/Tutorials/COSMO-RS/pKa_values.html#empirical-pka-calculation-method
    fit_a = 1
    fit_b = -1.74  # Correction for the standard state of liquid water, which is 55 mol/L

    delta_G = (base - acid) - (solvent_conj - solvent)
    return fit_a * delta_G / (R * T * math.log(10)) + fit_b


def get_solv(mol: Molecule, solvent_list: Iterable[str],
             coskf: Optional[str], job: Type[Job], s: Settings
             ) -> List[float]:
    """Calculate the solvation energy of *mol* in various *solvents*.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS Molecule.

    solvent_list : |List|_ [|str|_]
        A list of solvent molecules (*i.e.* .coskf files).

    coskf : str, optional
        The path+filename of the .coskf file of **mol**.

    job : |Callable|_
        A type Callable of a class derived from :class:`Job`, e.g. :class:`AMSJob`
        or :class:`Cp2kJob`.

    s : |plams.Settings|_
        The settings for **job**.

    Returns
    -------
    |list|_ [|float|_] & |list|_ [|float|_]
        A list of solvation energies and gammas.

    """
    # Return 3x np.nan if no coskf is None (i.e. the COSMO-surface construction failed)
    if coskf is None:
        i = 1 + 2 * len(solvent_list)
        return i * [np.nan]

    # Prepare a list of job settings
    s = Settings(s)
    s.input.compound[0]._h = coskf
    s.ignore_molecule = True
    s_list = []
    for solv in solvent_list:
        _s = s.copy()
        _s.name = solv.rsplit('.', 1)[0].rsplit(os.sep, 1)[-1]
        _s.input.compound[1]._h = solv
        s_list.append(_s)

    # Run the job
    mol_name = mol.properties.name
    job_list = [CRSJob(settings=s, name=s.name) for s in s_list]
    results_list = [_crs_run(job, mol_name, calc_type="pKa") for job in job_list]

    # Extract solvation energies and activity coefficients
    E_solv = []
    Gamma = []
    for results in results_list:
        results.wait()
        try:
            E_solv.append(results.get_energy())
            Gamma.append(results.get_activity_coefficient())
            logger.info(f'{results.job.__class__.__name__}: {mol_name} activity coefficient '
                        f'calculation ({results.job.name}) is successful')
        except Exception:
            logger.error(f'{results.job.__class__.__name__}: {mol_name} activity coefficient '
                         f'calculation ({results.job.name}) has failed')
            E_solv.append(np.nan)
            Gamma.append(np.nan)

    try:
        mol.properties.job_path += [join(job.path, job.name + '.in') for job in job_list]
    except IndexError:  # The 'job_path' key is not available
        mol.properties.job_path = [join(job.path, job.name + '.in') for job in job_list]

    logp = _get_logp(s_list[0], name=mol_name, logger=logger)
    return E_solv + Gamma + [logp]


def _get_logp(s: Settings, name: str, logger: logging.Logger) -> float:
    logp_s = s.copy()
    logp_s.update(get_template('qd.yaml')['COSMO-RS logp'])
    for v in logp_s.input.compound:
        v._h = v._h.format(os.environ["ADFRESOURCES"])

    logp_job = CRSJob(settings=logp_s, name='LogP')
    results = _crs_run(logp_job, name)
    try:
        logp = results.readkf('LOGP', 'logp')[0]
        logger.info(f'{results.job.__class__.__name__}: {name} LogP '
                    f'calculation ({results.job.name}) is successful')
    except Exception:
        logger.error(f'{results.job.__class__.__name__}: {name} LogP '
                     f'calculation ({results.job.name}) has failed')
        logp = np.nan
    return logp


def get_surface_charge_adf(mol: Molecule, job: Type[Job], s: Settings) -> Settings:
    """Perform a gas-phase ADF single point and return settings for a COSMO-ADF single point.

    The previous gas-phase calculation as moleculair fragment.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS Molecule.

    job : |Callable|_
        A type Callable of a class derived from :class:`Job`, e.g. :class:`AMSJob`
        or :class:`Cp2kJob`.

    s : |plams.Settings|_
        The settings for **job**.

    Returns
    -------
    |plams.Settings|_
        A new Settings intance, constructed from **s**, suitable for DFT COSMO-RS calculations.

    """
    s.input.allpoints = ''
    results = mol.job_single_point(job, s, ret_results=True)
    coskf = get_coskf(results)

    for at in mol:
        at.properties.adf.fragment = 'gas'
    s.update(get_template('qd.yaml')['COSMO-ADF'])
    s.input.fragments.gas = coskf

    return s


def get_coskf(results: Results, extensions: Collection[str] = ('.coskf', '.t21')) -> Optional[str]:
    """Return the file in **results** containing the COSMO surface.

    Parameters
    ----------
    results : |plams.Results|_
        A Results instance.

    extensions : |list|_ [|str|_]
        Valid filetypes which can contain COSMO surfaces.

    Returns
    -------
        Optional: The path+filename of a file containing COSMO surface charges.

    """
    for file in results.files:
        for ext in extensions:
            if ext in file:
                return results[file]
    logger.error(f'Failed to retrieve COSMO surface charges of {results.job.name}')
    return None
