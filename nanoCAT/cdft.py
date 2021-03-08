from math import nan
from typing import Iterable, Any, Type, List, Union

import yaml
import pandas as pd

from qmflows import templates as _templates
from qmflows.packages.SCM import ADF_Result
from scm.plams import Molecule, Settings, ADFJob, ADFResults, Units, Results
from scm.plams.core.basejob import Job
from CAT.workflows import WorkFlow, JOB_SETTINGS_CDFT, MOL
from CAT.jobs import job_single_point  # noqa: F401
from CAT.settings_dataframe import SettingsDataFrame

__all__ = ['init_cdft', 'get_global_descriptors', 'cdft']

_CDFT: str = """specific:
    adf:
        symmetry: nosym
        conceptualdft:
            enabled: yes
            analysislevel: extended
            electronegativity: yes
            domains:
                enabled: yes
        qtaim:
            enabled: yes
            analysislevel: extended
            energy: yes
        basis:
            core: none
            type: DZP
        xc:
            libxc:
                CAM-B3LYP
        numericalquality: good
"""

#: A QMFlows-style template for conceptual DFT calculations.
cdft = Settings(yaml.safe_load(_CDFT))
cdft.specific.adf += _templates.singlepoint.specific.adf.copy()


def init_cdft(ligand_df: SettingsDataFrame) -> None:
    r"""Initialize the ligand conceptual dft (CDFT) workflow.

    Parameters
    ----------
    ligand_df : |CAT.SettingsDataFrame|
        A DataFrame of ligands.

    """
    workflow = WorkFlow.from_template(ligand_df, name='cdft')

    # Import from the database and start the calculation
    idx = workflow.from_db(ligand_df)
    workflow(start_crs_jobs, ligand_df, index=idx)

    # Sets a nested list with the filenames of .in files
    # This cannot be done with loc is it will try to expand the list into a 2D array
    ligand_df[JOB_SETTINGS_CDFT] = workflow.pop_job_settings(ligand_df[MOL])

    # Export to the database
    job_recipe = workflow.get_recipe()
    workflow.to_db(ligand_df, index=idx, job_recipe=job_recipe)


def start_crs_jobs(mol_list: Iterable[Molecule],
                   jobs: Iterable[Type[Job]], settings: Iterable[Settings],
                   **kwargs: Any) -> List[pd.Series]:
    # Parse the job type
    job, *_ = jobs
    if job is not ADFJob:
        raise NotImplementedError(f"job: {job.__class__.__name__} = {job!r}")

    # Parse and update the input settings
    _s, *_ = settings
    s = Settings(_s)
    s.input += cdft.specific.adf

    ret = []
    for mol in mol_list:
        ret.append(run_cdft_job(mol, job, s))
    return ret


_BACKUP = pd.Series({
    'Electronic chemical potential (mu)': nan,
    'Electronegativity (chi=-mu)': nan,
    'Hardness (eta)': nan,
    'Softness (S)': nan,
    'Hyperhardness (gamma)': nan,
    'Electrophilicity index (w=omega)': nan,
    'Dissocation energy (nucleofuge)': nan,
    'Dissociation energy (electrofuge)': nan,
    'Electrodonating power (w-)': nan,
    'Electroaccepting power(w+)': nan,
    'Net Electrophilicity': nan,
    'Global Dual Descriptor Deltaf+': nan,
    'Global Dual Descriptor Deltaf-': nan,
    'Electronic chemical potential (mu+)': nan,
    'Electronic chemical potential (mu-)': nan
})
_BACKUP.index = pd.MultiIndex.from_product(
    [['cdft'], _BACKUP.index], names=['index', 'sub index']
)


def run_cdft_job(mol: Molecule, job: Type[ADFJob], s: Settings) -> pd.Series:
    """Run a conceptual DFT job and extract & return all global descriptors."""
    results = mol.job_single_point(job, s.copy(), name='CDFT',
                                   ret_results=True, read_template=False)

    if results.job.status in {'crashed', 'failed'}:
        return _BACKUP

    ret = get_global_descriptors(results)
    ret.index = pd.MultiIndex.from_product(
        [['cdft'], ret.index], names=['index', 'sub index']
    )
    return ret


def get_global_descriptors(results: Union[ADFResults, ADF_Result]) -> pd.Series:
    """Extract a dictionary with all ADF conceptual DFT global descriptors from **results**.

    Examples
    --------
    .. code:: python

        >>> import pandas as pd
        >>> from scm.plams import ADFResults
        >>> from CAT.recipes import get_global_descriptors

        >>> results = ADFResults(...)

        >>> series: pd.Series = get_global_descriptors(results)
        >>> print(dct)
        Electronic chemical potential (mu)     -0.113
        Electronegativity (chi=-mu)             0.113
        Hardness (eta)                          0.090
        Softness (S)                           11.154
        Hyperhardness (gamma)                  -0.161
        Electrophilicity index (w=omega)        0.071
        Dissocation energy (nucleofuge)         0.084
        Dissociation energy (electrofuge)       6.243
        Electrodonating power (w-)              0.205
        Electroaccepting power(w+)              0.092
        Net Electrophilicity                    0.297
        Global Dual Descriptor Deltaf+          0.297
        Global Dual Descriptor Deltaf-         -0.297
        Electronic chemical potential (mu+)    -0.068
        Electronic chemical potential (mu-)    -0.158
        Name: global descriptors, dtype: float64


    Parameters
    ----------
    results : :class:`plams.ADFResults` or :class:`qmflows.ADF_Result`
        A PLAMS Results or QMFlows Result instance of an ADF calculation.

    Returns
    -------
    :class:`pandas.Series`
        A Series with all ADF global decsriptors as extracted from **results**.

    """
    if not isinstance(results, Results):
        results = results.results
    file = results['$JN.out']

    with open(file) as f:
        # Identify the GLOBAL DESCRIPTORS block
        for item in f:
            if item == ' GLOBAL DESCRIPTORS\n':
                next(f)
                next(f)
                break
        else:
            raise ValueError(f"Failed to identify the 'GLOBAL DESCRIPTORS' block in {file!r}")

        # Extract the descriptors
        ret = {}
        for item in f:
            item = item.rstrip('\n')
            if not item:
                break

            _key, _value = item.rsplit('=', maxsplit=1)
            key = _key.strip()
            try:
                value = float(_value)
            except ValueError:
                value = float(_value.rstrip('(eV)')) * Units.conversion_ratio('ev', 'au')
            ret[key] = value

    # Fix the names of "mu+" and "mu-"
    ret['Electronic chemical potential (mu+)'] = ret.pop('mu+', nan)
    ret['Electronic chemical potential (mu-)'] = ret.pop('mu-', nan)
    return pd.Series(ret, name='global descriptors')
