"""
nanoCAT.recipes.cdft_utils
==========================

Recipes for running conceptual dft calculations.

Index
-----
.. currentmodule:: nanoCAT.recipes.cdft
.. autosummary::
    run_jobs
    get_global_descriptors
    cdft

API
---
.. autofunction:: run_jobs
.. autofunction:: get_global_descriptors
.. autodata:: cdft
    :annotation: : qmflows.Settings

"""
import inspect
from os import PathLike
from os.path import join
from typing import Mapping, Any, Union, Optional, TypeVar, Dict, MutableMapping, Iterable, FrozenSet

import yaml
import numpy as np
import pandas as pd

from scm.plams import Molecule, ADFResults, Results, config
from qmflows import adf, Settings, templates as _templates
from qmflows.utils import InitRestart
from qmflows.packages import registry, Package, Result
from qmflows.packages.SCM import ADF_Result
from noodles.run.threading.sqlite3 import run_parallel

from CAT.utils import SetAttr

__all__ = ['get_global_descriptors', 'run_jobs', 'cdft']

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

#: A QMFlows-style template for conceptual DFT calculations.
cdft = Settings()
cdft.specific.adf = _templates.singlepoint.specific.adf.copy()
cdft += Settings(yaml.safe_load("""
specific:
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
"""))


#: A :class:`frozenset` with all parameters of the
#: :func:`~noodles.run.threading.sqlite3.run_parallel` function. `
_RUN_PARALLEL_KEYS: FrozenSet[str] = frozenset(
    inspect.signature(run_parallel).parameters.keys()
)


def _split_dict(dct: MutableMapping[_KT, _VT], key_set: Iterable[_KT]) -> Dict[_KT, _VT]:
    """Create a new dictionary from popping all keys in **dct** which are specified in **keys**."""
    # Get the intersection of **keys** and the keys in **dct**
    try:
        keys = dct.keys() & key_set  # type: ignore
    except TypeError:
        keys = dct.keys() & set(key_set)
    return {k: dct.pop(k) for k in keys}


def run_jobs(mol: Molecule, *settings: Mapping,
             job_type: Package = adf,
             job_name: Optional[str] = None,
             path: Union[None, str, PathLike] = None,
             folder: Union[None, str, PathLike] = None,
             **kwargs: Any) -> Result:
    r"""Run multiple jobs in succession.

    Examples
    --------
    .. code:: python

        >>> from scm.plams import Molecule
        >>> from qmflows import Settings
        >>> from qmflows.templates import geometry
        >>> from qmflows.utils import InitRestart
        >>> from qmflows.packages.SCM import ADF_Result

        >>> from CAT.recipes import run_jobs, cdft

        >>> mol = Molecule(...)

        >>> settings_opt = Settings(...)
        >>> settings_opt += geometry
        >>> settings_cdft = Settings(...)
        >>> settings_cdft += cdft

        >>> result: ADF_Result = run_jobs(mol, settings_opt, settings_cdft)


    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule`
        The input molecule.

    \*settings : :class:`~collections.abc.Mapping`
        One or more input settings.
        A single job will be run for each provided settings object.
        The output molecule of each job will be passed on to the next one.

    job_type : :class:`~qmflows.packages.packages.Package`
        A QMFlows package instance.

    job_name : :class:`str`, optional
        The name basename of the job.
        The name will be append with :code:`".{i}"`, where ``{i}`` is the number of the job.

    path : :class:`str` or :class:`~os.PathLike`, optional
        The path to the working directory.

    folder : :class:`str` or :class:`~os.PathLike`, optional
        The name of the working directory.

    **kwargs : :data:`~typing.Any`
        Further keyword arguments for **job_type** and the noodles job runner.

    Returns
    -------
    :class:`~qmflows.packages.packages.Result`
        A QMFlows Result object as constructed by the last calculation.
        The exact type depends on the passed **job_type**.

    See Also
    --------
    :func:`noodles.run.threading.sqlite3.run_parallel`
        Run a workflow in parallel threads, storing results in a Sqlite3 database.

    """
    # The job name
    name = job_name if job_name is not None else f'{job_type.__class__.__name__.lower()}'

    # Collect keyword arguments for run_parallel()
    run_kwargs = {'n_threads': 1, 'echo_log': False, 'always_cache': True}
    run_kwargs.update(_split_dict(kwargs, _RUN_PARALLEL_KEYS))
    if 'n_processes' in run_kwargs:
        run_kwargs['n_threads'] = run_kwargs.pop('n_processes')

    # Construct the jobs
    job = Settings({'geometry': mol})
    for i, _s in enumerate(settings):
        s = Settings(_s) if not isinstance(_s, Settings) else _s
        job = job_type(settings=s, mol=job.geometry, job_name=f'{name}.{i}', **kwargs)

    # Run the jobs and return
    with InitRestart(path=path, folder=folder):
        db_file = join(config.default_jobmanager.workdir, 'cache.db')
        with SetAttr(config.log, 'stdout', 0):
            return run_parallel(job, registry=registry, db_file=db_file, **run_kwargs)


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
                value = float(_value.rstrip('(eV)'))
            ret[key] = value

    # Fix the names of "mu+" and "mu-"
    ret['Electronic chemical potential (mu+)'] = ret.pop('mu+', np.nan)
    ret['Electronic chemical potential (mu-)'] = ret.pop('mu-', np.nan)
    return pd.Series(ret, name='global descriptors')
