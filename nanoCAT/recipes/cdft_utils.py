"""Recipes for running conceptual dft calculations.

Index
-----
.. currentmodule:: nanoCAT.recipes
.. autosummary::
    run_jobs
    get_global_descriptors
    cdft

API
---
.. autofunction:: run_jobs
.. autofunction:: get_global_descriptors

.. data:: cdft
    :annotation: = qmflows.Settings(...)

    A QMFlows-style template for conceptual DFT calculations.

    .. code-block:: yaml

{cdft}

"""
import inspect
import textwrap
from os import PathLike
from os.path import join
from typing import Mapping, Any, Union, Optional, TypeVar, FrozenSet

from scm.plams import Molecule, config
from qmflows import adf, Settings
from qmflows.utils import InitRestart
from qmflows.packages import registry, Package, Result
from noodles.run.threading.sqlite3 import run_parallel
from nanoutils import SetAttr, split_dict

from nanoCAT.cdft import _CDFT, cdft, get_global_descriptors

__all__ = ['get_global_descriptors', 'run_jobs', 'cdft']

__doc__ = __doc__.format(cdft=textwrap.indent(_CDFT, 8 * ' '))

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

#: A :class:`frozenset` with all parameters of the
#: :func:`~noodles.run.threading.sqlite3.run_parallel` function. `
_RUN_PARALLEL_KEYS: FrozenSet[str] = frozenset(
    inspect.signature(run_parallel).parameters.keys()
)


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
    run_kwargs.update(split_dict(kwargs, disgard_keys=_RUN_PARALLEL_KEYS))
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
