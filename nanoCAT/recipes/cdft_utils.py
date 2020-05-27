"""
nanoCAT.recipes.cdft_utils
==========================

Recipes for running conceptual dft calculations.

Index
-----
.. currentmodule:: nanoCAT.recipes.cdft
.. autosummary::
    conceptual_dft
    cdft

API
---
.. autofunction:: conceptual_dft
.. autodata:: cdft
    :annotation: : qmflows.Settings

"""
from os.path import join
from typing import Optional, Mapping, Type, Any, overload, TYPE_CHECKING
from functools import partial

import yaml
from scm.plams import Settings, SingleJob, config
from qmflows import adf, Settings as QmSettings, templates as qm_templates
from qmflows.packages import registry, Package
from noodles.run.threading.sqlite3 import run_parallel

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore

if TYPE_CHECKING:
    from scm.plams import Molecule, ADFJob, ADFResults
    from qmflows.packages.SCM import ADF, ADF_Result
else:
    Molecule = 'scm.plams.Molecule'
    ADFJob = 'scm.plams.ADFJob'
    ADFResults = 'scm.plams.ADFResults'
    ADF = 'qmflows.packages.SCM.ADF'
    ADF_Result = 'qmflows.packages.SCM.ADF_Result'

__all__ = ['conceptual_dft', 'cdft']

QMFLOWS = False
PLAMS = True

#: A QMFlows-style template for conceptual DFT calculations.
cdft = QmSettings()
cdft.specific.adf = qm_templates.singlepoint.specific.adf.copy()
cdft += QmSettings(yaml.load("""
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
""", Loader=Loader))


@overload
def conceptual_dft(mol: Molecule, settings: Mapping, job_type: ADF = ..., template: Optional[Settings] = ..., **kwargs: Any) -> ADF_Result: ...  # noqa: E501
@overload
def conceptual_dft(mol: Molecule, settings: Mapping, job_type: Type[ADFJob] = ..., template: Optional[Settings] = ..., **kwargs: Any) -> ADFResults: ...  # noqa: E501
def conceptual_dft(mol, settings, job_type=adf, template=cdft.specific.adf, **kwargs):  # noqa: E501
    r"""Run a conceptual DFT workflow.

    Examples
    --------
    .. code:: python

        >>> from scm.plams import Molecule
        >>> from qmflows.utils import InitRestart
        >>> from qmflows.packages.SCM import ADF_Result

        >>> mol = Molecule(...)
        >>> settings = dict(...)

        >>> with InitRestart(path=..., folder=...):
        ...     result: ADF_Result = conceptual_dft(mol, settings)

    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule`
        The input molecule.

    settings : :class:`~collections.abc.Mapping`
        The input settings.
        The settings provided herein will, if specified,
        be updated with those from **template**.

    job_type : :class:`type` [:class:`plams.ADFJob<scm.plams.interfaces.adfsuite.adf.ADFJob>`] or :class:`qmflows.adf<qmflows.packages.SCM.ADF>`
        A PLAMS Job type or a QMFlows package instance.
        Accepted values are :code:`scm.plams.ADFJob` or :code:`qmflows.adf`.

    template : :class:`~scm.plams.core.settings.Settings`, optional
        A template used for (soft) updating the user-provided **settings**.
        (*e.g.* :data:`qmflows.templates.singlepoint.specific.adf`).

    **kwargs : :data:`~typing.Any`
        Further keyword arguments for **job_type**.

    Returns
    -------
    :class:`plams.ADFResults<scm.plams.interfaces.adfsuite.adf.ADFResults>` or :class:`qmflows.ADF_Result<qmflows.packages.SCM.ADF_Result>`
        A PLAMS :code:`Results` or QMFlows :code:`Result` object,
        depending on the value of **job_type**.

    """  # noqa: E501
    if isinstance(job_type, Package):
        flavor = QMFLOWS
    elif isinstance(job_type, type) and issubclass(job_type, SingleJob):
        flavor = PLAMS
    else:
        raise TypeError("'job_type' expected a SingleJob subclass or a Package instance; "
                        f"observed type: {job_type.__class__.__name__!r}")

    t = QmSettings() if template is None else template
    name = kwargs.pop('name', 'cdft_job')
    n_processes = kwargs.pop('n_processes', 1)

    if flavor is QMFLOWS:
        s = QmSettings(settings)
        s.input.soft_update(t)
        job = job_type(settings=s, mol=mol, name=name, **kwargs)
        cache = join(config.default_jobmanager.workdir, 'cache.db')
        runner = partial(run_parallel, n_threads=n_processes, registry=registry,
                         db_file=cache, always_cache=True, echo_log=False)

    else:
        s = Settings(settings)
        s.input.soft_update(t)
        job = job_type(settings=s, molecule=mol, name=name, **kwargs)
        runner = job_type.run

    return runner(job)
