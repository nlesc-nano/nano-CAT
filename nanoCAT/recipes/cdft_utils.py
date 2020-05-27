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
import warnings
from os import PathLike
from os.path import join
from typing import Union, Optional, Mapping, Type, Any, overload, TYPE_CHECKING
from functools import partial

import yaml
from scm.plams import Settings, SingleJob, config
from qmflows import adf, Settings as QmSettings, templates as qm_templates
from qmflows.utils import InitRestart
from qmflows.packages import registry, Package
from qmflows.packages.SCM import ADF
from noodles.run.threading.sqlite3 import run_parallel

from FOX.utils import get_importable

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

PathType = Union[str, PathLike]

@overload
def conceptual_dft(mol: Molecule, settings: Mapping, job_type: ADF = ..., template: Optional[str] = ..., path: Optional[PathType] = ..., folder: Optional[PathType] = ..., **kwargs: Any) -> ADF_Result: ...
@overload
def conceptual_dft(mol: Molecule, settings: Mapping, job_type: Type[ADFJob] = ..., template: Optional[str] = ..., path: Optional[PathType] = ..., folder: Optional[PathType] = ..., **kwargs: Any) -> ADFResults: ...
def conceptual_dft(mol, settings, job_type=adf, template='CAT.recipes.cdft.specific.adf', path=None, older=None, **kwargs):
    """Run a conceptual DFT workflow.

    Examples
    --------
    .. code:: python

        >>> from scm.plams import Molecule
        >>> from qmflows.packages.SCM import ADF_Result

        >>> mol = Molecule(...)
        >>> settings = dict(...)

        >>> result: ADF_Result = conceptual_dft(mol, settings)

    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule`
        The input molecule.

    settings : :class:`~collections.abc.Mapping`
        The input settings.
        The settings provided herein will, if specified,
        be updated with those imported from **template**.

    job_type : :class:`type` [:class:`plams.ADFJob<scm.plams.interfaces.adfsuite.adf.ADFJob>`] or :class:`qmflows.adf<qmflows.packages.SCM.ADF>`
        A PLAMS Job type or a QMFlows package instance.
        Accepted values are :code:`scm.plams.ADFJob` or :code:`qmflows.adf`.

    template : :class:`str`, optional
        A string pointing to a settings template
        The settings will be imported from the specified path
        (*e.g.* :data:`qmflows.templates.singlepoint.specific.adf`).

    path : :class:`str`, optional
        The path to the PLAMS working directory.

    folder : :class:`str`, optional
        The name of the PLAMS working directory.

    Returns
    -------
    :class:`plams.ADFResults<scm.plams.interfaces.adfsuite.adf.ADFResults>` or :class:`qmflows.ADF_Result<qmflows.packages.SCM.ADF_Result>`
        A PLAMS :code:`Results` or QMFlows :code:`Result` object, depending on the value of **job_type**.

    """  # noqa: E501
    if isinstance(job_type, Package):
        pass
    elif isinstance(job_type, type) and issubclass(job_type, SingleJob):
        pass
    else:
        raise TypeError("'job_type' expected a SingleJob subclass or a Package instance; "
                        f"observed type: {job_type.__class__.__name__!r}")

    t = QmSettings() if template is None else get_importable(template)
    name = kwargs.pop('name', 'cdft_job')
    n_processes = kwargs.pop('n_processes', 1)

    with InitRestart(path=path, folder=folder):
        if isinstance(job_type, Package):
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
