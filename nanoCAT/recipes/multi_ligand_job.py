"""Recipes for running classical forcefield calculations with CP2K on quantum dots with multiple non-unique ligands.

Index
-----
.. currentmodule:: nanoCAT.recipes.multi_ligand_job
.. autosummary::
    multi_ligand_job

API
---
.. autofunction:: multi_ligand_job

"""  # noqa: E501

import os
from os.path import join
from typing import (
    Union,
    MutableMapping,
    Optional,
    Any,
    List,
    Dict,
    Tuple,
    Iterable,
    FrozenSet
)

from scm.plams import Settings, Molecule, config
from qmflows import cp2k_mm, Settings as QmSettings
from qmflows.utils import InitRestart
from qmflows.packages import registry
from qmflows.packages.cp2k_mm import CP2KMM_Result
from noodles.run.threading.sqlite3 import run_parallel

from CAT.utils import SetAttr
from nanoCAT.ff import MatchJob
from nanoCAT.qd_opt_ff import constrain_charge
from FOX import PSFContainer, PRMContainer
from FOX.io.read_psf import overlay_rtf_file

__all__ = ['multi_ligand_job']

PathType = Union[str, os.PathLike]

SET_CONFIG_STDOUT = SetAttr(config.log, 'stdout', 0)


def multi_ligand_job(mol: Molecule,
                     psf: Union[PathType, PSFContainer],
                     settings: MutableMapping,
                     path: Optional[PathType] = None,
                     folder: Optional[PathType] = None,
                     **kwargs: Any) -> CP2KMM_Result:
    r"""Estimate forcefield parameters using MATCH and then run a MM calculation with CP2K.

    Examples
    --------
    .. code:: python

        >>> from qmflows import Settings
        >>> from qmflows.templates import geometry
        >>> from qmflows.packages import Result
        >>> from scm.plams import Molecule

        >>> from CAT.recipes import multi_ligand_job

        >>> mol = Molecule(...)
        >>> psf = str(...)

        # Example input settings for a geometry optimization
        >>> settings = Settings()
        >>> settings.specific.cp2k += geometry.specific.cp2k_mm
        >>> settings.charge = {
        ...     'param': 'charge',
        ...     'Cd': 2,
        ...     'Se': -2
        ... }
        >>> settings.lennard_jones = {
        ...     'param': ('epsilon', 'sigma'),
        ...     'unit': ('kcalmol', 'angstrom'),
        ...     'Cd': (1, 1),
        ...     'Se': (2, 2)
        ... }

        >>> results: Result = multi_ligand_job(mol, psf, settings)


    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule`
        The input molecule.

    psf : :class:`~FOX.io.read_psf.PSFContainer` or path-like
        A PSFContainer or path to a .psf file.

    settings : :class:`~scm.plams.core.settings.Settings`
        The QMFlows-style CP2K input settings.

    path : path-like, optional
        The path to the PLAMS working directory.

    folder : path-like, optional
        The name of the PLAMS working directory.

    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for
        :meth:`qmflows.cp2k_mm()<qmflows.packages.packages.Package.__call__>`.

    Returns
    -------
    :class:`~qmflows.packages.cp2k_mm.CP2KMM_Result`
        The results of the :class:`~qmflows.packages.cp2k_mm.CP2KMM` calculation.

    See Also
    --------
    :func:`FOX.recipes.generate_psf2()<FOX.recipes.psf.generate_psf2>`
        Generate a :class:`~FOX.io.read_psf.PSFContainer` instance for **qd**
        with multiple different **ligands**.

    :data:`qmflows.cp2k_mm()<qmflows.packages.cp2k_mm.cp2k_mm>`
        An instance of :class:`~qmflows.packages.cp2k_mm.CP2KMM`;
        used for running classical forcefield calculations with `CP2K <https://www.cp2k.org/>`_.

    `10.1002/jcc.21963<https://doi.org/10.1002/jcc.21963>`_
        MATCH: An atom-typing toolset for molecular mechanics force fields,
        J.D. Yesselman, D.J. Price, J.L. Knight and C.L. Brooks III,
        J. Comput. Chem., 2011.

    """
    path_: str = os.getcwd() if path is None else os.path.abspath(path)
    folder_: str = 'plams_workdir' if folder is None else os.path.normpath(folder)
    workdir = join(path_, folder_)
    db_file = join(workdir, 'cache.db')

    s = settings.copy() if isinstance(settings, QmSettings) else QmSettings(settings)
    psf_: PSFContainer = psf if isinstance(psf, PSFContainer) else PSFContainer.read(psf)

    with InitRestart(path=path_, folder=folder_):
        return _multi_ligand_job(mol, psf, s, workdir, db_file, **kwargs)


def _multi_ligand_job(mol: Molecule, psf: PSFContainer, settings: QmSettings,
                      workdir: str, db_file: str, **kwargs: Any) -> CP2KMM_Result:
    """Helper function for :func:`multi_ligand_job`."""
    # Run MATCH on all unique ligands
    lig_dict = _lig_from_psf(mol, psf)
    ff: str = kwargs.pop('forcefield', 'top_all36_cgenff_new')
    rtf_list, prm = _run_match(lig_dict.values(), forcefield=ff)

    # Update the .psf file with all MATCH results
    initial_charge = psf.charge.sum()
    for id_range, rtf_file in zip(lig_dict.keys(), rtf_list):
        overlay_rtf_file(psf, rtf_file, id_range)

    # Update the charge
    _constrain_charge(psf, settings, initial_charge)

    # Write the new .prm and .psf files and update the CP2K settings
    prm_name = join(workdir, 'mol.prm')
    psf_name = join(workdir, 'mol.psf')
    prm.write(prm_name)
    psf.write(psf_name)
    settings.prm = prm_name
    settings.psf = psf_name

    # Run the actual CP2K job
    with SET_CONFIG_STDOUT:
        job = cp2k_mm(mol=mol, settings=settings, **kwargs)
        return run_parallel(job, db_file=db_file, n_threads=1, always_cache=True,
                            registry=registry, echo_log=False)


def _constrain_charge(psf: PSFContainer, settings: MutableMapping,
                      initial_charge: float = 0.0) -> None:
    """Constrain the net moleculair charge such that it is equal to **initial_charge**."""
    atom_set = set(psf.atom_type[psf.residue_name == 'LIG'].values)

    # Check for user-specified atomic charges in the input settings
    charge_settings = settings.pop('charge', None)
    if charge_settings is not None:
        del charge_settings['param']
        for k, v in charge_settings.items():
            psf.charge[psf.atom_type == k] = v
            if k in atom_set:
                atom_set.remove(k)

    # Renormalize the charge
    constrain_charge(psf, initial_charge, atom_set=atom_set)


def _run_match(mol_list: Iterable[Molecule],
               forcefield: str = 'top_all36_cgenff_new') -> Tuple[List[str], PRMContainer]:
    """Run a :class:`MatchJob`, using **forcefield**, on all molecules in **mol_list**."""
    s = Settings()
    s.input.forcefield = forcefield

    rtf_list = []
    prm_list = []
    for i, mol in enumerate(mol_list):
        job = MatchJob(molecule=mol, settings=s, name=f'match_job{i}')
        results = job.run()

        rtf = results['$JN.rtf']
        prm = results['$JN.prm']
        rtf_list.append(rtf)
        prm_list.append(prm)

    prm_ = _concatenate_prm(prm_list)
    return rtf_list, prm_


# TODO: Create a dedicated PRMContainer.concatenate() method
def _concatenate_prm(file_list: Iterable[PathType]) -> PRMContainer:
    """Concatenate a list of .prm files into a single :class:`PRMContainer`."""
    prm_list = [PRMContainer.read(file) for file in file_list]
    iterator = iter(prm_list)
    ret = next(iterator)

    for prm in iterator:
        items = ((k.lower(), getattr(prm, k.lower())) for k in prm.HEADERS[:-2])
        for k, v in items:
            df = getattr(ret, k)
            if v is None:
                continue
            elif df is None:
                setattr(ret, k, v)
            else:
                setattr(ret, k, df.append(v))

    items = ((k.lower(), getattr(ret, k.lower())) for k in ret.HEADERS[:-2])
    for k, df in items:
        if df is None:
            continue
        df_ = df.loc[~df.index.duplicated(keep='first')]
        setattr(ret, k, df_)
    return ret


def _lig_from_psf(mol: Molecule, psf: PSFContainer) -> Dict[FrozenSet[int], Molecule]:
    residue_set = set(psf.segment_name[psf.residue_name == 'LIG'])

    ret: Dict[FrozenSet[int], Molecule] = {}
    for res in residue_set:
        _key = psf.residue_id[psf.segment_name == res]
        key = frozenset(_key)

        res_id = _key.iat[0]
        idx = psf.atoms[psf.residue_id == res_id].index
        i = idx[0] - 1
        j = idx[-1]

        ret[key] = mol.copy(atoms=mol.atoms[i:j])
        ret[key].round_coords(decimals=3)
    return ret
