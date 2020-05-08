import os
from os.path import join
from enum import IntEnum
from functools import partial
from contextlib import nullcontext
from typing import (
    Union,
    Type,
    MutableMapping,
    Optional,
    Any,
    List,
    Dict,
    Tuple,
    Iterable
)

from scm.plams import Settings, Molecule, SingleJob, readpdb, MoleculeError, config
from qmflows import run, cp2k_mm, PackageWrapper, Settings as QmSettings
from qmflows.utils import InitRestart
from qmflows.packages import Package

from nanoCAT.ff import MatchJob
from FOX import PSFContainer, PRMContainer, group_by_values
from FOX.io.read_psf import overlay_rtf_file

PathType = Union[str, os.PathLike]

__all__ = ['multi_ligand_opt']


def multi_ligand_opt(pdb_file: PathType,
                     settings: MutableMapping,
                     job_type: Union[Type[SingleJob], Package] = cp2k_mm,
                     path: Optional[PathType] = None,
                     folder: Optional[PathType] = None,
                     **kwargs: Any):
    """Multi-ligand job.

    Examples
    --------
    .. code:; python

        >>> from qmflows.templates import geometry
        >>> from scm.plams import Settings
        >>> from CAT.recipes import multi_ligand_opt

        >>> pdb = str(...)

        >>> settings = Settings()
        >>> s.specific.cp2k += geometry.specific.cp2k_mm

        >>> results = multi_ligand_opt(pdb, settings)


    """
    job_style, context, runner, s = _parse_job(settings, job_type, path=path, folder=folder)
    mol = readpdb(pdb_file)

    with context:
        ff = kwargs.pop('forcefield', 'top_all36_cgenff_new')

        if job_style == _JobStyle.PLAMS:
            job = job_type(molecule=mol, settings=s, **kwargs)
            return runner(job)

        lig_dict = _lig_from_pdb(mol)
        rtf_list, prm = _run_match(lig_dict.values(), runner, forcefield=ff)

        psf = PSFContainer()
        psf.generate_bonds(mol)
        psf.generate_angles(mol)
        psf.generate_dihedrals(mol)
        psf.generate_impropers(mol)
        psf.generate_atoms(mol)
        for id_range, rtf_file in zip(lig_dict.keys(), rtf_list):
            overlay_rtf_file(psf, rtf_file, id_range)

        workdir = config.default_jobmanager.workdir
        psf.write(join(workdir, 'mol.psf'))
        prm.write(join(workdir, 'mol.prm'))
        s.psf = join(workdir, 'mol.psf')
        s.prm = join(workdir, 'mol.prm')

        job = job_type(mol=mol, settings=s, **kwargs)
        return runner(job)


class _JobStyle(IntEnum):
    PLAMS = 0
    QMFLOWS = 1


def _run_match(mol_list: Iterable[Molecule], runner=None,
               forcefield: str = 'top_all36_cgenff_new') -> Tuple[List[str], PRMContainer]:
    s = Settings()
    s.input.forcefield = forcefield

    if runner is not None:
        runner_ = runner
        partial_job = partial(PackageWrapper(MatchJob), settings=s)
    else:
        runner_ = MatchJob.run
        partial_job = partial(MatchJob, settings=s)

    rtf_list = []
    prm_list = []
    for mol in mol_list:
        job = partial_job(molecule=mol) if runner is None else partial_job(mol=mol)
        results = runner_(job)

        rtf = results.files['$JN.rtf']
        prm = results.files['$JN.prm']
        rtf_list.append(rtf)
        prm_list.append(prm)

    prm_ = _concatenate_prm(prm_list)
    return rtf_list, prm_


def _concatenate_prm(file_list: Iterable[PathType]) -> PRMContainer:
    prm_list = [PRMContainer.read(file) for file in file_list]
    iterator = iter(prm_list)
    ret = next(iterator)

    for prm in iterator:
        items = ((k, getattr(prm, k)) for k in prm.HEADERS[:-2])
        for k, v in items:
            df = getattr(ret, k)
            if v is None:
                continue
            elif df is None:
                setattr(ret, k, v)
            else:
                setattr(ret, k, df.append(v))

    items = ((k, getattr(prm, k)) for k in prm.HEADERS[:-2])
    for k, df in items:
        df_ = df.loc[~df.index.duplicated(keep='first')]
        setattr(ret, k, df)
    return ret


def _lig_from_pdb(mol: Molecule) -> Dict[Tuple[int, ...], Molecule]:
    def get_res(mol: Molecule) -> int:
        return mol[1].properties.pdb_info.ResidueNumber

    if not mol.bonds:
        raise MoleculeError("The passed molecule has no bonds")

    mol_lig = mol.copy()
    core_atoms = [at for at in mol_lig if at.properties.pdb_info.ResidueName == 'COR']
    for at in core_atoms:
        mol_lig.delete_atom(at)
    ligands = mol_lig.separate()

    iterator = ((get_res(lig), tuple(at.atnum for at in lig)) for lig in ligands)
    grouped = group_by_values(iterator)
    ret = {}
    for _k, v in grouped.items():
        k = tuple(v)
        i = v[-1] - 2
        ret[k] = ligands[i]
    return ret


def _parse_job(settings, job_type, path=None, folder=None):
    if isinstance(job_type, Package):
        job_style = _JobStyle.QMFLOWS
        context = nullcontext()
        s = QmSettings(settings) if type(settings) is not QmSettings else settings
        runner = partial(run, path=path, folder=folder)

    elif isinstance(job_type, type) and issubclass(job_type, SingleJob):
        job_style = _JobStyle.PLAMS
        context = InitRestart(path=path, folder=folder)
        s = Settings(settings) if type(settings) is not Settings else settings
        runner = type(job_type).run

    else:
        raise TypeError("'job_type' expected a Job subclass or Package instance; "
                        f"observed type: {job_type.__class__.__name__!r}")
    return job_style, context, runner, s
