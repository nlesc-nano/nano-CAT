import os
from os.path import join
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

import numpy as np
import pandas as pd

from scm.plams import Settings, Molecule, SingleJob, writepdb, MoleculeError, config
from qmflows import run, cp2k_mm, PackageWrapper, Settings as QmSettings
from qmflows.utils import InitRestart
from qmflows.packages import Package, registry
from noodles.run.threading.sqlite3 import run_parallel

from nanoCAT.ff import MatchJob
from FOX import PSFContainer, PRMContainer, group_by_values
from FOX.io.read_psf import overlay_rtf_file

PathType = Union[str, os.PathLike]

__all__ = ['multi_ligand_opt']


def multi_ligand_opt(mol: Molecule,
                     settings: MutableMapping,
                     path: Optional[PathType] = None,
                     folder: Optional[PathType] = None,
                     **kwargs: Any):
    """Multi-ligand job.

    Examples
    --------
    .. code:: python

        >>> from qmflows.templates import geometry
        >>> from qmflows.packages import Result
        >>> from scm.plams import Molecule, readpdb

        >>> from CAT.recipes import multi_ligand_opt

        >>> mol: Molecule = readpdb(...)

        >>> settings = Settings(...)
        >>> settings.specific.cp2k += geometry.specific.cp2k_mm

        >>> results: Result = multi_ligand_opt(mol, settings)

    """
    _path = os.getcwd() if path is None else os.path.abspath(path)
    _folder = 'plams_workdir' if folder is None else folder
    workdir = join(_path, _folder)
    db_file = join(workdir, 'cache.db')

    s = QmSettings(settings)
    ff = kwargs.pop('forcefield', 'top_all36_cgenff_new')

    with InitRestart(path=path, folder=folder):
        lig_dict = _lig_from_pdb(mol)
        rtf_list, prm = _run_match(lig_dict.values(), workdir, forcefield=ff)

        psf = PSFContainer()
        psf.generate_bonds(mol)
        psf.generate_angles(mol)
        psf.generate_dihedrals(mol)
        psf.generate_impropers(mol)
        psf.generate_atoms(mol)

        start = int(psf.segment_name.iloc[-1].strip('MOL'))
        iterator = enumerate(zip(lig_dict.keys(), rtf_list), start=start)
        for i, (id_range, rtf_file) in iterator:
            overlay_rtf_file(psf, rtf_file, id_range)
            idx = np.zeros_like(psf.residue_id, dtype=bool)
            for j in id_range:
                idx[psf.residue_id == j] = True
            psf.atoms.loc[idx, 'segment name'] = f'MOL{i}'

        prm_name = join(workdir, 'mol.prm')
        psf_name = join(workdir, 'mol.psf')
        prm.write(prm_name)
        psf.write(psf_name)
        s.prm = prm_name
        s.psf = psf_name

        config.log.stdout = 0
        job = cp2k_mm(mol=mol, settings=s, **kwargs)
        return run_parallel(job, db_file=db_file, n_threads=1, always_cache=True,
                            registry=registry, echo_log=False)


def _run_match(mol_list: Iterable[Molecule], workdir,
               forcefield: str = 'top_all36_cgenff_new') -> Tuple[List[str], PRMContainer]:
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


def _concatenate_prm(file_list: Iterable[PathType]) -> PRMContainer:
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
