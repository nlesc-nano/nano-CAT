import yaml

from scm.plams import Settings, init, finish

from CAT import base
from CAT.logger import logger

from nanoCAT.ff.match_job import MatchJob
from nanoCAT.ff.cp2k_utils import set_cp2k_param
from nanoCAT.ff.psf import PSF


yaml_path = '/Users/basvanbeek/Documents/GitHub/CAT/examples/input_settings.yaml'
with open(yaml_path, 'r') as file:
    arg = Settings(yaml.load(file, Loader=yaml.FullLoader))

try:
    qd_df, *_ = base.prep(arg)
except Exception as ex:
    logger.critical(f'{ex.__class__.__name__}: {ex}', exc_info=True)
    raise ex

MOL = ('mol', '')
mol = qd_df[MOL].iloc[0]

init(path=qd_df.settings.optional.qd.dirname)
s = Settings()
s.input.forcefield = 'top_all36_cgenff'
job = MatchJob(molecule=mol, settings=s)
results = job.run()
finish()

lig_count = mol[-1].properties.pdb_info.ResidueNumber - 1
at_charge = iter(results.get_charges() * lig_count)
at_type = iter(results.get_atom_types() * lig_count)

at_gen = (at for at in mol)
for at in at_gen:
    if at.properties.pdb_info.ResidueName != 'COR':
        break

at.properties.charge = next(at_charge)
at.properties.atom_type = next(at_type)
for at, charge, at_type in zip(at_gen, at_charge, at_type):
    at.properties.charge = charge
    at.properties.atom_type = at_type

psf = PSF()
psf.generate_bonds(mol)
psf.generate_angles(mol)
psf.generate_dihedrals(mol)
psf.generate_impropers(mol)
psf.generate_atoms(mol)
psf.write('/Users/basvanbeek/Downloads/test_qd.psf')

yaml_path2 = '/Users/basvanbeek/Documents/GitHub/nano-CAT/nanoCAT/test_cp2k_ff/param.yaml'
with open(yaml_path2, 'r') as f:
    param_s = Settings(yaml.load(f, Loader=yaml.FullLoader))

s_cp2k = Settings()
set_cp2k_param(s_cp2k, param_s)
