import ase.io.cif
from FOX import MultiMolecule
from nanoCAT.core_construct import get_surface_intersections
from scm.plams import Molecule, Atom
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
"""
cif_file = r"/Users/basvanbeek/Downloads/PbS layered corrected.cif"
ase_mol = next(ase.io.cif.read_cif(cif_file, slice(None)))
mol = MultiMolecule.from_ase(ase_mol).delete_atoms("Pb")
"""

miller = [
    [1, 1, 1],
]
a, b = get_surface_intersections(miller, radius=12)

mol = Molecule()
for coords in a:
    atom = Atom(symbol="I", coords=coords)
    mol.add_atom(atom)
for translate in product([-1, 1], repeat=3):
    atom = Atom(symbol="Br", coords=np.array(mol[1].coords) - translate)
    mol.add_atom(atom)



"""
fig = plt.figure()

# Add an axes
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d', proj_type='ortho')
for x, y, z in b[0]:
    ax.plot([-x, x], [-y, y], [-z, z])
"""
