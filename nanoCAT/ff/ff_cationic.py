"""
nanoCAT.ff.ff_cationic
======================

A worklflow for assigning neutral CHARMM forcefield parameters to cationic molecules.

Index
-----
.. currentmodule:: nanoCAT.ff.ff_cationic
.. autosummary::
    run_ff_cationic

API
---
.. autofunction:: run_ff_cationic

"""

from scm.plams import Molecule, Atom, Bond, Settings, MoleculeError, add_Hs
from CAT.attachment.mol_split_cm import SplitMol

from .ff_assignment import run_match_job

__all__ = ['run_ff_cationic']


def run_ff_cationic(mol: Molecule, anchor: Atom, s: Settings) -> None:
    r"""Assign neutral parameters to a cationic species (*e.g.* ammonium).

    Consists of 3 distinct steps:

    * **mol** is converted into two neutral fragments,
      *e.g.* ammonium is converted into two amines:
      :math:`N^+(R)_4 \rightarrow N(R)_3 + RN(H)_2`.
    * Parameters are guessed for both fragments (using MATCH_) and then recombined into **mol**.
    * The atomic charge of **anchor** is adjusted such that
      the total moleculair charge becomes zero.

    Performs an inplace update of **mol**.

    .. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

    Parameters
    ----------
    mol : :class:`Molecule<scm.plams.mol.molecule.Molecule>`
        A cationic molecule.

    anchor : :class:`Atom<scm.plams.mol.atom.Atom>`
        The atom in **mol** with the formal positive charge.

    s : :class:`Settings<scm.plams.core.settings.Settings>`
        The job Settings to-be passed to :class:`MATCHJob<nanoCAT.ff.match_job.MATCHJob>`.

    See Also
    --------
    :func:`run_match_job()<nanoCAT.ff.ff_assignment.run_match_job>`
        Assign atom types and charges to **mol** based on the results of MATCH_.

    :func:`run_ff_anionic()<nanoCAT.ff.ff_anionic.run_ff_anionic>`
        Assign neutral parameters to an anionic species (*e.g.* carboxylate).

    """  # noqa
    if anchor not in mol:
        raise MoleculeError("Passed 'anchor' is not part of 'mol'")
    anchor.properties.charge = 0

    # Find the first bond attached to the anchor atom which is not part of a ring
    for bond in anchor.bonds:
        if not mol.in_ring(bond):
            break
    else:
        raise MoleculeError("All bonds attached to 'anchor' are part of a ring system")

    with SplitMol(mol, bond) as (frag1, frag2):
        # Identify the amine and the alkylic fragment
        if anchor in frag1:
            amine = frag1
            alkyl = frag2
        else:
            amine = frag2
            alkyl = frag1
        amine.delete_atom(anchor.bonds[-1].other_end(anchor))

        # Change the capping hydrogen into X
        # X is the same atom type as **anchor**
        alkyl_cap = alkyl[-1]
        alkyl_cap.atnum = anchor.atnum
        cap_bond = alkyl_cap.bonds[0]
        bond_length = alkyl_cap.radius + cap_bond.other_end(alkyl_cap).radius
        cap_bond.resize(alkyl_cap, bond_length)

        # Change X into XH_n
        alkyl_with_h = add_Hs(alkyl)
        properties = mol[1].properties
        for at in alkyl_with_h.atoms[len(alkyl):]:
            cap_h = Atom(atnum=at.atnum, coords=at.coords, mol=alkyl, settings=properties.copy())
            cap_h.properties.pdb_info.IsHeteroAtom = False
            cap_h.properties.pdb_info.Name = 'Hxx'
            alkyl.add_atom(cap_h)
            alkyl.add_bond(Bond(alkyl_cap, cap_h, mol=alkyl))

        # Get the match parameters
        run_match_job(amine, s)
        run_match_job(alkyl, s)

    # Set the total charge of the system to 0
    anchor.properties.charge_float -= sum(at.properties.charge_float for at in mol)
    return None
