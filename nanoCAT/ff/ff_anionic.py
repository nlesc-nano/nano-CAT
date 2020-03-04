"""
nanoCAT.ff.ff_anionic
=====================

A workflow for assigning neutral CHARMM forcefield parameters to anionic molecules.

Index
-----
.. currentmodule:: nanoCAT.ff.ff_anionic
.. autosummary::
    run_ff_anionic

API
---
.. autofunction:: run_ff_anionic

"""

from scm.plams import Molecule, Atom, Bond, Settings, MoleculeError, add_Hs

from .ff_assignment import run_match_job

__all__ = ['run_ff_anionic']


def run_ff_anionic(mol: Molecule, anchor: Atom, s: Settings) -> None:
    r"""Assign neutral parameters to an anionic species (*e.g.* carboxylate).

    Consists of 4 distinct steps:

    * **mol** is capped with a proton: *e.g.*
      :math:`RCO_2^- \rightarrow RCO_2H`.
    * Parameters are guessed for both fragments (using MATCH_) and then recombined into **mol**.
    * The capping proton is removed again.
    * The atomic charge of **anchor** is adjusted such that
      the total moleculair charge becomes zero.

    Performs an inplace update of **mol**.

    .. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software

    Parameters
    ----------
    mol : :class:`Molecule<scm.plams.mol.molecule.Molecule>`
        A cationic molecule.

    anchor : :class:`Atom<scm.plams.mol.atom.Atom>`
        The atom in **mol** with the formal negative charge.

    s : :class:`Settings<scm.plams.core.settings.Settings>`
        The job Settings to-be passed to :class:`MATCHJob<nanoCAT.ff.match_job.MATCHJob>`.

    See Also
    --------
    :func:`run_match_job()<nanoCAT.ff.ff_assignment.run_match_job>`
        Assign atom types and charges to **mol** based on the results of MATCH_.

    :func:`run_ff_cationic()<nanoCAT.ff.ff_cationic.run_ff_cationic>`
        Assign neutral parameters to a cationic species (*e.g.* ammonium).

    """  # noqa
    if anchor not in mol:
        raise MoleculeError("Passed 'anchor' is not part of 'mol'")
    anchor.properties.charge = 0

    # Cap the anion with a proton
    mol_with_h = add_Hs(mol)
    _cap_h = mol_with_h[-1]
    cap_h = Atom(atnum=_cap_h.atnum,
                 coords=_cap_h.coords,
                 mol=mol,
                 settings=mol[1].properties.copy())

    cap_h.properties.pdb_info.IsHeteroAtom = False
    cap_h.properties.pdb_info.Name = 'Hxx'
    mol.add_atom(cap_h)
    mol.add_bond(Bond(anchor, cap_h, mol=mol))

    # Guess parameters and remove the capping proton
    run_match_job(mol, s)
    mol.delete_atom(cap_h)

    # Set the total charge of the system to 0
    anchor.properties.charge_float -= sum(at.properties.charge_float for at in mol)
    return None
