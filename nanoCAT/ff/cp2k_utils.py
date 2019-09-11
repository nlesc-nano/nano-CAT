"""
nanoCAT.ff.set_subsys_kind
=========================

Functions for formating CP2K input settings for forcefield-related calculations.

Index
-----
.. currentmodule:: nanoCAT.ff.match_interface
.. autosummary::
    set_subsys_kind

API
---
.. autofunction:: set_subsys_kind

"""

from scm.plams import Molecule, Settings

__all__ = ['set_subsys_kind', 'set_subsys_kind']


def set_subsys_kind(settings: Settings,
                    mol: Molecule) -> None:
    """Set the FORCE_EVAL/SUBSYS/KIND_ keyword(s) in CP2K job settings.

    Performs an inplace update of the input.force_eval.subsys key in **settings**.

    .. _KIND: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/SUBSYS/KIND.html

    Parameters
    ----------
    settings : |plams.Settings|_
        CP2K settings.

    mol : |plams.Molecule|_
        A PLAMS molecule whose atoms poses the :attr:`Atom.properties` ``["atom_type"]`` attribute.

    """
    type_dict = {at.properties.atom_type: at.symbol for at in mol}
    subsys = settings.input.force_eval.subsys

    for k, v in type_dict.items():
        subsys[f'kind {k}'] = {'element': v}


def set_charge(settings: Settings,
               mol: Molecule) -> None:
    """Set the FORCE_EVAL/MM/FORCEFIELD/CHARGE_ keyword(s) in CP2K job settings.

    Performs an inplace update of the input.force_eval.mm.forcefield.charge key in **settings**.

    .. _KIND: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/MM/FORCEFIELD/CHARGE.html

    Parameters
    ----------
    settings : |plams.Settings|_
        CP2K settings.

    mol : |plams.Molecule|_
        A PLAMS molecule whose atoms poses the :attr:`Atom.properties` ``["atom_type"]``
        and :attr:`Atom.properties` ``["charge"]`` attributes.

    """
    charge_dict = {at.properties.atom_type: at.properties.charge for at in mol}
    charge = settings.input.force_eval.mm.forcefield.charge = []

    for k, v in charge_dict.items():
        new_charge = Settings({'atom': k, 'charge': f'{v:.6f}'})
        charge.append(new_charge)
