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

from scm.plams import (Molecule, Settings)

__all__ = ['set_cp2k_element', 'set_cp2k_charge', 'set_cp2k_value']


def set_cp2k_element(settings: Settings, mol: Molecule) -> None:
    """Set the FORCE_EVAL/SUBSYS/KIND/ELEMENT_ keyword(s) in CP2K job settings.

    Performs an inplace update of the input.force_eval.subsys key in **settings**.

    .. _ELEMENT: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/SUBSYS/KIND.html#ELEMENT

    Parameters
    ----------
    settings : |plams.Settings|_
        CP2K settings.

    mol : |plams.Molecule|_
        A PLAMS molecule whose atoms possess the
        :attr:`Atom.properties` ``["atom_type"]`` attribute.

    """
    # {atom_type: atom_symbol}
    symbol_dict = {at.properties.atom_type: at.symbol for at in mol}
    subsys = settings.input.force_eval.subsys

    for k, v in symbol_dict.items():
        subsys[f'kind {k}'] = {'element': v}


def set_cp2k_charge(settings: Settings, mol: Molecule) -> None:
    """Set the FORCE_EVAL/MM/FORCEFIELD/CHARGE_ keyword(s) in CP2K job settings.

    Performs an inplace update of the input.force_eval.mm.forcefield.charge key in **settings**.

    .. _CHARGE: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/MM/FORCEFIELD/CHARGE.html#CHARGE

    Parameters
    ----------
    settings : |plams.Settings|_
        CP2K settings.

    mol : |plams.Molecule|_
        A PLAMS molecule whose atoms possess the :attr:`Atom.properties` ``["atom_type"]``
        and :attr:`Atom.properties` ``["charge"]`` attributes.

    """
    # {atom_type: atom_charge}
    charge_dict = {at.properties.atom_type: at.properties.charge for at in mol}
    charge = settings.input.force_eval.mm.forcefield.charge = []

    for k, v in charge_dict.items():
        new_charge = Settings({'atom': k, 'charge': f'{v:.6f}'})
        charge.append(new_charge)


def set_cp2k_param(settings: Settings, param_dict: dict) -> None:
    """Placeholder."""
    for block_name, block in param_dict.items():
        # Create a to-be formatted string with user-specified units
        unit = f'[{block.unit}]' + ' {}' if 'unit' in block else '{}'

        # Get the to-be update list of settings
        s = settings.get_nested(block['keys'])
        if not isinstance(s, list):
            _s = settings.get_nested(block['keys'][:-1])
            s = _s[block['keys'][-1]] = []

        for k, v in block.items():
            if k in ('keys', 'unit'):  # Skip
                continue

            value = unit.format(v)
            atom = 'atoms' if len(k.split()) > 1 else 'atom'
            atom_list = [i[atom] for i in s]

            try:  # Intersecting set
                idx = atom_list.index(k)
                s[idx].update({block_name: value})
            except ValueError:  # Disjoint set
                new_block = Settings({atom: k, block_name: value})
                s.append(new_block)
