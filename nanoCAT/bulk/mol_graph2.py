"""Functions for computing the size and distances of branches within molecules."""

from __future__ import annotations

import sys
import dataclasses
import functools
import itertools
from typing import TYPE_CHECKING, Any
from collections.abc import Iterable

import numpy as np
from FOX.ff import degree_of_separation
from scm.plams import Molecule, Atom

from CAT.attachment.mol_split_cm import SplitMol
from CAT.attachment.ligand_opt import split_mol
from CAT.workflows import WorkFlow
from CAT.settings_dataframe import SettingsDataFrame

if sys.version_info >= (3, 10):
    dataclass = functools.partial(dataclasses.dataclass, slots=True)
else:
    dataclass = dataclasses.dataclass

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import int64 as i8

__all__ = ["BranchGraph", "get_branching_graph", "init_branch_distance"]


def init_branch_distance(ligand_df: SettingsDataFrame) -> None:
    """Initialize the branch size & distance workflow.

    Parameters
    ----------
    ligand_df : |CAT.SettingsDataFrame|
        A DataFrame of ligands.

    """
    workflow = WorkFlow.from_template(ligand_df, name='branch_distance')

    # Import from the database and start the calculation
    idx = workflow.from_db(ligand_df)
    workflow(_start_branch_jobs, ligand_df, index=idx)

    # Export to the database
    workflow.to_db(ligand_df, index=idx)


def _start_branch_jobs(mol_list: Iterable[Molecule], **kwargs: Any) -> list[tuple[str, str]]:
    """Helper function for the ``branch_distance`` workflow."""
    ret = []
    for mol in mol_list:
        anchor = mol.properties.dummies
        graph = get_branching_graph(mol, anchor)
        distance = "-".join(str(i) for i in graph.branch_distance)
        size = "-".join(str(i) for i in graph.branch_size)
        ret.append((distance, size))
    return ret


@dataclass
class BranchGraph:
    """A dataclass for containing branching-related information for a given molecule."""

    #: The matrix containg the degree of separation for each atom-pair in the (super-)molecule.
    degree_of_separation: NDArray[i8]

    #: Indices defining the various branches.
    branches: list[NDArray[i8]]

    #: The initial atom in each fragment, i.e. the one closest to the anchor.
    branch_start: list[int]

    @property
    def branch_size(self) -> list[int]:
        """Get the size of each branch in :attr:`branches`."""
        return [len(i) for i in self.branches]

    @property
    def anchor(self) -> int:
        """Get the index of the anchor atom."""
        return self.branch_start[0]

    @property
    def branch_distance(self) -> NDArray[i8]:
        """Get the degree of separation of all branches w.r.t. the :attr:`anchor`."""
        return self.degree_of_separation[self.anchor, self.branch_start]

    def __eq__(self, other: object) -> bool:
        """Implement :meth:`self == other <object.__eq__>`."""
        if not isinstance(other, BranchGraph):
            return NotImplemented

        branch_ziperator = itertools.zip_longest(self.branches, other.branches, fillvalue=np.nan)
        return (
            (self.degree_of_separation == other.degree_of_separation).all()
            and all(np.all(b1 == b2) for b1, b2 in branch_ziperator)
            and self.branch_start == other.branch_start
        )


def get_branching_graph(mol: Molecule, anchor: Atom) -> BranchGraph:
    """Construct a :class:`BranchGraph` for the given molecule and anchor atom."""
    mat = degree_of_separation(mol).astype(np.int64)
    bonds = split_mol(mol, anchor)
    anchor_idx = mol.index(anchor) - 1
    try:
        mol.set_atoms_id(start=0)
        branch_start = [anchor_idx]
        for at1, at2 in bonds:
            dist1 = mat[anchor_idx, at1.id]
            dist2 = mat[anchor_idx, at2.id]
            branch_start.append(at1.id if dist1 <= dist2 else at2.id)

        branches = []
        with SplitMol(mol, bonds) as frag_tup:
            for mol_frag in frag_tup:
                frag = np.fromiter([getattr(at, "id", -1) for at in mol_frag], dtype=np.int64)
                branches.append(frag[frag >= 0])
    finally:
        mol.unset_atoms_id()
    return BranchGraph(
        degree_of_separation=mat,
        branches=branches,
        branch_start=branch_start,
    )
