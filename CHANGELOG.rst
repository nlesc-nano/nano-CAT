###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.


0.3.0
*****
* Finalize the introduction of a new CAT template system (``WorkFlow()``).


0.2.4
*****
* Custom Job types and Job Settings can now be specified for the ligand
  Activation Strain workflow.
* Removed ``CRSJob()`` and ``CRSResults()``; import them from PLAMS instead.
* Import assertions from AssertionLib_ rather than CAT_.
* WiP: Introduction of a new CAT template system (``WorkFlow()``).


0.2.3
*****
* Moved the ligand bulkiness workflow from the `ligand` to the `qd` block in the CAT input.
* Updated the formula for the ligand bulkiness calculation.


0.2.2
*****
* Updated the ligand bulkiness workflow.


0.2.1
*****
* Added a workflow for calculating ligand bulkiness.


0.2.0
*****
* Implemented an interface to MATCH_: Multipurpose Atom-Typer for CHARMM.
* Added the ``PSFContainer()`` and ``PRMContainer()`` classes handling .psf and .prm files, respectively.
* Updated the handling of assertions, see ``CAT.assertions.assertion_manager``.
* A couple of bug fixes to the ligand dissociation module(s).
* Added tests.


0.1.4
*****
* Bug fix: polyatomic ions are now properly dissociated in the ligand dissociation module(s).


0.1.3
*****
* Lowered Python version requirement from >=3.7 to >=3.6.


0.1.2
*****
* Introduced a proper logger (see https://github.com/nlesc-nano/CAT/issues/46 and
  https://github.com/nlesc-nano/CAT/pull/47).


0.1.1
*****
* Added now features to the ligand dissociation module
  (see https://github.com/nlesc-nano/nano-CAT/issues/1).


[Unreleased]
************
* Empty Python project directory structure.


.. _AssertionLib: https://github.com/nlesc-nano/AssertionLib
.. _CAT: https://github.com/nlesc-nano/CAT
.. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software
