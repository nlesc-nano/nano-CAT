###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.


0.7.3
*****
* *placeholder*.


0.7.2
*****
* Apply qmflows templates when using the dissociation workflow.
* Fixed an issue wherein the `core_index` option was not properly respected.
* Add the option to optimize QDs and XYn-dissociated QDs in the dissociation workflow.
* Fix the frequency analysis component of the BDE workflow being incompatible with custom CP2K settings.
* Allow computing the bulkiness for multiple lattice spacing values.
* Add a new workflow for computing ligand cone angles.


0.7.1
*****
* Deprecate usage of ``Molecule.get_formula`` in favor of a PLAMS <=1.5.1-based backport.
* Fix a failure in the documentation generation


0.7.0
*****
* Added a new fast-bulkiness workflow.
* Added a new COSMO-RS recipe.
* Various fixes.


0.6.1
*****
* Added a new conceptual DFT (CDFT) workflow.


0.6.0
*****
* Moved a number of functions to the `nanoutils <https://github.com/nlesc-nano/Nano-Utils>`_ package.


0.5.4
*****
* Add the ``nth_shell`` argument to the ``replace_surface()`` recipe.
  Used for applying the workflow to surfaces of one or more sub-surface shells.


0.5.3
*****
* Added a new recipe for running conceptual DFT calculations with ADF.


0.5.2
*****
* Added on option for setting ``diameter`` to ``None``.
* Added recipes for filtering molecules based on properties.
* Add an option to filter molecules based on the number of functional groups in the ``bulk_workflow()`` recipe.


0.5.1
*****
* Added a recipe for multi-ligand optimizations.
* Officially dropped Nano-CAT support for Python 3.6.
* Marked Nano-CAT as a typed package.
* Added recipe for extracting coordination numbers from Molecules;
  courtesy of https://github.com/juliette1996
  (https://github.com/nlesc-nano/nano-CAT/pull/45).
* Added a short recipe for calculating and rescaling ligand charges
  (https://github.com/nlesc-nano/nano-CAT/pull/44).


0.5.0
*****
* Moved the ``CAT.recipes`` module to Nano-CAT.
* Moved the ``CAT.attachment.qd_opt_ff`` module to Nano-CAT.
* Improved handling of Exceptions.
* Updated tests.


0.4.5
*****
* Created a separate module for the ``identify_surface()`` function,
  the latter being used for identifying which atoms are located on the surface,
  rather than in the bulk.


0.4.4
*****
* Added the ``EnenergyGatherer()`` class, a Mapping for managing all
  forcefield potential energies.
* Changed the minimum Pandas version to ``>= 0.24.0``.
* Updated the keyword arguments of ``get_asa_md()``.
* Fixed an issue were the charge was not properly set to an integer value
  when optimizing individual ligands.


0.4.3
*****
* Changed the a number of function signatures in md_asa.py to ensure signature
  compatiblity with Auto-FOX 0.7.2: https://github.com/nlesc-nano/auto-FOX/pull/79.


0.4.2
*****
* Import the now fixed ``add_Hs()`` function from PLAMS.
* Cleaned up the ``md_asa`` module.
* Following the convention of CP2K, 1,4-electrostatic interactions are now
  ignored during the MD-ASA workflow.


0.4.1
*****
* Updated the MD-ASA workflow: The ligand interaction is now calculated using
  neutral parameters; the strain using ionic parameters.


0.4.0
*****
* Made Auto-FOX a hard dependency; removed a number of (now-duplicate) functions and modules.
* Added a workflow for perfoming activation strain analyses on entire MD trajectories.


0.3.2
*****
* Reworked the ligand dissociation procedure.


0.3.1
*****
* Finalized the implementation an acitvation strain workflow with custom MATCH-based forcefields.


0.3.0
*****
* Finalize the introduction of a new CAT template system (``WorkFlow()``).
* WiP: Implement an acitvation strain workflow with custom MATCH-based forcefields.


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
