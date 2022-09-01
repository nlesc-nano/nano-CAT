.. image:: https://github.com/nlesc-nano/nano-CAT/workflows/Build%20with%20Conda/badge.svg
   :target: https://github.com/nlesc-nano/nano-CAT/actions?query=workflow%3A%22Build+with+Conda%22
.. image:: https://readthedocs.org/projects/cat/badge/?version=latest
   :target: https://cat.readthedocs.io/en/latest/
.. image:: https://badge.fury.io/py/nano-CAT.svg
   :target: https://badge.fury.io/py/nano-CAT

|

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
   :target: https://docs.python.org/3.7/
.. image:: https://img.shields.io/badge/python-3.8-blue.svg
   :target: https://docs.python.org/3.8/
.. image:: https://img.shields.io/badge/python-3.9-blue.svg
   :target: https://docs.python.org/3.9/
.. image:: https://img.shields.io/badge/python-3.10-blue.svg
   :target: https://docs.python.org/3.10/


########
Nano-CAT
########

**Nano-CAT** is a collection of tools for the analysis of nanocrystals,
building on the framework of the Compound Attachment Tools package (CAT_).


Installation
============

- Download miniconda for python3: miniconda_ (also you can install the complete anaconda_ version).

- Install according to: installConda_.

- Create a new virtual environment, for python 3.7, using the following commands:

  - ``conda create --name CAT python``

- The virtual environment can be enabled and disabled by, respectively, typing:

  - Enable: ``conda activate CAT``

  - Disable: ``conda deactivate``


Dependencies installation
-------------------------

Using the conda environment the following packages should be installed:

- rdkit_: ``conda install -y --name CAT --channel conda-forge rdkit``


Package installation
--------------------
Finally, install **Nano-CAT** using pip:

- **Nano-CAT**: ``pip install nano-CAT --upgrade``

Now you are ready to use **Nano-CAT**.


.. _miniconda: http://conda.pydata.org/miniconda.html
.. _anaconda: https://www.continuum.io/downloads
.. _installConda: https://docs.anaconda.com/anaconda/install/
.. _CAT: https://github.com/nlesc-nano/CAT
.. _rdkit: http://www.rdkit.org
