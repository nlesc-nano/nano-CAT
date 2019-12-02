.. image:: https://travis-ci.org/nlesc-nano/nano-CAT.svg?branch=master
   :target: https://travis-ci.org/nlesc-nano/nano-CAT

|

.. image:: https://img.shields.io/badge/python-3.6-blue.svg
   :target: https://www.python.org

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
   :target: https://www.python.org


##############
Nano-CAT 0.4.0
##############

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

- **Nano-CAT**: ``pip install git+https://github.com/nlesc-nano/nano-CAT@master --upgrade``

Now you are ready to use **Nano-CAT**.


.. _miniconda: http://conda.pydata.org/miniconda.html
.. _anaconda: https://www.continuum.io/downloads
.. _installConda: https://docs.anaconda.com/anaconda/install/
.. _CAT: https://github.com/nlesc-nano/CAT
.. _rdkit: http://www.rdkit.org
