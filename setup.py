#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit CAT/__version__.py
version = {}
with open(os.path.join(here, 'CAT', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='nano-CAT',
    version=version['__version__'],
    description='A collection of tools for the analysis of nanocrystals.',
    long_description=readme + '\n\n',
    author=['Bas van Beek'],
    author_email='b.f.van.beek@vu.nl',
    url='https://github.com/nlesc-nano/nano-CAT',
    packages=[
        'nano_CAT'
    ],
    package_dir={'nano': 'nano-CAT'},
    package_data={
        'nano_CAT': []
    },
    entry_points={},
    include_package_data=True,
    license='GNU Lesser General Public License v3 or later',
    zip_safe=False,
    keywords=[
        'quantum-mechanics',
        'molecular-mechanics',
        'science',
        'chemistry',
        'python-3',
        'python-3-7',
        'automation',
        'scientific-workflows'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU Lesser General Public License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pyyaml>=5.1',
        'schema',
        'plams@git+https://github.com/SCM-NV/PLAMS@release',
        'qmflows@git+https://github.com/SCM-NV/qmflows@master',
        'CAT@git+https://github.com/nlesc-nano/CAT@master'
    ],
    setup_requires=[
        'pytest-runner',
        'sphinx',
        'sphinx_rtd_theme',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={
        'test': ['pytest', 'pytest-cov', 'pytest-mock', 'nbsphinx', 'pycodestyle'],
        'doc': ['sphinx', 'sphinx_rtd_theme', 'sphinx-autodoc-typehints']
    }
)
