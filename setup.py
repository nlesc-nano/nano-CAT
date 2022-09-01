#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit nano_CAT/__version__.py
version = {}
with open(os.path.join(here, 'nanoCAT', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='Nano-CAT',
    version=version['__version__'],
    description='A collection of tools for the analysis of nanocrystals.',
    long_description=readme + '\n\n',
    author=['Bas van Beek'],
    author_email='b.f.van.beek@vu.nl',
    url='https://github.com/nlesc-nano/nano-CAT',
    packages=[
        'nanoCAT',
        'nanoCAT.bde',
        'nanoCAT.asa',
        'nanoCAT.ff',
        'nanoCAT.recipes',
        'nanoCAT.bulk',
    ],
    package_dir={'nanoCAT': 'nanoCAT'},
    package_data={'nanoCAT': ['data/*csv', 'py.typed', '*.pyi']},
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
        'python-3-8',
        'python-3-9',
        'python-3-10',
        'automation',
        'scientific-workflows'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Typing :: Typed',
    ],
    test_suite='tests',
    python_requires='>=3.7',
    install_requires=[
        'Nano-Utils>=2.3.1',
        'numpy>=1.15.0',
        'scipy>=0.19.1',
        'pandas>=0.24.0',
        'AssertionLib>=2.3',
        'noodles>=0.3.3',
        'more-itertools>=1.0',
        'plams>=1.5.1',
        'qmflows>=0.11.1',
        'nlesc-CAT>=0.11.1',
        'Auto-FOX>=0.10.0',
        'rdkit-pypi>=2018.03.1',
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'flake8',
    ],
    extras_require={
        'test': [
            'pyyaml>=5.1',
            'pytest',
            'pytest-cov',
            'pytest-mock',
        ],
    }
)
