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
        'nanoCAT.recipes'
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
        'python-3-6',
        'python-3-7',
        'python-3-8',
        'automation',
        'scientific-workflows'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU Lesser General Public License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    test_suite='tests',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'pandas>=0.24.0',
        'AssertionLib>=2.1',
        'noodles',
        'plams@git+https://github.com/SCM-NV/PLAMS@a5696ce62c09153a9fa67b2b03a750913e1d0924',
        'qmflows@git+https://github.com/SCM-NV/qmflows@master',
        'CAT@git+https://github.com/nlesc-nano/CAT@master',
        'Auto-FOX@git+https://github.com/nlesc-nano/auto-FOX@master'
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle'
    ],
    extras_require={
        'test': ['pytest',
                 'pytest-cov',
                 'pytest-mock',
                 'pycodestyle'],
        'doc': ['sphinx', 'sphinx_rtd_theme', 'sphinx-autodoc-typehints']
    }
)
