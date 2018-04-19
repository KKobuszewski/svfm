# Superfluid Vortex Filament Model numerical solver.

This repository consists software for numerical simulation of single superfluid vortex in presence of impurities.
A columnar vortex approximation was assumed. For further details got to .pdf file in `thesis` directory.


# Requirements

The code was tested on Ubuntu 16.04 operational system.

Requirements:
* Python 2.7
* numpy 1.9.1
* scipy 0.19
* FFTW library installed in $PATH.


# Compilation

The code was written in CPython with C extensions.
To compile the extensions go to `python` directory and type:

`python setup.py build_ext`
