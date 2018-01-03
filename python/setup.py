from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

# get the annotated file as well
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

ext_modules = [
    Extension('cfunctions',
              sources            = ['cfunctions.pyx'],
              include_dirs       = [numpy.get_include(),'.'],
              language           = "c++",                                                                                         # generate C++ code
              extra_compile_args = ['-fopenmp','-pthread','-fPIC','-mtune=native','-march=native','-O3','-falign-functions=64','-fext-numeric-literals'], #'-std=c99',
              extra_link_args    = ['-fopenmp','-pthread'],
              libraries          = ['fftw3','gomp'],
              library_dirs       = ['/usr/local/lib']),
    
]
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)

import os
import shutil

srcfile = 'build/lib.linux-x86_64-2.7/cfunctions.so'
dstfile = './cfunctions.so'

assert not os.path.isabs(srcfile)
shutil.copy(srcfile, dstfile)

