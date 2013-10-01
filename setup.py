from distutils.core import setup, Extension
from Cython.Distutils import build_ext

# Get numpy path
import os, numpy
numpy_base_path = os.path.split(numpy.__file__)[0]
numpy_incl_path = os.path.join(numpy_base_path, "core", "include")

error_func = Extension('ptf/analyze/error_functions',
                        sources=['ptf/analyze/error_functions.c'],
                        include_dirs=[numpy_incl_path])

aov = Extension('ptf.analyze.aov',
                sources=['ptf/analyze/aov.pyx'],
                include_dirs=[numpy_incl_path])

setup(name='PTF',
      version='2.0',
      description="adrn's PTF project",
      cmdclass={'build_ext': build_ext},
      ext_modules=[error_func, aov])

# Now generate various data files:
from ptf.util import cache_all_fields
cache_all_fields()