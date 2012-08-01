from distutils.core import setup, Extension

gaussian_error_func = Extension('ptf/analyze/gaussian',
                                sources=['ptf/analyze/gaussian.c'],
                                include_dirs=['/scr4/dlevitan/sw/epd-7.1-1-rh5-x86_64/lib/python2.7/site-packages/numpy/core/include'])

setup(name='PTF',
      version='1.0',
      description="adrn's PTF project",
      ext_modules=[gaussian_error_func])