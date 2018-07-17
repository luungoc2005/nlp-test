from distutils.core import setup
from Cython.Build import cythonize
# from distutils.extension import Extension

import numpy

setup(
    name="Botbot-NLP",
    ext_modules=cythonize([
        # "common/_cutils/*.pyx",
        "text_classification/fast_text/_cutils/*.pyx"
    ]
        , include_path=[
            numpy.get_include()
        ]
    ),
)
