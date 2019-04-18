from setuptools import setup

import os
BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='dde',
      py_modules=['dde'],
      install_requires=[
          'torch'
      ],
)
