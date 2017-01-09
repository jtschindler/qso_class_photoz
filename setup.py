#!/usr/bin/env python

from distutils.core import setup, Command

try:  # Python 3.x
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from distutils.command.build_py import build_py

setup(name='class_photoz',
      version='0.1.1',
      description='Machine Learning and Photometric Fitting for phot. redshift estimation and candidate classification',
      author='Jan-Torge Schindler',
      author_email='jtschindler@email.arizona.edu,',
      license='GPL',
      url='',
      packages=['class_photoz'],
      provides=['class_photoz'],
      package_data={'class_photoz':['data/*.*']},
      requires=['numpy', 'matplotlib','scipy','astropy','sklearn','pandas'],
      keywords=['Scientific/Engineering'],
     )

