#!/usr/bin/env python

from setuptools import setup

setup(name='cantata',
    version='0.0',
    packages=[
        'cantata',
        'cantata.plotting',
        'cantata.training'
    ],
    package_data={'cantata': [
        'configs/default.yaml',

        'configs/embryo.yaml',

        'configs/sines.yaml'
    ]},
)
