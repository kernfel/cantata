#!/usr/bin/env python

from setuptools import setup

setup(name='cantata',
    version='0.0',
    packages=[
        'cantata',
    ],
    package_data={'cantata': [
        'configs/default.yaml',
    ]},
)
