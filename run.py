#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:39:32 2020

@author: felix
"""


from brian2 import *
import brian2genn
from model import *
from buildtools import *
from runtools import *
set_device('genn')

pops = build_populations()
poisson = add_poisson(*pops)
synapses = build_network(*pops)
monitors = [SpikeMonitor(g) for g in pops]

N = Network(*pops, *poisson, *synapses, *monitors)
N.run(60*second)

fig = figure(figsize=(20,5))
ax = fig.subplots()
raster(monitors, ax)
