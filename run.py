#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:39:32 2020

@author: felix
"""


from brian2 import *
import brian2genn
from buildtools import *
from runtools import *
import model as M
set_device('genn')

pops = build_populations(M)
poisson = add_poisson(pops)
synapses = build_network(M, pops)
monitors = [SpikeMonitor(g) for key, g in pops.items()]

N = Network(v(pops), v(poisson), v(synapses), *monitors)
N.run(60*second)

fig = figure(figsize=(20,5))
ax = fig.subplots()
raster(monitors, ax)
