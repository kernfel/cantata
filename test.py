#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 2020

@author: felix
"""

#%%
from brian2 import *
from buildtools import *
from runtools import *
import model as M

#%% Perform model check
print("Model check (current)")
start_scope()
pops = build_populations(M)
check_neuron_models(pops)

#%% Perform model check (with noise inputs)
print("Model check (+noise)")
start_scope()
pops = build_populations(M)
inputs = add_poisson(pops)
check_neuron_models(pops, extra_elems = v(inputs), Itest = 0*nA, ttest=10000*ms)

#%% Network check

print("Connectivity check")
start_scope()
pops = build_populations(M)
synapses = build_network(M, pops)

for S in synapses.values():
    visualise_connectivity(S)

#%% STDP check

print("STDP check")
check_stdp(M, 'E:E')
check_stdp(M, 'I:E')

#%% Short-term plasticity check

print("STP check")
check_stp(M, 'E:E')
check_stp(M, 'E:I')


#%% Function check
print('Starting function check...')

start_scope()
pops = build_populations(M)
poisson = add_poisson(pops)
synapses = build_network(M, pops)
monitors = [SpikeMonitor(g) for key, g in pops.items()]
tracers = [StateMonitor(g, 'V', range(5)) for key, g in pops.items() if hasattr(g, 'V')]
# wtrace = [StateMonitor(g, 'w_stdp', True) for g in synapses if hasattr(g, 'w_stdp')]

N = Network(v(pops), v(poisson), v(synapses), *monitors, *tracers)
N.run(5000*ms)

raster(monitors)
trace_plots(tracers)
# trace_plots(wtrace, variable='w_stdp', unit=1, offset=0)
