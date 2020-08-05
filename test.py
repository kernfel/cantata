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
import tabas as M

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

visualise_circuit(M, synapses)
for S in synapses.values():
    visualise_connectivity(S)

#%% STDP check

print("STDP check")
for key, value in M.syns.items():
    if '_STDP' in value:
        check_stdp(M, key)

#%% Short-term plasticity check

print("STP check")
for key, value in M.syns.items():
    if '_varela_DD' in value or '_tsodyks' in value:
        check_stp(M, key)


#%% Function check
print('Starting function check...')

start_scope()
pops = build_populations(M)
poisson = add_poisson(pops)
synapses = build_network(M, pops)
monitors = [SpikeMonitor(g) for key, g in pops.items()]
# tracers = [StateMonitor(g, 'V', range(5)) for key, g in pops.items() if hasattr(g, 'V')]
# wtrace = [StateMonitor(g, 'w_stdp', True) for g in synapses if hasattr(g, 'w_stdp')]

N = Network(v(pops), v(poisson), v(synapses), *monitors)
runtime = 10*second
N.run(runtime)

raster(monitors)
# trace_plots(tracers)
# trace_plots(wtrace, variable='w_stdp', unit=1, offset=0)

##%% Diagnostics


for m in monitors:
    print(m.source.name, m.num_spikes / runtime / m.source.N)
