#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 2020

@author: felix
"""

#%%
from brian2 import *
from S2 import *
from runtools import *

#%% Neuron model check
def check_neuron_models(groups, Itest = 1*nA, tpre = 50*ms, tpost=50*ms, ttest=100*ms, extra_elems = []):
    traces, spikes = [],[]
    input_spikes = SpikeMonitor(groups[0])
    for g in groups[1:]:
        traces.append(StateMonitor(g, 'V', record = 0))
        spikes.append(SpikeMonitor(g))
    
    Net = Network(*groups, *traces, *spikes, input_spikes, *extra_elems)
    Net.run(tpre)
    for g in groups[1:]:
        g.I = Itest
    Net.run(ttest)
    for g in groups[1:]:
        g.I = 0*nA
    Net.run(tpost)
    
    figure()
    plot(input_spikes.t/ms, input_spikes.i, '.k')
    xlabel('time (ms)')
    ylabel('input neuron #')
    
    for g, trace, spike in zip(groups[1:], traces, spikes):
        tspike = spike.t[spike.i==0]
        vm = trace[0].V[:]
        for t in tspike:
            i = int(t / defaultclock.dt)
            vm[i] = 20*mV
        
        figure()
        plot(trace.t / ms, vm / mV)
        xlabel('time (ms)')
        ylabel(g.name + ' V (mV)')
        
        npre, npost = sum(spike.t<=tpre), sum(spike.t>tpre+ttest)
        print(g.name, 'frequencies pre, test, post:',
              npre / tpre / g.N,
              (spike.num_spikes-npre-npost)/ttest/g.N,
              npost / tpost / g.N)
    
#%% Perform model check
print("Model check (current)")
start_scope()
pops = build_populations()
check_neuron_models(pops)
    
#%% Perform model check (with noise inputs)
print("Model check (+noise)")
start_scope()
pops = build_populations()
inputs = add_poisson(*pops)
check_neuron_models(pops, extra_elems = inputs, Itest = 0*nA, ttest=10000*ms)

#%% Network check

print("Connectivity check")
start_scope()
pops = build_populations()
synapses = build_network(*pops)

for S in synapses:
    visualise_connectivity(S)
    
#%% STDP check

def check_stdp(build_fn):
    model = '''
g_ampa: siemens
g_gaba: siemens
'''
    Probe = NeuronGroup(1000, model, threshold = 't > i*ms', refractory=1*second)
    ProbeSyn = build_fn(Probe, Probe, connect=False)
    ProbeSyn.connect(j='500')
    ProbeSyn.w_stdp = 1
    
    probemon = StateMonitor(ProbeSyn, 'w_stdp', True)
    
    run(1*second)
    
    figure(figsize=(15,10))
    subplot(221)
    title(ProbeSyn.name + ' STDP weight change, pre before post')
    xlabel('time (s)')
    for w in probemon.w_stdp[:501]:
        plot(probemon.t, w-1)
    
    subplot(222)
    title(ProbeSyn.name + ' STDP weight change, post before pre')
    xlabel('time (s)')
    for w in probemon.w_stdp[501:]:
        plot(probemon.t, w-1)
    
    subplot(223)
    xlabel('$t_{post} - t_{pre}$ (ms)')
    for i,w in enumerate(probemon.w_stdp[:501]):
        plot(i-500, w[-1]-1, 'k.')
    
    subplot(224)
    xlabel('$t_{post} - t_{pre}$ (ms)')
    for i,w in enumerate(probemon.w_stdp[501:]):
        plot(i+1, w[-1]-1, 'k.')
    
print("STDP check")
start_scope()
check_stdp(build_EE)
check_stdp(build_IE)


#%% Function check
print('Starting function check...')

start_scope()
pops = build_populations()
poisson = add_poisson(*pops)
synapses = build_network(*pops)
monitors = [SpikeMonitor(g) for g in pops]
tracers = [StateMonitor(g, 'V', range(5)) for g in pops if hasattr(g, 'V')]
# wtrace = [StateMonitor(g, 'w_stdp', True) for g in synapses if hasattr(g, 'w_stdp')]

N = Network(*pops, *poisson, *synapses, *monitors, *tracers)
N.run(5000*ms)

raster(monitors)
trace_plots(tracers)
# trace_plots(wtrace, variable='w_stdp', unit=1, offset=0)
