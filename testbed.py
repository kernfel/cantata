#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:33:42 2020

@author: felix
"""

#%%
from brian2 import *
from scaffold1 import *

#%% Neuron model check
def check_neuron_models(groups, Itest = 1*nA, tpre = 50*ms, tpost=50*ms, ttest=100*ms):
    traces, spikes = [],[]
    input_spikes = SpikeMonitor(groups[0])
    for g in groups[1:]:
        traces.append(StateMonitor(g, 'V', record = 0))
        spikes.append(SpikeMonitor(g))
    
    Net = Network(*groups, *traces, *spikes, input_spikes)
    Net.run(tpre)
    for g in groups[1:]:
        g.I[0] = Itest
    Net.run(ttest)
    for g in groups[1:]:
        g.I = 0*nA
    Net.run(tpost)
    
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
        
        npre, npost = sum(tspike<=tpre), sum(tspike>tpre+ttest)
        print(g.name, 'frequencies pre, test, post:',
              npre / tpre,
              (len(tspike)-npre-npost)/ttest,
              npost / tpost)
    

start_scope()
pops = build_populations()
check_neuron_models(pops)

#%% Network check

def visualise_connectivity(S):
    figure(figsize=(10,10))
    weight = S.weight if hasattr(S, 'weight') else S.namespace['weight']
    scatter(S.x_pre, S.x_post, weight/nS)
    xlabel(S.source.name + ' x')
    ylabel(S.target.name + ' x')

start_scope()
pops = build_populations()
synapses = build_network(*pops)

for S in synapses:
    visualise_connectivity(S)

#%% Function check

def raster(monitors):
    total = 0
    ticks = []
    labels = []
    for m in monitors:
        plot(m.t/ms, m.i + total, '.k')
        ticks.append(total + m.source.N/2)
        labels.append(m.source.name)
        total += m.source.N
        ticks.append(total)
        labels.append('')
    yticks(ticks, labels)

start_scope()
pops = build_populations()
synapses = build_network(*pops)
monitors = [SpikeMonitor(g) for g in pops]

N = Network(*pops, *synapses, *monitors)
N.run(5*second)

raster(monitors)
