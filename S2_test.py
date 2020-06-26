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

def check_stdp(syn_params):
    model = '''
g_ampa: siemens
g_gaba: siemens
'''
    Probe = NeuronGroup(1000, model, threshold = 't > i*ms', refractory=1*second)
    ProbeSyn = build_synapse(Probe, Probe, syn_params, connect=False)
    ProbeSyn.connect(j='500')
    ProbeSyn.w_stdp = 1

    probemon = StateMonitor(ProbeSyn, 'w_stdp', True)

    N = Network(Probe, ProbeSyn, probemon)
    N.run(1*second)

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
check_stdp(params_EE)
check_stdp(params_IE)

#%% Short-term plasticity check

def check_stp(syn_params, target):
    freq = [5, 10, 20, 40, 80, 160] # Hz
    recovery = [30, 100, 300, 1000, 3000, 10000] # ms
    nspikes = 10
    nf, nr = len(freq), len(recovery)
    n = nf*nr

    if target.N < n:
        print("Warning: Too few neurons for complete test ({0} < {1})".format(target.N,n))

    # for k in range(nr):
    #     for j in range(nf):
    #         for i in range(nspikes):
    #             idx = j*nr + k
    #             t = i * 1000/freq[j] * ms
    #         recprobe_idx = j*nr + k
    #         recprobe_t = ((nspikes-1)*1000/freq[j] + recovery[k]) * ms
    indices = [j*nr + k for i in range(nspikes) for j in range(nf) for k in range(nr)] \
            + [j*nr + k for j in range(nf) for k in range(nr)]
    times = [i * 1000./f * ms for i in range(nspikes) for f in freq for k in range(nr)] \
          + [((nspikes-1)*1000./f + r) * ms for f in freq for r in recovery]
    sg = SpikeGeneratorGroup(len(freq)*len(recovery), indices, times)

    syn = build_synapse(sg, target, syn_params, connect=False)
    syn.connect('i==j')
    if hasattr(syn, 'weight'):
        syn.weight = 1 * psiemens
    if hasattr(syn, 'w_stdp'):
        syn.w_stdp = 1
    if hasattr(syn, 'df'):
        syn.df = 1
    if hasattr(syn, 'ds'):
        syn.ds = 1
    mon = StateMonitor(target, ['g_ampa', 'g_gaba'], range(n))

    N = Network(target, sg, mon, syn)
    N.run(max(times) + 20*ms)

    figure(figsize=(12, 3*nf))
    for j in range(nf):
        tsplit = int(((nspikes-1)*1000/freq[j] + 20) * ms / mon.clock.dt)
        # tmax = int(((nspikes-1)*1000/freq[j] + max(recovery) + 20) * ms / mon.clock.dt)
        tmax = int((max(times) + 20*ms) / mon.clock.dt)
        subplot(nf, 2, 2*j+1)
        for k in range(nr):
            plot(mon.t[:tsplit]/ms, mon.g_ampa[j*nr + k][:tsplit]/psiemens)
            plot(mon.t[:tsplit]/ms, -mon.g_gaba[j*nr + k][:tsplit]/psiemens)
        subplot(nf, 2, 2*j+2)
        for k in range(nr):
            plot(mon.t[:tmax]/ms, mon.g_ampa[j*nr + k][:tmax]/psiemens)
            plot(mon.t[:tmax]/ms, -mon.g_gaba[j*nr + k][:tmax]/psiemens)

print("STP check")
check_stp(params_TE, build_E())
check_stp(params_TI, build_I())
check_stp(params_EE, build_E())
check_stp(params_EI, build_I())
check_stp(params_IE, build_E())
check_stp(params_II, build_I())


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
