#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:44:04 2020

@author: felix
"""

from brian2 import *
import buildtools as build

def visualise_connectivity(S):
    figure(figsize=(10,10))
    if type(S) == Synapses:
        S = [S]
    for syn in S:
        weight = syn.weight if hasattr(syn, 'weight') else syn.namespace['weight']
        scatter(syn.x_pre, syn.x_post, weight/nS)
        xlabel(syn.source.name + ' x')
        ylabel(syn.target.name + ' x')
    title(S[0].name)

def raster(monitors, ax = None):
    total = 0
    ticks = [0]
    labels = ['']
    if ax == None:
        figure()
        fig, ax = subplots()
    if not iterable(monitors):
        monitors = [monitors]
    for m in monitors:
        ax.plot(m.t/ms, m.i + total, '.k')
        ticks.append(total + m.source.N/2)
        labels.append(m.source.name)
        total += m.source.N
        ticks.append(total)
        labels.append('')
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

def trace_plots(monitors, variable = 'V', unit = mV, offset = 10 * mV):
    for m in monitors:
        figure()
        title(m.source.name)
        for i, mview in enumerate(m):
            plot(m.t / ms, (i*offset + getattr(mview, variable)) / unit)


#%% Test tools

def check_neuron_models(pops, Itest = 1*nA, tpre = 50*ms, tpost=50*ms, ttest=100*ms, extra_elems = []):
    traces, spikes = [],[]
    input_spikes = SpikeMonitor(pops['T'])
    groups = [g for key,g in pops.items() if key != 'T']
    for g in groups:
        traces.append(StateMonitor(g, 'V', record = 0))
        spikes.append(SpikeMonitor(g))

    Net = Network(pops['T'], *groups, *traces, *spikes, input_spikes, *extra_elems)
    Net.run(tpre)
    for g in groups:
        g.I = Itest
    Net.run(ttest)
    for g in groups:
        g.I = 0*nA
    Net.run(tpost)

    figure()
    plot(input_spikes.t/ms, input_spikes.i, '.k')
    xlabel('time (ms)')
    ylabel('input neuron #')

    for g, trace, spike in zip(groups, traces, spikes):
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

def check_stp(M, tag):
    freq = [5, 10, 20, 40, 80, 160] # Hz
    recovery = [30, 100, 300, 1000, 3000, 10000] # ms
    nspikes = 10
    nf, nr = len(freq), len(recovery)
    n = nf*nr

    target = build.build_neuron(M.pops[tag.split(':')[1]], n)

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

    syn = build.build_synapse(sg, target, M.syns[tag], connect=False)
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

def check_stdp(M, tag):
    model = '''
g_ampa: siemens
g_gaba: siemens
'''
    Probe = NeuronGroup(1000, model, threshold = 't > i*ms', refractory=1*second)
    ProbeSyn = build.build_synapse(Probe, Probe, M.syns[tag], connect=False)
    ProbeSyn.connect(j='500')
    ProbeSyn.w_stdp = 1

    probemon = StateMonitor(ProbeSyn, 'w_stdp', True)

    N = Network(Probe, ProbeSyn, probemon)
    N.run(1*second)

    figure(figsize=(15,10))
    subplot(221)
    title(tag + ' STDP weight change, pre before post')
    xlabel('time (s)')
    for w in probemon.w_stdp[:501]:
        plot(probemon.t, w-1)

    subplot(222)
    title(tag + ' STDP weight change, post before pre')
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
