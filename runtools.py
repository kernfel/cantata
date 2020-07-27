#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:44:04 2020

@author: felix
"""

from brian2 import *
import buildtools as build
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost

#%% Visualisation

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

def get_sorted_tags(M):
    parts = [key.split('_') for key in M.pops]

    stages = [p[0] for p in parts]
    stages = list(set(stages))
    stages.sort()

    layers = [p[1] for p in parts if len(p) > 1]
    layers = list(set(layers))
    layers.sort()

    ctypes = [p[2] for p in parts if len(p) > 2]
    ctypes = list(set(ctypes))
    ctypes.sort()

    i = 0
    tags, sax, lax, tax = [], [], [], []
    for s in stages:
        si = i
        if s in M.pops:
            tags.append(s)
            i += 1
        for l in layers:
            li = i
            tag = '{0}_{1}'.format(s,l)
            if tag in M.pops:
                tags.append(tag)
                i += 1
            for t in ctypes:
                tag = '{0}_{1}_{2}'.format(s,l,t)
                if tag in M.pops:
                    tags.append(tag)
                    i += 1
                    tax.append((i, t))
            if i > li:
                lax.append((i, l))
        if i > si:
            sax.append((i, s))
    return tags, sax, lax, tax

# only once: Create connectivity colormap
inh = plt.cm.get_cmap('Oranges_r', 128)
exc = plt.cm.get_cmap('Blues', 128)
merge = vstack((inh(linspace(0, 1, 128)),
                exc(linspace(0, 1, 128))))
conn_cmap = ListedColormap(merge, name='OrangeBlue')

def plot_connectivity_matrix(data, tags, sax, lax, tax, fig = figure()):
    ax1 = SubplotHost(fig, 111)
    fig.add_subplot(ax1)

    ax1.imshow(data, cmap=conn_cmap, vmin=-1, vmax=1, origin='lower')

    # First X axis
    tticks, tlabels = zip(*tax)
    ax1.set_xticks(array(tticks)-1)
    ax1.set_xticklabels(tlabels)
    ax1.set_yticks(array(tticks)-1)
    ax1.set_yticklabels(tlabels)

    # Extra X axes
    configure_twin_axis(ax1.twiny(), 'bottom', (0,-25), lax, len(tags))
    configure_twin_axis(ax1.twiny(), 'bottom', (0,-50), sax, len(tags))

    # Extra Y axes
    configure_twin_axis(ax1.twinx(), 'left', (-25,0), lax, len(tags))
    configure_twin_axis(ax1.twinx(), 'left', (-50,0), sax, len(tags))

    ax1.axis['top'].set_visible(True)
    ax1.axis['right'].set_visible(True)
    ax1.axis['top'].major_ticks.set_ticksize(0)
    ax1.axis['right'].major_ticks.set_ticksize(0)

    ax1.text(-0.7, -1.4, 'source', rotation='90')
    ax1.text(-1.4, -0.7, 'target')

def configure_twin_axis(ax, location, offset, ticktuples, length):
    new_axisline = ax.get_grid_helper().new_fixed_axis
    ax.axis[location] = new_axisline(loc=location, axes=ax, offset=offset)
    for k,v in ax.axis.items():
        v.set_visible(False)
    ax.axis[location].set_visible(True)
    limb = ax.xaxis if location in ['top', 'bottom'] else ax.yaxis

    ticks, labels = zip(*ticktuples)
    ticks = [0] + list(ticks)
    locs = ticks[:-1] + diff(array(ticks))/2
    limb.set_major_locator(ticker.FixedLocator(array(ticks)/length))
    limb.set_major_formatter(ticker.NullFormatter())
    limb.set_minor_locator(ticker.FixedLocator(locs/length))
    limb.set_minor_formatter(ticker.FixedFormatter(labels))

    ax.axis[location].minor_ticks.set_ticksize(0)

def visualise_circuit(M):
    tags, sax, lax, tax = get_sorted_tags(M)

    data = zeros((len(tags), len(tags)))
    for syn, p in M.syns.items():
        source, target = syn.split(':')
        if p['transmitter'] == 'gaba':
            value = -1
        elif p['transmitter'] == 'ampa':
            value = 1
        data[tags.index(target), tags.index(source)] = value

    fig = figure(figsize=(5,5))
    plot_connectivity_matrix(data, tags, sax, lax, tax, fig)


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
