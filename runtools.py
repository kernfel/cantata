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

def visualise_circuit(M, synapses, figsize=(12,8), **kwargs):
    indeg, outdeg, weight = get_synapse_degrees(M, synapses)

    eweight = weight.copy()
    iweight = weight.copy()
    eweight[weight<=0] = nan
    iweight[weight>=0] = nan

    fig,ax = subplots(figsize=figsize, **kwargs)
    _cmap_red = ListedColormap(plt.cm.get_cmap('Reds')(linspace(.3,1,128)))
    _cmap_green = ListedColormap(plt.cm.get_cmap('Greens')(linspace(.3,1,128)))
    exc = ax.imshow(eweight, origin = 'upper', cmap = _cmap_green)
    inh = ax.imshow(-iweight, origin = 'upper', cmap = _cmap_red)

    for i in range(len(indeg)):
        for j in range(len(indeg[0])):
            if weight[i,j] != 0:
                ax.text(j,i, '{:.1f}'.format(indeg[i,j]), ha='center', va='bottom', c='white')
                ax.text(j,i, '{:.1f}'.format(outdeg[i,j]), ha='center', va='top', c='gray')

    ebar = fig.colorbar(exc, ax=ax)
    ebar.ax.set_ylabel('Exc. weight (S)', rotation=-90, va="bottom")
    ibar = fig.colorbar(inh, ax=ax)
    ibar.ax.set_ylabel('Inh. weight (S)', rotation=-90, va="bottom")

    ticks, extras = get_hierarchical_ticks(M)
    ax.set_xticks(range(len(ticks)))
    ax.set_yticks(range(len(ticks)))
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.xaxis.tick_top()
    ax.yaxis.set_tick_params(labelrotation=90)

    xscale, yscale = diff(ax.get_window_extent().get_points(), axis=0)[0]
    xscale = -12 * len(weight) / xscale
    yscale = -12 * len(weight) / yscale
    for offset, e in enumerate(extras):
        for a in e:
            ax.text((a['begin']+a['end'])/2, -.5 + yscale * (1.5*offset+2.5), a['tag'],
                    ha = 'center', va = 'center')
            ax.annotate('', xy = (a['begin']-.5, -.5 + yscale * (1.5*offset+2)), xycoords='data',
                        xytext = (a['end']+.5, -.5 + yscale * (1.5*offset+2)), textcoords = 'data',
                        arrowprops = {'arrowstyle': '-'}, va = 'center',
                        annotation_clip = False)
            ax.text(-.5 + xscale * (1.5*offset+2.5), (a['begin']+a['end'])/2, a['tag'],
                    ha = 'center', va = 'center', rotation=90)
            ax.annotate('', xy = (-.5 + xscale * (1.5*offset+2), a['begin']-.5), xycoords='data',
                        xytext = (-.5 + xscale * (1.5*offset+2), a['end']+.5), textcoords = 'data',
                        arrowprops = {'arrowstyle': '-'}, va = 'center',
                        annotation_clip = False)
    ax.text(3*xscale-.5, .5*yscale-.5, 'source', ha='center', va='center')
    ax.text(.5*xscale-.5, 3*yscale-.5, 'target', ha='center', va='center', rotation = 90)

def get_hierarchical_ticks(M):
    tokens = [s.split('_') for s in sorted(M.pops)]
    tickmarks = [t[-1] for t in tokens]
    levels = max([len(tok) for tok in tokens])
    extras = [None]*(levels-1)
    for l in range(levels-1):
        extras[l] = []
        last = [None]*levels
        i = l+1
        for j, tok in enumerate(tokens):
            # print(tok, l, i, j, last)
            if len(tok) < i+1:
                continue
            if tok[:-i] == last[:-i]:
                extras[l][-1]['end'] = j
            else:
                extras[l].append({'begin': j, 'end': j, 'tag': tok[-i-1]})
            last = tok
    return tickmarks, extras

def get_synapse_degrees(M, synapses):
    shape = (len(M.pops), len(M.pops))
    indeg, outdeg, weight = zeros(shape), zeros(shape), zeros(shape)
    for tag, Ss in synapses.items():
        pre, post, w = array([], dtype=int32),\
                       array([], dtype=int32),\
                       array([], dtype=float64)
        for S in Ss:
            pre = concatenate((pre, S._synaptic_pre))
            post = concatenate((post, S._synaptic_post))
            w = concatenate((w, S.weight))
        i,j = [sorted(M.pops).index(k) for k in tag.split(':')]
        indeg[i,j] = mean(bincount(post))
        outdeg[i,j] = mean(bincount(pre))
        weight[i,j] = mean(w) * (-1 if Ss[0].namespace['transmitter']=='gaba' else 1)
    return indeg, outdeg, weight

def raster(monitors, ax = None):
    total = 0
    ticks, lticks = [0], []
    labels = []
    if ax == None:
        fig, ax = subplots(figsize=(20,15))
    if not iterable(monitors):
        monitors = [monitors]
    for m in monitors:
        ax.plot(m.t/ms, m.i + total, '.k')
        lticks.append(total + m.source.N/2)
        labels.append(m.source.name)
        total += m.source.N
        ticks.append(total)
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_yticks(lticks, minor=True)
    ax.set_yticklabels(labels, minor=True)
    ax.yaxis.set_tick_params(which='minor', length=0)
    ax.set_ylim(0, ticks[-1])
    for i in range(1, len(ticks)-1, 2):
        ax.axhspan(ticks[i], ticks[i+1], fc='gray', alpha=0.2)
    return ax

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

    fig, axes = subplots(nf, 2, figsize=(15, 3*nf), constrained_layout=True)
    get_tsplit = lambda f: int(((nspikes-1)*1000/f + 20) * ms / mon.clock.dt)
    tsplit_max = get_tsplit(freq[0])
    for j in range(nf):
        tsplit = get_tsplit(freq[j])
        tmax = int((max(times) + 20*ms) / mon.clock.dt)
        ax1 = subplot(nf, 2, 2*j+1)
        if j == nf-1:
            xlabel("Time after burst onset [ms]")
        ylabel("frequency = {} Hz\nPSC [pS]".format(freq[j]))
        for k in range(nr):
            plot(mon.t[:tsplit]/ms, mon.g_ampa[j*nr + k][:tsplit]/psiemens)
            plot(mon.t[:tsplit]/ms, -mon.g_gaba[j*nr + k][:tsplit]/psiemens)
        ax1.set_xlim(-10, mon.t[tsplit_max]/ms + 10)

        ax2 = subplot(nf, 2, 2*j+2, sharey = ax1)
        ax2.yaxis.tick_right()
        if j == nf-1:
            xlabel("Time after burst onset [ms]")
        for k in range(nr):
            plot(mon.t[:tmax]/ms, mon.g_ampa[j*nr + k][:tmax]/psiemens)
            plot(mon.t[:tmax]/ms, -mon.g_gaba[j*nr + k][:tmax]/psiemens)
    suptitle(tag)
    fig.set_constrained_layout_pads(wspace = 0.1, hspace = 0.1)

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

#%% Utility

def print_var_distribution(objdict, var):
    for tag, objs in objdict.items():
        mean, variance, N = 0, 0, 0
        for obj in objs:
            if hasattr(obj, var):
                n = obj.N
                mean += np.sum(getattr(obj, var))
                variance += n*np.var(getattr(obj, var))
                N += n
        if N > 0:
            print("{}\t{}\t{:.3f} +- {:.3f}".format(tag, var, mean/N, sqrt(variance/N)))
