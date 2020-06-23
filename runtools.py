#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:44:04 2020

@author: felix
"""

from brian2 import *

def visualise_connectivity(S):
    figure(figsize=(10,10))
    if type(S) == Synapses:
        S = [S]
    for syn in S:
        weight = syn.weight if hasattr(syn, 'weight') else syn.namespace['weight']
        scatter(syn.x_pre, syn.x_post, weight/nS)
        xlabel(syn.source.name + ' x')
        ylabel(syn.target.name + ' x')

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