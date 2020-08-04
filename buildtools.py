#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:06:11 2020

@author: felix
"""

#%% Imports
from brian2 import *
import copy

#%% Building the populations

def build_populations(model):
    return {key : build_neuron(params) for key, params in model.pops.items()}

def add_poisson(pops):
    return {key : PoissonInput(pop, 'g_ampa',
                               pop.namespace['poisson_N'],
                               pop.namespace['poisson_rate'],
                               pop.namespace['poisson_weight'])
            for key, pop in pops.items() if 'poisson_N' in pop.namespace}

def build_neuron(params, n = 0):
    instr = instructions(copy.deepcopy(params))
    if n > 0:
        instr['build']['N'] = n
    G = NeuronGroup(**instr['build'], namespace = params)
    for k,v in instr['init'].items():
        setattr(G, k, v)
    if 'run_regularly' in instr:
        G.run_regularly(**instr['run_regularly'])
    return G


#%% Building the network

def build_network(model, pops, banded_delays = True):
    return {key : build_synapse(pops[key.split(':')[0]],
                                pops[key.split(':')[1]],
                                params,
                                stepped_delays = banded_delays)
            for key, params in model.syns.items()}

def build_synapse(source, target, params, connect = True, stepped_delays = True):
    name = '{source}_to_{target}{{0}}'.format(source=source.name, target=target.name)
    instr = instructions(copy.deepcopy(params))
    if not connect:
        return Synapses(source, target, namespace = params, **instr['build'], name=name.format(''))

    bands = get_connectivity(source, target, params, instr['connect'], stepped_delays)
    syns = []
    for i, band in enumerate(bands):
        syn = Synapses(source, target, **instr['build'], namespace = params,
                       delay=band['delay'], name=name.format('_{0}'.format(i) if i else ''))
        syn.connect(i = band['i'], j = band['j'])
        for k, v in instr['init'].items():
            setattr(syn, k, v)
        if 'run_regularly' in instr:
            syn.run_regularly(**instr['run_regularly'])
        syns.append(syn)
    if not stepped_delays and connect:
        syns[0].delay = 'delay_per_oct * abs(x_pre-x_post)'
    return syns

def get_connectivity(source, target, params, conn, banded_delays):
    if 'autapses' not in conn: conn['autapses'] = True
    if 'mindist' not in conn: conn['mindist'] = 0
    if 'maxdist' not in conn: conn['maxdist'] = 1
    if 'p' not in conn: conn['p'] = 1
    if 'distribution' not in conn: conn['distribution'] = 'unif'
    if 'peak' not in conn: conn['peak'] = conn['mindist']
    Ni, Nj = source.N, target.N

    if banded_delays:
        if conn['maxdist'] < params['delay_k0']:
            n_bands = 1
        else:
            n_bands = 1 + ceil(log(conn['maxdist']/params['delay_k0'])/log(params['delay_f']))
        hbounds = params['delay_k0'] * params['delay_f'] ** arange(n_bands)
        hbounds[-1] = conn['maxdist']
        valid = hbounds > conn['mindist']
        hbounds = hbounds[nonzero(valid)[0]]
        n_bands = len(hbounds)
        lbounds = array([conn['mindist']] + hbounds[:-1].tolist())
        delays = hbounds * params['delay_per_oct']
    else:
        n_bands = 1
        lbounds, hbounds = [conn['mindist']], [conn['maxdist']]
        delays = [None]

    ret = []

    if conn['distribution'].startswith('unif'):
        def pd(x):
            x[:] = conn['p']
            return x
    elif conn['distribution'].startswith('norm'):
        def pd(x):
            return conn['p'] * exp(-(x-conn['peak'])**2 / (2*conn['sigma']**2))
    else:
        raise RuntimeError('Unknown distribution {d}'.format(d=conn['distribution']))

    if Nj == Ni:
        for lo,hi,delay in zip(lbounds, hbounds, delays):
            loj, hij = int(ceil(lo*Nj)), int(floor(hi*Nj))
            p = zeros(Nj)
            p[loj:hij] = pd(linspace(loj/Nj, hij/Nj, hij-loj))
            if source==target and not conn['autapses']:
                p[0] = 0

            M = zeros((Ni, Nj))
            for i in range(Ni):
                if i == 0:
                    M[i] = p
                else:
                    M[i] = np.concatenate((p[i:0:-1], p[:-i]))
            i,j = nonzero(np.random.random_sample((Ni, Nj)) < M)
            ret.append({'i': i, 'j': j, 'delay': delay})
    else:
        for lo,hi,delay in zip(lbounds, hbounds, delays):
            M = zeros((Ni, Nj))
            for i in range(Ni):
                dist = abs(linspace(0,1,Nj) - i/(Ni-1))
                idx = nonzero((dist>=lo)*(dist<hi))[0]
                M[i][idx] = pd(dist[idx])
            if lo==0 and source==target and not conn['autapses']:
                fill_diagonal(M, 0)
            i,j = nonzero(np.random.random_sample((Ni, Nj)) < M)
            ret.append({'i': i, 'j': j, 'delay': delay})
    return ret

#%% Generic helpers

def instructions(p):
    instr = {'build':{}, 'init':{}}
    for key, value in p.items():
        if key.startswith('_') and type(value) == dict:
            for ke, va in value.items():
                if ke == 'build':
                    for k, v in va.items():
                        if k in instr['build']:
                            if type(v) == Equations and type(instr['build'][k]) == str:
                                instr['build'][k] = Equations(instr['build'][k])
                            instr['build'][k] = instr['build'][k] + '\n' + v
                        else:
                            instr['build'][k] = v
                elif ke == 'connect':
                    if 'connect' in instr:
                        print('Warning: connect spec discarded: {}'.format(instr['connect']))
                    instr['connect'] = va
                    instr['connect']['__key__'] = key
                elif ke == 'init':
                    for k, v in va.items():
                        if k in instr['init']:
                            print('Warning: Overwriting init statement {}, "{}"->"{}"'.format(
                                k,instr['init'][k],v))
                        instr['init'][k] = v
                elif ke in instr:
                    instr[ke] += va
                else:
                    instr[ke] = va

    repl = {}
    repl['weight'] = '*'.join(instr['weight']) if 'weight' in instr else '1'
    repl['transmitter'] = p['transmitter'] if 'transmitter' in p else ''
    for key, value in instr['build'].items():
        if type(value) == str:
            instr['build'][key] = value.format(**repl)

    return instr

def v(d):
    return list(d.values())
