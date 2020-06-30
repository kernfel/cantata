#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:06:11 2020

@author: felix
"""

#%% Imports
from brian2 import *
from model import *
import copy

#%% Building the populations

def build_populations():
    return build_T(), \
           build_E(), \
           build_I()

def add_poisson(T, E, I):
    poisson_E = PoissonInput(E, 'g_ampa', params_E['poisson_N'],
                             params_E['poisson_rate'], params_E['poisson_weight'])
    poisson_I = PoissonInput(I, 'g_ampa', params_I['poisson_N'],
                             params_I['poisson_rate'], params_I['poisson_weight'])

    return poisson_E, poisson_I


#%% Building the network

def build_network(T, E, I, stepped_delays = True):
    return (
        build_synapse(T, E, params_TE, stepped_delays = stepped_delays),
        build_synapse(T, I, params_TI, stepped_delays = stepped_delays),
        build_synapse(E, E, params_EE, stepped_delays = stepped_delays),
        build_synapse(E, I, params_EI, stepped_delays = stepped_delays),
        build_synapse(I, E, params_IE, stepped_delays = stepped_delays),
        build_synapse(I, I, params_II, stepped_delays = stepped_delays)
        )

#%% Helper functions
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
        syns.append(syn)
    if not stepped_delays and connect:
        syns[0].delay = delay_eqn
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
        n_bands = 1 + ceil(log(conn['maxdist']/params['delay_k0'])/log(params['delay_f']))
        hbounds = params['delay_k0'] * params['delay_f'] ** arange(n_bands)
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

def instructions(p):
    instr = {'build':{}, 'init':{}}
    for key, value in p.items():
        if key.startswith('_') and type(value) == dict:
            for ke, va in value.items():
                if ke == 'build':
                    for k, v in va.items():
                        if k in instr['build']:
                            instr['build'][k] = instr['build'][k] + '\n' + v
                        else:
                            instr['build'][k] = v
                elif ke == 'connect':
                    if 'connect' in instr:
                        # NYI
                        print('Warning: connect spec discarded: {key}:{va}'.format(key=key,va=va))
                    else:
                        instr['connect'] = va
                elif ke == 'init':
                    for k, v in va.items():
                        if k in instr['init']:
                            print('Warning: Duplicate init statement {k}={v} discarded'.format(k=k,v=v))
                        else:
                            instr['init'][k] = v
                elif ke in instr:
                    instr[ke] += va
                else:
                    instr[ke] = va

    weight = '*'.join(instr['weight']) if 'weight' in instr else '1'
    for key, value in instr['build'].items():
        if type(value) == str:
            instr['build'][key] = value.format(weight=weight)

    return instr