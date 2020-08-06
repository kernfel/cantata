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
        setattr(G, k, get_init(v))
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

def build_synapse(source, target, params, connect = True, stepped_delays = True, instr_out = None):
    name = '{source}_to_{target}{{0}}'.format(source=source.name, target=target.name)
    instr = instructions(copy.deepcopy(params))
    if instr_out != None:
        instr_out.update(instr)
    if not connect:
        return Synapses(source, target, namespace = params, **instr['build'], name=name.format(''))

    bands = get_connectivity(source, target, params, instr['connect'], stepped_delays)
    syns = []
    for i, band in enumerate(bands):
        syn = Synapses(source, target, **instr['build'], namespace = params,
                       delay=band['delay'], name=name.format('_{0}'.format(i) if i else ''))
        syn.connect(i = band['i'], j = band['j'])
        for k, v in instr['init'].items():
            setattr(syn, k, get_init(v))
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
            if len(i) > 0:
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

class Prio_Table:
    def __init__(self, appending, name = ''):
        self.table = {}
        self.append = appending
        self.name = name

    def add(self, key, value, warn = False, context = ''):
        key, prio = self.get_prio(key)
        if prio not in self.table:
            self.table[prio] = {key: value}
        elif key in self.table[prio]:
            if self.append:
                self.table[prio][key] = \
                    self.table[prio][key]+ '\n' + value
            else:
                if warn and key in self.table[prio]:
                    print("Warning: Replacing {k}:{oldv} with {newv} from {context} in {name}".format(
                        k=key, oldv=self.table[prio][key], newv=value,
                        context=context, name=self.name))
                self.table[prio][key] = value
        else:
            self.table[prio][key] = value

    def ensure_type(self, key, a_type):
        for p, d in self.table.items():
            if key in d and type(d[key]) != a_type:
                d[key] = a_type(d[key])

    def get(self, warn = False, context = 'priority'):
        if len(self.table) == 0:
            return self.table
        table = self.table
        self.table = {}
        for p in sorted(table):
            for k,v in table[p].items():
                self.add(k, v, warn, context)
        collapsed = self.table[0]
        self.table = table
        return collapsed

    @staticmethod
    def get_prio(key):
        # TODO numeric priority
        if key.startswith('[priority]'):
            key = key[10:].lstrip()
            prio = -1
        elif key.startswith('[defer]'):
            key = key[7:].lstrip()
            prio = 1
        else:
            prio = 0
        return key, prio

def instructions(p):
    build, init = Prio_Table(True, 'build'), Prio_Table(False, 'init')
    instr = {}
    for key, value in p.items():
        if key.startswith('_') and type(value) == dict:
            for ke, va in value.items():
                if ke == 'build':
                    for k, v in va.items():
                        if type(v) == Equations:
                            build.ensure_type(k, Equations)
                        build.add(k, v, True, key)
                elif ke == 'connect':
                    if 'connect' in instr:
                        print('Warning: connect spec discarded: {}'.format(instr['connect']))
                    instr['connect'] = va
                    instr['connect']['__key__'] = key
                elif ke == 'init':
                    for k, v in va.items():
                        init.add(k, v, True, key)
                elif ke in instr:
                    instr[ke] += va
                else:
                    instr[ke] = va
    instr['build'] = build.get(True)
    instr['init'] = init.get(True)

    repl = {}
    repl['weight'] = '*'.join(instr['weight']) if 'weight' in instr else '1'
    repl['transmitter'] = p['transmitter'] if 'transmitter' in p else ''
    for key, value in instr['build'].items():
        if type(value) == str:
            instr['build'][key] = value.format(**repl)
    return instr

def v(d):
    return list(d.values())

def get_init(statement):
    if type(statement) != dict:
        return statement
    if statement.get('type', 'rand') == 'uniform':
        out = '{min} + rand()*({max}-{min})'
    else:
        if statement.get('type', 'rand') == 'rand' or statement.get('type') == 'normal':
            out = '{mean} + randn()*{sigma}'
        elif statement.get('type') == 'distance':
            distance = '{x}_pre-{x}_post'.format(x=statement.get('distvar', 'x'))
            out = '{mean} * exp(-(' + distance + ')**2/(2*{sigma}**2))'
        else:
            raise Exception('Unknown init statement type in {}'.format(statement))

        if 'min' in statement and 'max' in statement:
            out = 'clip(' + out + ', {min}, {max})'
        elif 'min' in statement:
            out = 'clip(' + out + ', {min}, inf)'
        elif 'max' in statement:
            out = 'clip(' + out + ', -inf, {max})'
    if 'unit' in statement:
        out = '(' + out + ')*{unit}'
    try:
        out = out.format(**statement)
    except KeyError:
        raise Exception('Failed to parse init statement {}'.format(statement))
    return out
