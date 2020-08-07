#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:06:11 2020

@author: felix
"""

#%% Imports
from brian2 import *
import copy
from priotable import Prio_Table

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
                                banded_delays = banded_delays)
            for key, params in model.syns.items()}

def build_synapse(source, target, params, connect = True, banded_delays = True, instr_out = None):
    '''
    Build, and optionally connect, a `Synapse` object

    Parameters
    ----------
    source : `NeuronGroup`
    target : `NeuronGroup`
    params : dict
        `Synapse` parameters, including the synapse namespace (i.e., its actual
        parameters), and instructions on how to build the synapse. Instruction
        keys must start with an underscore and are evaluated in the order
        provided. Each instruction must be a dict and may contain any of the
        following elements:
        'build' : dict containing kwargs to `Synapses`, excluding
                  source, target, namespace, delay, name.
                  String entries such as 'on_pre' may contain a template token
                  '{weight}', which is substituted with the weight expression,
                  see below.
                  Further, they may contain a template token {transmitter}, which
                  is replaced with params['transmitter'].
                  Keys may specify numeric order priority as 'key @ priority',
                  and colliding entries are concatenated in order.
                  For more details on ordering, see `Prio_Table`.
        'weight' : list of str. All weight entries are joined with '*' to
                   constitute the weight expression, which is substituted
                   into the merged 'build' dict.
        'connect' : dict, see `get_connectivity` for details.
        'init' : dict of init statements as described in `get_init`.
                 Keys may specify numeric order priority as 'key @ priority',
                 and colliding entries are replaced in order.
                 For more details on ordering, see `Prio_Table`.
        'priority' : int. This is added to the key-defined order to yield
                     the ordering of build and init entries.
    connect : bool, optional
        Whether to connect the `Synapse`. The default is True.
    banded_delays : bool, optional
        Controls delay banding to allow heterogeneous delays in GeNN. Delay is
        computed as ``delay_per_oct * abs(dist)``, where ``dist`` is either
        ``x_pre-x_post`` (no banding) or the upper boundary of a band.
        Band boundaries are defined as ``delay_k0 * delay_f**k``, where k=0,..
        is the band index. The default is True.
    instr_out : dict, optional
        If provided, instr_out will be populated with the merged instructions.
        The default is None.

    Returns
    -------
    syns : list
        List containing the `Synapse` object(s), with one object for each delay
        band, if applicable

    See also
    --------
    instructions

    '''
    name = '{source}_to_{target}{{0}}'.format(source=source.name, target=target.name)
    instr = instructions(copy.deepcopy(params))
    if instr_out != None:
        instr_out.update(instr)
    if not connect:
        return Synapses(source, target, namespace = params, **instr['build'], name=name.format(''))

    bands = get_connectivity(source, target, params, instr['connect'], banded_delays)
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
    if not banded_delays and connect:
        syns[0].delay = 'delay_per_oct * abs(x_pre-x_post)'
    return syns

def get_connectivity(source, target, params, conn, banded_delays):
    '''
    Get a list of connection specifications, including pre- and postsynaptic
    indices i and j as well as the delay value for each band.

    Parameters
    ----------
    source : `NeuronGroup`
    target : `NeuronGroup`
    params : `dict`
        synapse parameters; see `build_synapse` for general details.
        Should contain namespace entries `delay_per_oct`, `delay_k0`, `delay_f`,
        which otherwise default to 0, 1, and 2 respectively. See the comment on
        banded_delays in `build_synapse` for details on delays.
    conn: dict
        'connect' instructions provided e.g. from `instructions`. The following
        keys are recognised:
            'autapses' : bool, optional, defaults to True
                Whether to allow autapses when `source`==`target`
            'mindist' : float, optional, defaults to 0.
                Minimal projection distance
            'maxdist' : float, optional, defaults to 1.
                Maximal projection distance
            'p' : float, optional, defaults to 1.
                Base connection probability
            'distribution': str, optional, defaults to 'unif'
                If 'unif*', the distribution is uniform with probability p.
                If 'norm*', the distribution is distance-dependent with
                probability ``p * N(dist-peak, sigma)``, ``dist = abs(x_post-x_pre)``
                regardless of delay bands
            'peak' : float, optional, defaults to ``mindist``
            'sigma' : float, optional, defaults to ``maxdist-mindist``
    banded_delays : bool
        Whether to separate the specifications into delay bands. See the comment
        on banded_delays in `build_synapse` for details. If False, the returned
        delay will be None.

    Raises
    ------
    RuntimeError
        If an unknown distribution is passed through `conn`.

    Returns
    -------
    list
        A list of dict(i=array, j=array, delay=[`Quantity` or None]), where i and j
        are the pre- and postsynaptic indices, respectively, and delay is the
        associated delay time. As noted above, delay=None if banded_delays=False.

    '''
    if 'autapses' not in conn: conn['autapses'] = True
    if 'mindist' not in conn: conn['mindist'] = 0
    if 'maxdist' not in conn: conn['maxdist'] = 1
    if 'p' not in conn: conn['p'] = 1
    if 'distribution' not in conn: conn['distribution'] = 'unif'
    if 'peak' not in conn: conn['peak'] = conn['mindist']
    if 'sigma' not in conn: conn['sigma'] = conn['maxdist']-conn['mindist']
    delay_per_oct = params.get('delay_per_oct', 0)
    delay_k0 = params.get('delay_k0', 1)
    delay_f = params.get('delay_f', 2)
    Ni, Nj = source.N, target.N

    if banded_delays and delay_per_oct > 0:
        if conn['maxdist'] <= delay_k0:
            n_bands = 1.
        else:
            n_bands = 1 + ceil(log(conn['maxdist']/delay_k0)/log(delay_f))
        hbounds = delay_k0 * delay_f ** arange(n_bands)
        hbounds[-1] = conn['maxdist']
        valid = hbounds > conn['mindist']
        hbounds = hbounds[nonzero(valid)[0]]
        n_bands = len(hbounds)
        lbounds = array([conn['mindist']] + hbounds[:-1].tolist())
        delays = hbounds * delay_per_oct
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
            if len(i) > 0:
                ret.append({'i': i, 'j': j, 'delay': delay})
    return ret

#%% Generic helpers

def instructions(p):
    '''
    Merges instructions into a single instructions dict. Every entry whose
    key in p starts with '_' is treated as a separate instruction. Entries
    are processed in the order they appear in p. Instruction items are generally
    concatenated with +=, except for those with the following keys:
        'build': Merged through `Prio_Table`, appending
        'connect': Replaced without regard for priority
        'init': Merged through `Prio_Table`, replacing

    Priority can be specified in instruction entry keys as well as through
    the instructions' 'priority' entry. Both are optional and default to 0.

    Resulting string entries in 'build' are formatted with the following
    template parameters:
        {weight} : The inserted value is all instructions['weight'] string-joined
                   by '*'.
        {transmitter} : The inserted value is p['transmitter']

    Parameters
    ----------
    p : dict
        A full parameter namespace

    Returns
    -------
    instr : dict
        The processed instructions dictionary.

    '''
    build, init = Prio_Table(True, 'build'), Prio_Table(False, 'init')
    instr = {}
    for key, value in p.items():
        if key.startswith('_') and type(value) == dict:
            priority = value.get('priority', 0)
            for ke, va in value.items():
                if ke == 'build':
                    for k, v in va.items():
                        if type(v) == Equations:
                            build.ensure_type(k, Equations)
                        build.add(k, v, True, key, priority)
                elif ke == 'connect':
                    if 'connect' in instr:
                        print('Warning: connect spec discarded: {}'.format(instr['connect']))
                    instr['connect'] = va
                    instr['connect']['__key__'] = key
                elif ke == 'init':
                    for k, v in va.items():
                        init.add(k, v, True, key, priority)
                elif ke in instr:
                    instr[ke] += va
                elif ke != 'priority':
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
    '''
    Produce a valid brian2 variable init value

    Parameters
    ----------
    statement : dict
        If not a dict, the statement is returned without processing.
        Otherwise, the statement should specify the following:
            type : str, defaults to 'rand'
                   One of 'rand', 'normal', 'uniform', 'distance'.
                   - 'uniform' draws random values from a uniform distribution
                       between 'min' and 'max'
                   - 'rand', 'normal' draws random values from a normal
                       distribution ``N(mean, sigma)``
                   - 'distance' produces deterministic values based on a
                       distance-dependent normal distribution
                       ``mean * N(dist, sigma)``, where dist is defined as
                       ``abs({distvar}_pre - {distvar}_post)``
            min, max : Optional, except for `uniform` type.
                       Smallest and largest permitted value, respectively.
            mean, sigma : Required for types other than `uniform`.
            distvar : Optional for type `distance`, defaults to 'x'
            unit : str, optional
                   String representation of the brian2 unit to use (e.g. 'psiemens')

    Returns
    -------
    str
        Processed statement that can be passed to NeuronGroup or Synapse variables.

    '''
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
    except Exception as e:
        e.args = ("Failed to parse init statement {}".format(statement),) + e.args
        raise
    return out
