#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 2020

@author: felix

"""

#%% Startup
from brian2 import *
import defaults
import copy

params = {
    'scale': 100
}

#%% Neurons

pops = {}

# ================ Thalamus ===================================
pops['T'] = {**defaults.localised_neuron, **params}
pops['T']['max_rate'] = 50 * Hz
pops['T']['width'] = 0.2 # affects rate

pops['T']['period'] = 2 * second
pops['T']['_'] = {
    'build': {
        'model': Equations('''
rates = max_rate * exp(-alpha * (current_freq - x)**2) : Hz
current_freq = -(cos(2*pi*t/period) - 1)/2 : 1
alpha = 1/(2*width**2) : 1
'''),
        'threshold': 'rand() < rates*dt',
        'name': 'Thalamus',
        'N': 20
    }
}

#%% Cortex, stage 1

# ======================== Layer 2/3 =========================================
pops['S1_L23_Ex'] = {**defaults.LIF, **defaults.localised_neuron, **params}
pops['S1_L23_Ex']['gL'] = 10 * nS
pops['S1_L23_Ex']['tau'] = 20 * ms
pops['S1_L23_Ex']['_'] = {'build': {'name': 'S1_L23_Ex', 'N': params['scale']}}

pops['S1_L23_Inh'] = {**defaults.LIF, **defaults.localised_neuron, **params}
pops['S1_L23_Inh']['gL'] = 6 * nS
pops['S1_L23_Inh']['tau'] = 20 * ms
pops['S1_L23_Inh']['_'] = {'build': {'name': 'S1_L23_Inh', 'N': 0.2*params['scale']}}

# ======================== Layer 5 ===========================================
pops['S1_L5_Ex'] = copy.deepcopy(pops['S1_L23_Ex'])
pops['S1_L5_Ex']['_']['build']['name'] = 'S1_L5_Ex'

pops['S1_L5_Inh'] = copy.deepcopy(pops['S1_L23_Inh'])
pops['S1_L5_Inh']['_']['build']['name'] = 'S1_L5_Inh'


#%% Network: Cortex, stage 1

params_synapses = params.copy()
params_synapses['delay_per_oct'] = 5 * ms # per full patch width
params_synapses['delay_k0'] = 0.1 # radius of local neighborhood
params_synapses['delay_f'] = 2 # distance scaling factor for higher delay steps

syns = {}

# ================= EE =========================================
syns['S1_L23_Ex:S1_L23_Ex'] = {**defaults.weighted_synapse, **defaults.STDP, **defaults.varela_DD, **params_synapses}
syns['S1_L23_Ex:S1_L23_Ex']['gbar'] = 3 * nS
syns['S1_L23_Ex:S1_L23_Ex']['width'] = 0.1 # affects weight
syns['S1_L23_Ex:S1_L23_Ex']['transmitter'] = 'ampa'

syns['S1_L23_Ex:S1_L23_Ex']['_'] = {
    'connect': {
        'autapses': False,
        'maxdist': 0.5,
        'p': 0.6,
        'distribution': 'normal',
        'sigma': 0.2 },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}

syns['S1_L23_Ex:S1_L5_Ex'] = copy.deepcopy(syns['S1_L23_Ex:S1_L23_Ex'])
syns['S1_L5_Ex:S1_L23_Ex'] = copy.deepcopy(syns['S1_L23_Ex:S1_L23_Ex'])

# ================= II =========================================
syns['S1_L23_Inh:S1_L23_Inh'] = {**defaults.weighted_synapse, **params_synapses}
syns['S1_L23_Inh:S1_L23_Inh']['gbar'] = 5 * nS
syns['S1_L23_Inh:S1_L23_Inh']['width'] = 0.1 # affects weight
syns['S1_L23_Inh:S1_L23_Inh']['transmitter'] = 'gaba'

syns['S1_L23_Inh:S1_L23_Inh']['_'] = {
    'connect': {
        'autapses': False,
        'maxdist': 0.3,
        'p': 0.8,
        'distribution': 'normal',
        'sigma': 0.1 },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}

syns['S1_L5_Inh:S1_L5_Inh'] = copy.deepcopy(syns['S1_L23_Inh:S1_L23_Inh'])

# ================= EI =========================================
syns['S1_L23_Ex:S1_L23_Inh'] = {**defaults.weighted_synapse, **defaults.varela_DD, **params_synapses}
syns['S1_L23_Ex:S1_L23_Inh']['gbar'] = 5 * nS
syns['S1_L23_Ex:S1_L23_Inh']['width'] = 0.2 # affects weight
syns['S1_L23_Ex:S1_L23_Inh']['transmitter'] = 'ampa'

syns['S1_L23_Ex:S1_L23_Inh']['_'] = {
    'connect': {
        'maxdist': 0.5,
        'p': 0.6,
        'distribution': 'normal',
        'sigma': 0.2 },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}

syns['S1_L5_Ex:S1_L5_Inh'] = copy.deepcopy(syns['S1_L23_Ex:S1_L23_Inh'])

# ================= IE =========================================
syns['S1_L23_Inh:S1_L23_Ex'] = {**defaults.weighted_synapse, **defaults.STDP, **params_synapses}
syns['S1_L23_Inh:S1_L23_Ex']['gbar'] = 5 * nS
syns['S1_L23_Inh:S1_L23_Ex']['width'] = 0.2 # affects weight
syns['S1_L23_Inh:S1_L23_Ex']['transmitter'] = 'gaba'
syns['S1_L23_Inh:S1_L23_Ex']['etapost'] =  syns['S1_L23_Inh:S1_L23_Ex']['etapre']

syns['S1_L23_Inh:S1_L23_Ex']['_'] = {
    'connect': {
        'maxdist': 0.3,
        'p': 0.9,
        'distribution': 'normal',
        'sigma': 0.1 },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}

syns['S1_L5_Inh:S1_L5_Ex'] = copy.deepcopy(syns['S1_L23_Inh:S1_L23_Ex'])

#%% Network: Thalamo-cortical

# ================= TE =========================================
syns['T:S1_L23_Ex'] = params_synapses.copy()
syns['T:S1_L23_Ex']['weight'] = 1.1 * nS
syns['T:S1_L23_Ex']['delay_per_oct'] = 0 * ms
syns['T:S1_L23_Ex']['delay_k0'] = 1
syns['T:S1_L23_Ex']['transmitter'] = 'ampa'

syns['T:S1_L23_Ex']['_'] = {
    'build': {'on_pre': 'g_{transmitter}_post += weight'},
    'connect': {
        'distribution': 'normal',
        'sigma': 0.1,
    }
}

# ================= TI =========================================
syns['T:S1_L23_Inh'] = params_synapses.copy()
syns['T:S1_L23_Inh']['weight'] = 0.5 * nS
syns['T:S1_L23_Inh']['delay_per_oct'] = 0 * ms
syns['T:S1_L23_Inh']['delay_k0'] = 1
syns['T:S1_L23_Inh']['transmitter'] = 'ampa'

syns['T:S1_L23_Inh']['_'] = {
    'build': {'on_pre': 'g_{transmitter}_post += weight'},
    'connect': {
        'distribution': 'normal',
        'sigma': 0.15,
    }
}
