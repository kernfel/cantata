#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 2020

@author: felix

"""

#%% Startup
from brian2 import *
import defaults

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

# ================= E =========================================
pops['E'] = {**defaults.LIF, **defaults.localised_neuron, **params}
pops['E']['gL'] = 10 * nS
pops['E']['tau'] = 20 * ms
pops['E']['_'] = {'build': {'name': 'Excitatory', 'N': params['scale']}}

# ================= I =========================================
pops['I'] = {**defaults.LIF, **defaults.localised_neuron, **params}
pops['I']['gL'] = 6 * nS
pops['I']['tau'] = 20 * ms
pops['I']['_'] = {'build': {'name': 'Inhibitory', 'N': 0.2*params['scale']}}


#%% Network: Cortex

params_synapses = params.copy()
params_synapses['delay_per_oct'] = 5 * ms # per full patch width
params_synapses['delay_k0'] = 0.1 # radius of local neighborhood
params_synapses['delay_f'] = 2 # distance scaling factor for higher delay steps

syns = {}

# ================= EE =========================================
syns['E:E'] = {**defaults.weighted_synapse, **defaults.STDP, **defaults.varela_DD, **params_synapses}
syns['E:E']['gbar'] = 3 * nS
syns['E:E']['width'] = 0.1 # affects weight

syns['E:E']['_'] = {
    'build': {'on_pre': 'g_ampa_post += {weight}'},
    'connect': {
        'autapses': False,
        'maxdist': 0.5,
        'p': 0.6,
        'distribution': 'normal',
        'sigma': 0.2 },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}

# ================= II =========================================
syns['I:I'] = {**defaults.weighted_synapse, **params_synapses}
syns['I:I']['gbar'] = 5 * nS
syns['I:I']['width'] = 0.1 # affects weight

syns['I:I']['_'] = {
    'build': {'on_pre': 'g_gaba_post += {weight}'},
    'connect': {
        'autapses': False,
        'maxdist': 0.3,
        'p': 0.8,
        'distribution': 'normal',
        'sigma': 0.1 },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}

# ================= EI =========================================
syns['E:I'] = {**defaults.weighted_synapse, **defaults.varela_DD, **params_synapses}
syns['E:I']['gbar'] = 5 * nS
syns['E:I']['width'] = 0.2 # affects weight

syns['E:I']['_'] = {
    'build': {'on_pre': 'g_ampa_post += {weight}'},
    'connect': {
        'maxdist': 0.5,
        'p': 0.6,
        'distribution': 'normal',
        'sigma': 0.2 },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}

# ================= IE =========================================
syns['I:E'] = {**defaults.weighted_synapse, **defaults.STDP, **params_synapses}
syns['I:E']['gbar'] = 5 * nS
syns['I:E']['width'] = 0.2 # affects weight
syns['I:E']['etapost'] =  syns['I:E']['etapre']

syns['I:E']['_'] = {
    'build': {'on_pre': 'g_gaba_post += {weight}'},
    'connect': {
        'maxdist': 0.3,
        'p': 0.9,
        'distribution': 'normal',
        'sigma': 0.1 },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}

#%% Network: Thalamo-cortical

# ================= TE =========================================
syns['T:E'] = params_synapses.copy()
syns['T:E']['weight'] = 1.1 * nS
syns['T:E']['delay_per_oct'] = 0 * ms
syns['T:E']['delay_k0'] = 1

syns['T:E']['_'] = {
    'build': {'on_pre': 'g_ampa_post += weight'},
    'connect': {
        'distribution': 'normal',
        'sigma': 0.1,
    }
}

# ================= TI =========================================
syns['T:I'] = params_synapses.copy()
syns['T:I']['weight'] = 0.5 * nS
syns['T:I']['delay_per_oct'] = 0 * ms
syns['T:I']['delay_k0'] = 1

syns['T:I']['_'] = {
    'build': {'on_pre': 'g_ampa_post += weight'},
    'connect': {
        'distribution': 'normal',
        'sigma': 0.15,
    }
}
