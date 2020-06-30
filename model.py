#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 2020

@author: felix

"""

#%% Startup
from brian2 import *
import defaults

#%% Neurons

params = dict()
params = {
    'scale': 100
}

# ================ Thalamus ===================================
params_T = {**defaults.localised_neuron, **params}
params_T['max_rate'] = 50 * Hz
params_T['width'] = 0.2 # affects rate

params_T['period'] = 2 * second
params_T['_'] = {
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
params_E = {**defaults.LIF, **defaults.localised_neuron, **params}
params_E['gL'] = 10 * nS
params_E['tau'] = 20 * ms
params_E['_'] = {'build': {'name': 'Excitatory', 'N': params['scale']}}

# ================= I =========================================
params_I = {**defaults.LIF, **defaults.localised_neuron, **params}
params_I['gL'] = 6 * nS
params_I['tau'] = 20 * ms
params_I['_'] = {'build': {'name': 'Inhibitory', 'N': 0.2*params['scale']}}


#%% Network: Cortex

params_synapses = params.copy()
params_synapses['delay_per_oct'] = 5 * ms # per full patch width
params_synapses['delay_k0'] = 0.1 # radius of local neighborhood
params_synapses['delay_f'] = 2 # distance scaling factor for higher delay steps
delay_eqn = 'delay_per_oct * abs(x_pre-x_post)'

# ================= EE =========================================
params_EE = {**defaults.weighted_synapse, **defaults.STDP, **defaults.varela_DD, **params_synapses}
params_EE['gbar'] = 3 * nS
params_EE['width'] = 0.1 # affects weight

params_EE['_'] = {
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
params_II = {**defaults.weighted_synapse, **params_synapses}
params_II['gbar'] = 5 * nS
params_II['width'] = 0.1 # affects weight

params_II['_'] = {
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
params_EI = {**defaults.weighted_synapse, **defaults.varela_DD, **params_synapses}
params_EI['gbar'] = 5 * nS
params_EI['width'] = 0.2 # affects weight

params_EI['_'] = {
    'build': {'on_pre': 'g_ampa_post += {weight}'},
    'connect': {
        'maxdist': 0.5,
        'p': 0.6,
        'distribution': 'normal',
        'sigma': 0.2 },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}

# ================= IE =========================================
params_IE = {**defaults.weighted_synapse, **defaults.STDP, **params_synapses}
params_IE['gbar'] = 5 * nS
params_IE['width'] = 0.2 # affects weight
params_IE['etapost'] =  params_IE['etapre']

params_IE['_'] = {
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
params_TE = params_synapses.copy()
params_TE['weight'] = 1.1 * nS
params_TE['delay_per_oct'] = 0 * ms
params_TE['delay_k0'] = 1

params_TE['_'] = {
    'build': {'on_pre': 'g_ampa_post += weight'},
    'connect': {
        'distribution': 'normal',
        'sigma': 0.1,
    }
}

# ================= TI =========================================
params_TI = params_synapses.copy()
params_TI['weight'] = 0.5 * nS
params_TI['delay_per_oct'] = 0 * ms
params_TI['delay_k0'] = 1

params_TI['_'] = {
    'build': {'on_pre': 'g_ampa_post += weight'},
    'connect': {
        'distribution': 'normal',
        'sigma': 0.15,
    }
}
