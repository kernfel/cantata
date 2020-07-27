#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:07:13 2020

@author: felix

Attempt to replicate the model posited in [1]. Other than scaling and tuning,
the following fundamental changes are made:
    - Spiking neuron populations replace mean-field reductions
    - Tonotopic gradient is contiguous, not discrete
    - Synapse weights are independent of distance; but connection probability
      approximately replicates the proposed scheme

[1] A. Tabas, G. Mihai, S. Kiebel, R. Trampel, and K. von Kriegstein,
 ‘Priors based on abstract rules modulate the encoding of pure tones in the subcortical auditory pathway’,
 presented at the 29th Annual Computational Neuroscience Meeting, Jul. 2020.

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
pops['T']['max_rate'] = 200 * Hz
pops['T']['spectral_width'] = 0.02
pops['T']['temporal_width'] = 15 * ms
pops['T']['latency'] = 15 * ms
pops['T']['base_rate'] = 1 * Hz
pops['T']['frequency'] = TimedArray(np.random.rand(500), dt = 250 * ms)

pops['T']['_'] = {
    'build': {
        'model': Equations('''
rates = base_rate + r0 * exp(-(t - t0)**2 / (2*temporal_width**2)) : Hz
r0: Hz
t0: second (shared)
'''),
        'threshold': 'rand() < rates*dt',
        'name': 'Thalamus',
        'N': 50
    },
    'run_regularly': {
        'code': '''
t0 = t + latency
r0 = max_rate * exp(-(frequency(t) - x)**2 / (2*spectral_width**2))
''',
        'dt': 250 * ms
    }
}


#%% Cortex, stage 1

# ======================== Layer 2/3 =========================================
pops['S1_sustainer_e'] = {**defaults.LIF, **defaults.localised_neuron, **params}
pops['S1_sustainer_e']['gL'] = 10 * nS
pops['S1_sustainer_e']['tau'] = 20 * ms
pops['S1_sustainer_e']['_'] = {'build': {'name': 'S1_sustainer_e', 'N': params['scale']}}

pops['S1_sustainer_i'] = {**defaults.LIF, **defaults.localised_neuron, **params}
pops['S1_sustainer_i']['gL'] = 6 * nS
pops['S1_sustainer_i']['tau'] = 20 * ms
pops['S1_sustainer_i']['_'] = {'build': {'name': 'S1_sustainer_i', 'N': 0.2*params['scale']}}

# ======================== Layer 5 ===========================================
pops['S1_decoder_e'] = copy.deepcopy(pops['S1_sustainer_e'])
pops['S1_decoder_e']['_']['build']['name'] = 'S1_decoder_e'

pops['S1_decoder_i'] = copy.deepcopy(pops['S1_sustainer_i'])
pops['S1_decoder_i']['_']['build']['name'] = 'S1_decoder_i'


#%% Network: Cortex, stage 1

connect_wide = {
    'autapses': False,
    'maxdist': 0.6,
    'distribution': 'normal',
    'sigma': 0.3
}
connect_narrow = {
    'autapses': False,
    'maxdist': 0.4,
    'distribution': 'normal',
    'sigma': 0.1
}
p_e = 0.6
p_i = 0.8

params_synapses = params.copy()
params_synapses['delay_per_oct'] = 5 * ms # per full patch width
params_synapses['delay_k0'] = 0.1 # radius of local neighborhood
params_synapses['delay_f'] = 2 # distance scaling factor for higher delay steps

syns = {}

# ================= EE =========================================
syns['S1_sustainer_e:S1_sustainer_e'] = { # **defaults.STDP,
                                         **defaults.varela_DD,
                                         **params_synapses}
syns['S1_sustainer_e:S1_sustainer_e']['weight'] = 3 * nS
syns['S1_sustainer_e:S1_sustainer_e']['transmitter'] = 'ampa'

syns['S1_sustainer_e:S1_sustainer_e']['_'] = {
    'build': {'on_pre': 'g_{transmitter}_post += weight'},
    'connect': {
        **connect_narrow,
        'p': p_e }
}

syns['S1_decoder_e:S1_decoder_e'] = copy.deepcopy(syns['S1_sustainer_e:S1_sustainer_e'])
syns['S1_decoder_e:S1_sustainer_e'] = copy.deepcopy(syns['S1_sustainer_e:S1_sustainer_e'])

# ================= II =========================================
syns['S1_sustainer_i:S1_sustainer_i'] = {**params_synapses}
syns['S1_sustainer_i:S1_sustainer_i']['weight'] = 5 * nS
syns['S1_sustainer_i:S1_sustainer_i']['transmitter'] = 'gaba'

syns['S1_sustainer_i:S1_sustainer_i']['_'] = {
    'build': {'on_pre': 'g_{transmitter}_post += weight'},
    'connect': {
        **connect_narrow,
        'p': p_i }
}

syns['S1_sustainer_i:S1_decoder_i'] = copy.deepcopy(syns['S1_sustainer_i:S1_sustainer_i'])

syns['S1_decoder_i:S1_decoder_i'] = copy.deepcopy(syns['S1_sustainer_i:S1_sustainer_i'])
syns['S1_decoder_i:S1_decoder_i']['_']['connect'] = {
    **connect_wide,
    'p': p_i,
    'peak': 0.2 }

# ================= EI =========================================
syns['S1_sustainer_e:S1_sustainer_i'] = {**defaults.varela_DD,
                                         **params_synapses}
syns['S1_sustainer_e:S1_sustainer_i']['weight'] = 5 * nS
syns['S1_sustainer_e:S1_sustainer_i']['transmitter'] = 'ampa'

syns['S1_sustainer_e:S1_sustainer_i']['_'] = {
    'build': {'on_pre': 'g_{transmitter}_post += weight'},
    'connect': {
        **connect_narrow,
        'p': p_e }
}

syns['S1_decoder_e:S1_decoder_i'] = copy.deepcopy(syns['S1_sustainer_e:S1_sustainer_i'])
syns['S1_decoder_e:S1_decoder_i']['_']['connect'] = {
    **connect_wide,
    'p': p_e }

# ================= IE =========================================
syns['S1_sustainer_i:S1_sustainer_e'] = {# **defaults.STDP,
                                         **params_synapses}
syns['S1_sustainer_i:S1_sustainer_e']['weight'] = 5 * nS
syns['S1_sustainer_i:S1_sustainer_e']['transmitter'] = 'gaba'
# syns['S1_sustainer_i:S1_sustainer_e']['etapost'] =  syns['S1_sustainer_i:S1_sustainer_e']['etapre']

syns['S1_sustainer_i:S1_sustainer_e']['_'] = {
    'build': {'on_pre': 'g_{transmitter}_post += weight'},
    'connect': {
        **connect_narrow,
        'p': p_i }
}

syns['S1_decoder_i:S1_decoder_e'] = copy.deepcopy(syns['S1_sustainer_i:S1_sustainer_e'])
syns['S1_decoder_i:S1_decoder_e']['_']['connect'] = {
    **connect_wide,
    'mindist': 0.1,
    'peak': 0.2,
    'p': p_i }

#%% Network: Thalamo-cortical

# ================= TE =========================================
syns['T:S1_decoder_e'] = {**defaults.weighted_synapse,
                       **params_synapses}
syns['T:S1_decoder_e']['gbar'] = 1.1 * nS
syns['T:S1_decoder_e']['width'] = 0.1
syns['T:S1_decoder_e']['delay_per_oct'] = 0 * ms
syns['T:S1_decoder_e']['delay_k0'] = 1
syns['T:S1_decoder_e']['transmitter'] = 'ampa'

syns['T:S1_decoder_e']['_'] = {
    'connect': {
        'distribution': 'normal',
        'sigma': 0.05,
    },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}
