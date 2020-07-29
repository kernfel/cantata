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
    'scale': 1000
}

#%% Neurons

pops = {}

# ================ Thalamus ===================================
pops['T'] = {**defaults.localised_neuron, **params}
pops['T']['max_rate'] = 200 * Hz
pops['T']['spectral_width'] = 0.02
pops['T']['temporal_width'] = 75 * ms
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
        'N': 0.2 * params['scale']
    },
    'run_regularly': {
        'code': '''
t0 = t + latency
r0 = max_rate * exp(-(frequency(t) - x)**2 / (2*spectral_width**2))
''',
        'dt': 500 * ms
    }
}


#%% Cortex, stage 1

# ======================== Layer 2/3 =========================================
pops['S1_sustainer_e'] = {**defaults.LIF, **defaults.localised_neuron, **params}
pops['S1_sustainer_e']['poisson_N'] = 750
pops['S1_sustainer_e']['poisson_weight'] = .05 * nS
pops['S1_sustainer_e']['gL'] = 10 * nS
pops['S1_sustainer_e']['tau'] = 20 * ms
pops['S1_sustainer_e']['_'] = {'build': {'name': 'S1_sustainer_e', 'N': params['scale']}}

pops['S1_sustainer_i'] = {**defaults.LIF, **defaults.localised_neuron, **params}
pops['S1_sustainer_i']['poisson_N'] = 500
pops['S1_sustainer_i']['poisson_weight'] = .05 * nS
pops['S1_sustainer_i']['gL'] = 6 * nS
pops['S1_sustainer_i']['tau'] = 20 * ms
pops['S1_sustainer_i']['_'] = {'build': {'name': 'S1_sustainer_i', 'N': 0.4*params['scale']}}

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
    'sigma': 0.2
}
connect_narrow = {
    'autapses': False,
    'maxdist': 0.4,
    'distribution': 'normal',
    'sigma': 0.1
}
p_e = .1
p_i = .4

params_synapses = params.copy()
params_synapses['delay_per_oct'] = 5 * ms # per full patch width
params_synapses['delay_k0'] = 0.1 # radius of local neighborhood
params_synapses['delay_f'] = 2 # distance scaling factor for higher delay steps

params_synapses['df_mean'] = .5
params_synapses['df_std'] = .02
params_synapses['ds_mean'] = .2
params_synapses['ds_std'] = .1

syns = {}

# ================= EE =========================================
syns['S1_sustainer_e:S1_sustainer_e'] = { # **defaults.STDP,
                                         **defaults.weighted_synapse,
                                         **defaults.varela_DD,
                                         **params_synapses}
syns['S1_sustainer_e:S1_sustainer_e']['gbar'] = 1.5 * nS
syns['S1_sustainer_e:S1_sustainer_e']['transmitter'] = 'ampa'

syns['S1_sustainer_e:S1_sustainer_e']['_'] = {
    'init': {'weight': 'gbar',
             'df': 'abs(randn()*df_std + df_mean)',
             'ds': 'abs(randn()*ds_std + ds_mean)'},
    'connect': {
        **connect_narrow,
        'p': p_e }
}

syns['S1_decoder_e:S1_decoder_e'] = copy.deepcopy(syns['S1_sustainer_e:S1_sustainer_e'])
syns['S1_decoder_e:S1_decoder_e']['gbar'] = 1*nS

syns['S1_decoder_e:S1_sustainer_e'] = copy.deepcopy(syns['S1_sustainer_e:S1_sustainer_e'])

# ================= II =========================================
defaults_ii = {**defaults.weighted_synapse,
               **params_synapses}
defaults_ii['gbar'] = 1.5 * nS
defaults_ii['transmitter'] = 'gaba'
defaults_ii['_'] = {
    'init': {'weight': 'gbar'},
    'connect': {
        **connect_narrow,
        'p': p_i }
}

syns['S1_sustainer_i:S1_sustainer_i'] = copy.deepcopy(defaults_ii)
syns['S1_sustainer_i:S1_sustainer_i']['_']['connect']['p'] = 1.5*p_i

syns['S1_sustainer_i:S1_decoder_i'] = copy.deepcopy(defaults_ii)

syns['S1_decoder_i:S1_decoder_i'] = copy.deepcopy(defaults_ii)
syns['S1_decoder_i:S1_decoder_i']['_']['connect'] = {
    **connect_wide,
    'p': p_i,
    'peak': 0.4,
    'maxdist': 1 }

# ================= EI =========================================
syns['S1_sustainer_e:S1_sustainer_i'] = {**defaults.weighted_synapse,
                                         **defaults.varela_DD,
                                         **params_synapses}
syns['S1_sustainer_e:S1_sustainer_i']['gbar'] = 1.5 * nS
syns['S1_sustainer_e:S1_sustainer_i']['transmitter'] = 'ampa'

syns['S1_sustainer_e:S1_sustainer_i']['_'] = {
    'init': {'weight': 'gbar',
             'df': 'abs(randn()*df_std + df_mean)',
             'ds': 'abs(randn()*ds_std + ds_mean)'},
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
                                         **defaults.weighted_synapse,
                                         **params_synapses}
syns['S1_sustainer_i:S1_sustainer_e']['gbar'] = 1.5 * nS
syns['S1_sustainer_i:S1_sustainer_e']['transmitter'] = 'gaba'
# syns['S1_sustainer_i:S1_sustainer_e']['etapost'] =  syns['S1_sustainer_i:S1_sustainer_e']['etapre']

syns['S1_sustainer_i:S1_sustainer_e']['_'] = {
    'init': {'weight': 'gbar'},
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
syns['T:S1_decoder_e']['gbar'] = 1 * nS
syns['T:S1_decoder_e']['width'] = 0.05
syns['T:S1_decoder_e']['delay_per_oct'] = 0 * ms
syns['T:S1_decoder_e']['delay_k0'] = 1
syns['T:S1_decoder_e']['transmitter'] = 'ampa'

syns['T:S1_decoder_e']['_'] = {
    'connect': {
        'distribution': 'normal',
        'p': 0.4,
        'sigma': 0.05,
    },
    'init': {'weight': 'gbar * exp(-(x_pre-x_post)**2/(2*width**2))'}
}
