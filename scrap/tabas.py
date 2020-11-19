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
def generate_timed_array(_, p):
    p['frequency'] = TimedArray(np.random.rand(500), dt = 2500 * ms)
    p['frequency'].values[0] = -100 # avoid input during initial settling

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
    },
    'run_initially': generate_timed_array
}


#%% Cortex, stage 1

default_e = {**defaults.LIF, **defaults.localised_neuron, **params}
default_e['poisson_N'] = 900
default_e['poisson_weight'] = .04 * nS
default_e['gL'] = 10 * nS
default_e['tau'] = 20 * ms
default_e['_'] = {'build': {'N': params['scale']}}

default_i = {**defaults.LIF, **defaults.localised_neuron, **params}
default_i['poisson_N'] = 500
default_i['poisson_weight'] = .04 * nS
default_i['gL'] = 6 * nS
default_i['tau'] = 20 * ms
default_i['_'] = {'build': {'N': 0.4*params['scale']}}

# ======================== Layer 2/3 =========================================
pops['S1_sustainer_e'] = copy.deepcopy(default_e)
pops['S1_sustainer_e']['_']['build']['name'] = 'S1_sustainer_e'

pops['S1_sustainer_i'] = copy.deepcopy(default_i)
pops['S1_sustainer_i']['_']['build']['name'] = 'S1_sustainer_i'
pops['S1_sustainer_i']['poisson_weight'] = 0.05 * nS

# ======================== Layer 5 ===========================================
pops['S1_decoder_e'] = copy.deepcopy(default_e)
pops['S1_decoder_e']['_']['build']['name'] = 'S1_decoder_e'

pops['S1_decoder_i'] = copy.deepcopy(default_i)
pops['S1_decoder_i']['_']['build']['name'] = 'S1_decoder_i'


#%% Network: Cortex, stage 1

connect_wide = {
    'autapses': False,
    'maxdist': 0.4,
    'distribution': 'normal',
    'sigma': 0.05
}
connect_narrow = {
    'autapses': False,
    'maxdist': 0.1,
    'distribution': 'normal',
    'sigma': 0.03
}

p_e = .1
p_i = .4

params_synapses = params.copy()
params_synapses['delay_per_oct'] = 5 * ms # per full patch width
params_synapses['delay_k0'] = 0.04 # radius of local neighborhood
params_synapses['delay_f'] = 2 # distance scaling factor for higher delay steps

# E, I Tsodyks-Markram parameters as described in
#    Misha Tsodyks, A. Uziel, and H. Markram,
#       ‘t Synchrony Generation in Recurrent Networks with Frequency-Dependent Synapses’,
#       J. Neurosci., vol. 20, no. 1, pp. RC50–RC50, Jan. 2000,
#       doi: 10.1523/JNEUROSCI.20-01-j0003.2000.
params_exc = params_synapses.copy()
params_exc['tau_psc'] = defaults.LIF['tau_ampa']
params_exc['_exc'] = {
    'init': {
        'U': {'type': 'normal', 'min': 0, 'max': 1, 'mean': 0.5, 'sigma': 0.25},
        'tau_rec': {
            'type': 'normal',
            'min': 2*params_exc['tau_psc'],
            'mean': 800,
            'sigma': 400,
            'unit': 'ms'}
        }
}

params_inh = params_synapses.copy()
params_inh['tau_psc'] = defaults.LIF['tau_gaba']
params_inh['_inh'] = {
    'init': {
        'U': {'type': 'normal', 'min': 0, 'max': 1, 'mean': 0.04, 'sigma': 0.02},
        'tau_fac': {'type': 'normal', 'min': 1, 'mean': 1000, 'sigma': 500, 'unit': 'ms'},
        'tau_rec': {
            'type': 'normal',
            'min': 2*params_inh['tau_psc'],
            'mean': 100,
            'sigma': 50,
            'unit': 'ms'}
        }
}

syns = {}

# ================= EE =========================================
default_ee = { # **defaults.STDP,
               **defaults.weighted_synapse,
               **defaults.tsodyks,
               **params_exc}
default_ee['gbar'] = 5 * nS
default_ee['transmitter'] = 'ampa'
default_ee['_'] = {
    'init': {'weight': 'gbar'},
    'connect': {
        **connect_narrow,
        'p': p_e }
}

syns['S1_sustainer_e:S1_sustainer_e'] = copy.deepcopy(default_ee)
syns['S1_sustainer_e:S1_sustainer_e']['_']['connect']['p'] = 2*p_e

syns['S1_decoder_e:S1_decoder_e'] = copy.deepcopy(default_ee)
syns['S1_decoder_e:S1_decoder_e']['gbar'] = 1.5*nS

syns['S1_decoder_e:S1_sustainer_e'] = copy.deepcopy(default_ee)

# ================= II =========================================
default_ii = {**defaults.weighted_synapse,
               **defaults.tsodyks_fac,
               **params_inh}
default_ii['gbar'] = 5 * nS
default_ii['transmitter'] = 'gaba'
default_ii['_'] = {
    'init': {'weight': 'gbar'},
    'connect': {
        **connect_narrow,
        'p': p_i }
}

syns['S1_sustainer_i:S1_sustainer_i'] = copy.deepcopy(default_ii)
syns['S1_sustainer_i:S1_sustainer_i']['_']['connect']['p'] = 2*p_i

syns['S1_decoder_i:S1_sustainer_i'] = copy.deepcopy(default_ii)
syns['S1_decoder_i:S1_sustainer_i']['_']['connect']['p'] = 2*p_i
syns['S1_decoder_i:S1_sustainer_i']['gbar'] = 8*nS

syns['S1_decoder_i:S1_decoder_i'] = copy.deepcopy(default_ii)
syns['S1_decoder_i:S1_decoder_i']['_']['connect'] = {
    **connect_wide,
    'p': p_i,
    'peak': 0.1 }

# ================= EI =========================================
default_ei = {**defaults.weighted_synapse,
               **defaults.tsodyks,
               **params_exc}
default_ei['gbar'] = 5 * nS
default_ei['transmitter'] = 'ampa'
default_ei['_'] = {
    'init': {'weight': 'gbar'},
    'connect': {
        **connect_narrow,
        'p': p_e }
}

syns['S1_sustainer_e:S1_sustainer_i'] = copy.deepcopy(default_ei)
syns['S1_sustainer_e:S1_sustainer_i']['_']['connect']['p'] = 2*p_e

syns['S1_decoder_e:S1_decoder_i'] = copy.deepcopy(default_ei)
syns['S1_decoder_e:S1_decoder_i']['_']['connect'] = {
    **connect_wide,
    'p': p_e,
    'sigma': 0.1 }

# ================= IE =========================================
default_ie = {# **defaults.STDP,
               **defaults.weighted_synapse,
               **defaults.tsodyks_fac,
               **params_inh}
default_ie['gbar'] = 5 * nS
default_ie['transmitter'] = 'gaba'
# default_ie['etapost'] =  syns['S1_sustainer_i:S1_sustainer_e']['etapre']
default_ie['_'] = {
    'init': {'weight': 'gbar'},
    'connect': {
        **connect_narrow,
        'p': p_i }
}

syns['S1_sustainer_i:S1_sustainer_e'] = copy.deepcopy(default_ie)
syns['S1_sustainer_i:S1_sustainer_e']['_']['connect']['p'] = 2*p_i
syns['S1_sustainer_i:S1_sustainer_e']['gbar'] = 8*nS

syns['S1_decoder_i:S1_decoder_e'] = copy.deepcopy(default_ie)
syns['S1_decoder_i:S1_decoder_e']['_']['connect'] = {
    **connect_wide,
    'p': p_i,
    'peak': 0.1,
    'mindist': 0.01 }

#%% Network: Thalamo-cortical

# ================= TE =========================================
syns['T:S1_decoder_e'] = {**defaults.weighted_synapse,
                       **params_synapses}
syns['T:S1_decoder_e']['gbar'] = 1 * nS
syns['T:S1_decoder_e']['width'] = connect_narrow['sigma']
syns['T:S1_decoder_e']['delay_per_oct'] = 0 * ms
syns['T:S1_decoder_e']['delay_k0'] = 1
syns['T:S1_decoder_e']['transmitter'] = 'ampa'

syns['T:S1_decoder_e']['_'] = {
    'connect': {
        **connect_narrow,
        'p': 0.2
    },
    'init': {'weight': {'type': 'distance', 'mean': 'gbar', 'sigma': 'width'}},
}

syns['T:S1_decoder_i'] = copy.deepcopy(syns['T:S1_decoder_e'])
