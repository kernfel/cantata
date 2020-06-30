#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 2020

@author: felix

"""

#%% Startup
from brian2 import *

#%% Definitions

# ================ Basic LIF ===================================
LIF = Equations('''
dV/dt = ((Vrest-V) + (Isyn + I)/gL) / tau : volt (unless refractory)
I : amp
''')

simple_synapse = Equations('''
I = g*(E_syn - V) : amp
dg/dt = -g/tau_syn : siemens
''')

LIF_eqn = (LIF +
           simple_synapse.substitute(g='g_ampa', I = 'I_ampa', E_syn = 'E_ampa', tau_syn = 'tau_ampa') +
           simple_synapse.substitute(g='g_gaba', I = 'I_gaba', E_syn = 'E_gaba', tau_syn = 'tau_gaba') +
           'Isyn = I_ampa + I_gaba: amp')

LIF_defaults = {
    'gL': 5 * nS,       # Leak conductance
    'Vrest': -60 * mV,  # Resting potential
    'tau': 20 * ms,     # Membrane time constant

    'E_ampa': 0 * mV,   # AMPA reversal potential
    'tau_ampa': 5 * ms, # AMPA time constant

    'E_gaba': -80 * mV, # GABA reversal potential
    'tau_gaba': 10 * ms,# GABA time constant

    'threshold': -50 * mV,  # Spike threshold
    'refractory': 2 * ms,   # Refractory period
    'poisson_N': 1000,          # Number of poisson inputs per neuron

    'poisson_rate': 10 * Hz,     # Firing rate of each input spike train
    'poisson_weight': 0.02 * nS,# Weight of poisson inputs

    '_LIF': {
        'build': {
            'model': LIF_eqn,
            'method': 'euler',
            'threshold': 'V > threshold',
            'reset': 'V = Vrest',
            'refractory': 'refractory'
        },
        'init': {
            'V': 'Vrest'
        }
    }
}

# ========================= Neuron with 1D location x ========================
loc_defaults = {
    '_1D_location': {
        'build': {
            'model': 'x : 1'
        },
        'init': {
            'x': 'i * 1.0 / (N-1)'
        }
    }
}

# ================ Synapse with variable weight ==============================
weighted_synapse_defaults = {
    '_syn_weighted': {
        'build': {'model': Equations('weight: siemens')},
        'weight': ['weight']
    }
}

# ================ Spike-timing dependent plasticity =========================

# STDP in line with Vogels et al., 2011

STDP_eqn = Equations('''
w_stdp : 1
dapre/dt = -apre/taupre : 1 (event-driven)
dapost/dt = -apost/taupost : 1 (event-driven)
''')
STDP_onpre = '''
apre += 1
w_stdp = clip(w_stdp + apost*etapost, wmin, wmax)
'''
STDP_onpost = '''
apost += 1
w_stdp = clip(w_stdp + (apre - alpha)*etapre, wmin, wmax)
'''

STDP_defaults = {
    'taupre': 10 * ms,  # pre before post time constant
    'taupost': 10 * ms, # post before pre time constant
    'etapre': 1e-2,    # pre before post learning rate
    'etapost': -1e-2,  # post before pre learning rate
    'wmin': 0,          # Min weight factor
    'wmax': 2,          # Max weight factor
    'alpha': 0.2,       # Depression factor

    '_syn_STDP': {
        'build': {
            'model': STDP_eqn,
            'on_pre': STDP_onpre,
            'on_post': STDP_onpost
        },
        'init': {
            'w_stdp': 1
        },
        'weight': ['w_stdp']
    }
}

# ================ Short-term plasticity ===================================

varela_DD_eqn = Equations('''
ddf/dt = (1-df)/tauDf : 1 (event-driven)
dds/dt = (1-ds)/tauDs : 1 (event-driven)
''')
varela_DD_onpre = '''
df *= Df
ds *= Ds
'''

varela_DD_defaults = { # From Kudela et al., 2018
    'Df': 0.46,             # Fast depression factor [0..1]
    'tauDf': 38 * ms,       # Fast depression recovery time constant
    'Ds': 0.76,             # Slow depression factor [0..1]
    'tauDs': 9.2 * second,  # Slow depression recovery time constant

    '_syn_varela_DD': {
        'build': {
            'model': varela_DD_eqn,
            'on_pre': varela_DD_onpre
        },
        'init': {
            'df': 1,
            'ds': 1
        },
        'weight': ['df*ds']
    }
}

#%% Neurons

params = dict()
params = {
    'scale': 100
}

# ================ Thalamus ===================================
params_T = {**params, **loc_defaults}
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
params_E = {**params, **LIF_defaults, **loc_defaults}
params_E['gL'] = 10 * nS
params_E['tau'] = 20 * ms
params_E['_'] = {'build': {'name': 'Excitatory', 'N': params['scale']}}

# ================= I =========================================
params_I = {**params, **LIF_defaults, **loc_defaults}
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
params_EE = {**weighted_synapse_defaults, **STDP_defaults, **varela_DD_defaults, **params_synapses}
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
params_II = {**weighted_synapse_defaults, **params_synapses}
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
params_EI = {**weighted_synapse_defaults, **varela_DD_defaults, **params_synapses}
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
params_IE = {**weighted_synapse_defaults, **STDP_defaults, **params_synapses}
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
