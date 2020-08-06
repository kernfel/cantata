#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:08:36 2020

@author: felix
"""

#%% Startup
from brian2 import *

#%% Neuron models

# ================ Basic LIF ===================================
LIF_base = Equations('''
dV/dt = ((Vrest-V) + (Isyn + I)/gL) / tau : volt (unless refractory)
I : amp
''')

simple_synapse = Equations('''
I = g*(E_syn - V) : amp
dg/dt = -g/tau_syn : siemens
''')

LIF_eqn = (LIF_base +
           simple_synapse.substitute(g='g_ampa', I = 'I_ampa', E_syn = 'E_ampa', tau_syn = 'tau_ampa') +
           simple_synapse.substitute(g='g_gaba', I = 'I_gaba', E_syn = 'E_gaba', tau_syn = 'tau_gaba') +
           'Isyn = I_ampa + I_gaba: amp')

LIF = {
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
            'V': 'Vrest + rand()*(threshold-Vrest)'
        }
    }
}

# ========================= Neuron with 1D location x ========================
localised_neuron = {
    '_1D_location': {
        'build': {
            'model': 'x : 1'
        },
        'init': {
            'x': 'i * 1.0 / (N-1)'
        }
    }
}

#%% Synapse models

# ================ Synapse with variable weight ==============================
weighted_synapse = {
    'transmitter': 'ampa',
    '_weighted': {
        'build': {
            'model': Equations('weight: siemens'),
            'on_pre': 'g_{transmitter}_post += {weight}'},
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

STDP = {
    'taupre': 10 * ms,  # pre before post time constant
    'taupost': 10 * ms, # post before pre time constant
    'etapre': 1e-2,    # pre before post learning rate
    'etapost': -1e-2,  # post before pre learning rate
    'wmin': 0,          # Min weight factor
    'wmax': 2,          # Max weight factor
    'alpha': 0.2,       # Depression factor

    '_STDP': {
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

varela_DD = { # From Kudela et al., 2018
    'Df': 0.46,             # Fast depression factor [0..1]
    'tauDf': 38 * ms,       # Fast depression recovery time constant
    'Ds': 0.76,             # Slow depression factor [0..1]
    'tauDs': 9.2 * second,  # Slow depression recovery time constant

    '_varela_DD': {
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

# =============== Tsodyks-Markram ===========================================
# Following:
#    Misha Tsodyks, A. Uziel, and H. Markram,
#       ‘t Synchrony Generation in Recurrent Networks with Frequency-Dependent Synapses’,
#       J. Neurosci., vol. 20, no. 1, pp. RC50–RC50, Jan. 2000,
#       doi: 10.1523/JNEUROSCI.20-01-j0003.2000.
# cf. https://github.com/nest/nest-simulator/blob/v2.20.0/models/tsodyks_connection.h
#
# Caution: Potentially unstable with tau_psc ~= tau_rec,
#          ensure tau_rec > tau_psc to avoid issues.

tsodyks_eqn = Equations('''
ts_x: 1
ts_y: 1
lastupdate: second

U: 1
tau_rec: second
''')
tsodyks_fac_eqn = tsodyks_eqn + Equations('''
ts_u: 1
tau_fac: second
''')

tsodyks_template_onpre_early = '''
T = t-lastupdate

yy = exp(-T/tau_psc)
zz = exp(-T/tau_rec)
xy = ((zz-1.0) * tau_rec - (yy-1.0) * tau_psc) / (tau_psc - tau_rec)
xz = 1.0 - zz
ts_z = 1.0 - ts_x - ts_y;

{u_update}
ts_x += xy*ts_y + xz*ts_z
ts_y *= yy
'''
# compute synaptic output based on ts_x at this point, then:
tsodyks_template_onpre_late = '''
ts_x -= {delta}
ts_y += {delta}

lastupdate = t
'''

tsodyks_delta = 'U * ts_x'
tsodyks = {
    'tau_psc': 5*ms, # Formally, this should be equal to the transmitter tau.
    '_tsodyks': {
        'build': {
            'model': tsodyks_eqn,
            '[priority] on_pre': tsodyks_template_onpre_early.format(
                u_update=''),
            '[defer] on_pre': tsodyks_template_onpre_late.format(
                delta=tsodyks_delta)
        },
        'init': {
            'ts_x': 1,
            'ts_y': 0,
            'lastupdate': 0*ms
        },
        'weight': [tsodyks_delta]
    }
}

tsodyks_fac_u_update = 'ts_u = ts_u * exp(-T/tau_fac) + U * (1-ts_u)'
tsodyks_fac_delta = 'ts_u * ts_x'
tsodyks_fac = {
    'tau_psc': 5*ms, # Formally, this should be equal to the transmitter tau.
    '_tsodyks': {
        'build': {
            'model': tsodyks_fac_eqn,
            '[priority] on_pre': tsodyks_template_onpre_early.format(
                u_update = tsodyks_fac_u_update),
            '[defer] on_pre': tsodyks_template_onpre_late.format(
                delta=tsodyks_fac_delta)
        },
        'init': {
            '[defer] ts_u': 'U',
            'ts_x': 1,
            'ts_y': 0,
            'lastupdate': 0*ms
        },
        'weight': [tsodyks_fac_delta]
    }
}
