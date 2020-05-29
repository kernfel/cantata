#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 2020

@author: felix

Scaffold model #2:
* LIF neurons
** Plastic synapses (Varela)
* Two neuron populations (e, i)
* One layer
* No hierarchy
* 1D tonotopy
* Simplistic thalamic input
"""

#%% Startup
from brian2 import *

#%% Definitions

# ================ Neurons ===================================
LIF = Equations('''
dV/dt = (-gL*(V-EL) - Isyn + n_sig*sqrt(2/n_tau)*xi*ms + I) / C : volt (unless refractory)
I : amp
''')

simple_synapse = Equations('''
I = g*(V-E_syn) : amp
dg/dt = -g/tau_syn : siemens
''')

LIF_eqn = (LIF +
           simple_synapse.substitute(g='g_ampa', I = 'I_ampa', E_syn = 'E_ampa', tau_syn = 'tau_ampa') +
           simple_synapse.substitute(g='g_gaba', I = 'I_gaba', E_syn = 'E_gaba', tau_syn = 'tau_gaba') +
           'Isyn = I_ampa + I_gaba: amp')

LIF_defaults = {
    'gL': 5 * nS,       # Leak conductance
    'EL': -60 * mV,     # Leak potential
    'C': 100 * pF,      # Capacitance
    
    'n_sig': 50 * pA,   # Noise current, sigma
    'n_tau': 5 * ms,    # Noise current, time constant
    
    'E_ampa': 0 * mV,   # AMPA reversal potential
    'tau_ampa': 5 * ms, # AMPA time constant
    
    'E_gaba': -70 * mV, # GABA reversal potential
    'tau_gaba': 10 * ms,# GABA time constant
    
    'threshold': -50 * mV,  # Spike threshold
    'refractory': 2 * ms    # Refractory period
}

# ================ Synapses ===================================
varela_FDD_weight = 'df*ds*f'
varela_FDD = Equations('''
ddf/dt = (1-df)/tauDf : 1 (event-driven)
dds/dt = (1-ds)/tauDs : 1 (event-driven)
df/dt = (1-f)/tauF : 1 (event-driven)
''')
varela_FDD_onpre = '''
df *= Df
ds *= Ds
f += F
'''

varela_FDD_defaults = { # From Varela et al., 1997, Figure 3 (EPSCs)
    'Df': 0.416,            # Fast depression factor [0..1]
    'tauDf': 380 * ms,      # Fast depression recovery time constant
    'Ds': 0.975,            # Slow depression factor [0..1]
    'tauDs': 9.2 * second,  # Slow depression recovery time constant
    'F': 0.917,             # Facilitation constant >= 0
    'tauF': 94 * ms         # Facilitation recovery time constant
}


STDP_weight = 'w'
STDP = Equations('''
w : 1
dapre/dt = -apre/taupre : 1 (event-driven)
dapost/dt = -apost/taupost : 1 (event-driven)
''')
STDP_onpre = '''
apre += Apre
w = clip(w + apost, wmin, wmax)
'''
STDP_onpost = '''
apost += Apost
w = clip(w + apre, wmin, wmax)
'''

STDP_defaults = {
    'taupre': 20 * ms,  # pre before post time constant
    'taupost': 20 * ms, # post before pre time constant
    'Apre': 0.01,       # pre before post weight change
    'Apost' : -0.01,    # post before pre weight change 
    'wmin': 0,          # Min weight factor
    'wmax': 5           # Max weight factor
}

#%% Parameters

params = dict()
params['octaves'] = 1.

# ================ Thalamus ===================================
params_T = params.copy()
params_T['max_rate'] = 50 * Hz
params_T['width'] = 0.2 # octaves; affects rate

NT = 20*params['octaves']

# ================= E =========================================
params_E = {**params, **LIF_defaults}
params_E['gL'] = 6 * nS
params_E['C'] = 200 * pF
print("E membrane time constant:", params_E['C']/params_E['gL'])

NE = 100*params['octaves']

# ================= I =========================================
params_I = {**params, **LIF_defaults}
params_I['gL'] = 4 * nS
params_I['C'] = 100 * pF
print("I membrane time constant:", params_I['C']/params_I['gL'])

params_I['n_sig'] = 20 * pA

NI = 0.2*NE

# ================= Network: Cortex ======================================
params_synapses = params.copy()
params_synapses['delay_per_oct'] = 5 * ms # per octave

params_EE = params_synapses.copy()
params_EE['wmax'] = 3 * nS
params_EE['width_bin'] = 0.5 # octaves; binary connection probability
params_EE['width'] = 0.1 # octaves; affects weight

params_II = params_synapses.copy()
params_II['wmax'] = 5 * nS
params_II['width_bin'] = 0.5 # octaves; binary connection probability
params_II['width'] = 0.1 # octaves; affects weight

params_EI = params_synapses.copy()
params_EI['wmax'] = 5 * nS
params_EI['width_bin'] = 0.5 # octaves; binary connection probability
params_EI['width'] = 0.2 # octaves; affects weight

params_IE = params_synapses.copy()
params_IE['wmax'] = 5 * nS
params_IE['width_bin'] = 0.5 # octaves; binary connection probability
params_IE['width'] = 0.2 # octaves; affects weight

# ================== Network: Thalamo-cortical ===========================
params_TE = params_synapses.copy()
params_TE['weight'] = 1.1 * nS
params_TE['width_p'] = 0.1 # octaves; affects connection probability

params_TI = params_synapses.copy()
params_TI['weight'] = 0.5 * nS
params_TI['width_p'] = 0.15 # octaves; affects connection probability

#%% Experimental setup

exp_period = 2 * second
T_rate = Equations('''
rates = max_rate * exp(-alpha * (current_freq - best_freq)**2) : Hz
current_freq = -(cos(2*pi*t/exp_period) - 1)/2 : 1
best_freq = i * 1.0/N : 1
alpha = 1/(2*width**2) : 1
''')

#%% Building the populations

def build_populations():
    T = NeuronGroup(NT, T_rate + 'x : 1',
                    threshold='rand()<rates*dt',
                    namespace = params_T,
                    name = 'Thalamus')
    T.x = 'i * octaves / N'
    
    eqn = LIF_eqn + 'x : 1'
    E = NeuronGroup(NE, eqn, method='euler',
                    threshold = 'V > threshold',
                    reset = 'V = EL',
                    refractory = params_E['refractory'],
                    namespace = params_E,
                    name = 'Excitatory')
    E.V = params_E['EL']
    E.x = 'i * octaves / N'


    I = NeuronGroup(NI, eqn, method='euler',
                    threshold = 'V > threshold',
                    reset = 'V = EL',
                    refractory = params_I['refractory'],
                    namespace = params_I,
                    name = 'Inhibitory')
    I.V = params_I['EL']
    I.x = 'i * octaves / N'
    
    return T, E, I

#%% Building the network

def build_network(T, E, I):
    delay_eqn = 'delay_per_oct * abs(x_pre-x_post)'
    
    TE = Synapses(T, E,
                  on_pre = 'g_ampa_post += weight',
                  namespace = params_TE,
                  name='Thal_Exc')
    TE.connect(p = 'exp(-(i*octaves/N_pre - j*octaves/N_post)**2 / (2*width_p**2))')
    
    TI = Synapses(T, I,
                  on_pre = 'g_ampa_post += weight',
                  namespace = params_TI,
                  name='Thal_Inh')
    TI.connect(p = 'exp(-(i*octaves/N_pre - j*octaves/N_post)**2 / (2*width_p**2))')
    
    EE = Synapses(E, E,
                  model = 'weight : siemens',
                  on_pre = 'g_ampa_post += weight',
                  namespace = params_EE,
                  name = 'Exc_Exc')
    EE.connect(condition = 'i!=j and abs(x_pre-x_post) < width_bin')
    EE.weight = 'wmax * exp(-(x_pre-x_post)**2/(2*width**2))'
    EE.delay = delay_eqn
    
    EI = Synapses(E, I,
                  model = 'weight : siemens',
                  on_pre = 'g_ampa_post += weight',
                  namespace = params_EI,
                  name = 'Exc_Inh')
    EI.connect(condition = 'abs(x_pre-x_post) < width_bin')
    EI.weight = 'wmax * exp(-(x_pre-x_post)**2/(2*width**2))'
    
    IE = Synapses(I, E,
                  model = 'weight : siemens',
                  on_pre = 'g_gaba_post += weight',
                  namespace = params_IE,
                  name = 'Inh_Exc')
    IE.connect(condition = 'abs(x_pre-x_post) < width_bin')
    IE.weight = 'wmax * exp(-(x_pre-x_post)**2/(2*width**2))'
    
    II = Synapses(I, I,
                  model = 'weight : siemens',
                  on_pre = 'g_gaba_post += weight',
                  namespace = params_II,
                  name = 'Inh_Inh')
    II.connect(condition = 'i!=j and abs(x_pre-x_post) < width_bin')
    II.weight = 'wmax * exp(-(x_pre-x_post)**2/(2*width**2))'
    
    return TE, TI, \
           EE, EI, \
           IE, II
