import torch
import numpy as np
from cantata import util, init
import cantata.elements as ce

class Membrane(torch.nn.Module):
    '''
    Leaky integrator membrane with a refractory mechanism
    Input: Instantaneous spikes; synaptic current
    Output: Membrane voltage
    Internal state: Voltage, refractory state
    '''
    def __init__(self, conf, batch_size, dt):
        super(Membrane, self).__init__()
        N = init.get_N(conf)
        ref_dtype = torch.int16

        # Parameters
        tm = init.expand_to_neurons(conf, 'tau_mem')
        tmg = init.expand_to_neurons(conf, 'tau_mem_gamma') * 1.0
        G = torch.distributions.gamma.Gamma(tmg, tmg/tm)
        self.register_buffer('alpha', util.decayconst(G.sample(), dt))

        tau_ref = init.expand_to_neurons(conf, 'tau_ref')
        tau_ref = torch.round(tau_ref/dt).to(ref_dtype)
        tau_ref = torch.clip(tau_ref, min = 1).expand(batch_size, N)
        self.register_buffer('tau_ref', tau_ref, persistent = False)

        # Models
        noise = ce.Noise(conf, batch_size, dt)
        self.noisy = noise.active
        if self.noisy:
            self.noise = noise

        # States
        self.register_buffer('V', torch.zeros(batch_size, N))
        self.register_buffer(
            'ref', torch.zeros(batch_size, N, dtype=ref_dtype))
        self.reset()

    def reset(self):
        self.V = torch.rand_like(self.V)
        self.ref = torch.zeros_like(self.ref)

    def forward(self, X, current):
        self.V = self.V*self.alpha + current
        if self.noisy:
            self.V += self.noise()

        with torch.no_grad():
            spiking = X > 0
            self.ref[spiking] = self.tau_ref[spiking]
            refractory = self.ref > 0
            self.V[refractory] = 0
            self.ref[refractory] -= 1

        return self.V
