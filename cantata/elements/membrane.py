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

        # Parameters
        if conf.tau_mem_gamma > 0:
            a, b = torch.tensor([
                conf.tau_mem_gamma,
                conf.tau_mem_gamma/conf.tau_mem
            ])
            G = torch.distributions.gamma.Gamma(a, b)
            self.register_buffer(
                'alpha', util.decayconst(G.sample((N,)), dt))
        else:
            self.alpha = util.decayconst(conf.tau_mem, dt)

        tau_ref = np.round(conf.tau_ref / dt)
        self.tau_ref = int(max(1, tau_ref))

        # Models
        noise = ce.Noise(conf, batch_size, dt)
        self.noisy = noise.active
        if self.noisy:
            self.noise = noise

        # States
        self.register_buffer('V', torch.zeros(batch_size, N))
        self.register_buffer(
            'ref', torch.zeros(batch_size, N, dtype=torch.int16))
        self.reset()

    def reset(self):
        self.V = torch.rand_like(self.V)
        self.ref = torch.zeros_like(self.ref)

    def forward(self, X, current):
        self.V = self.V*self.alpha + current
        if self.noisy:
            self.V += self.noise()

        with torch.no_grad():
            self.ref[X > 0] = self.tau_ref
            refractory = self.ref > 0
            self.V[refractory] = 0
            self.ref[refractory] -= 1

        return self.V
