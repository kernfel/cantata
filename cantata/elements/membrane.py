import torch
from cantata import util, init, cfg

class Membrane(torch.nn.Module):
    '''
    Leaky integrator membrane with a refractory mechanism
    Input: Instantaneous spikes; synaptic current
    Output: Membrane voltage
    Internal state: Voltage, refractory state
    '''
    def __init__(self):
        super(Membrane, self).__init__()
        N = init.get_N()
        shape = (cfg.batch_size, N)

        # Parameters
        if cfg.model.tau_mem_gamma > 0:
            a, b = torch.tensor([
                cfg.model.tau_mem_gamma,
                cfg.model.tau_mem_gamma/cfg.model.tau_mem
            ])
            G = torch.distributions.gamma.Gamma(a, b)
            self.register_buffer('alpha', util.decayconst(G.sample((N,))))
        else:
            self.alpha = util.decayconst(cfg.model.tau_mem)

        tau_ref = np.round(cfg.model.tau_ref / cfg.time_step)
        self.tau_ref = int(max(1, tau_ref))

        self.register_buffer('noise_N',
            init.expand_to_neurons('noise_N').expand(shape),
            persistent = False)
        self.register_buffer('noise_p',
            (init.expand_to_neurons('noise_rate') * cfg.time_step).clamp(0,1),
            persistent = False)
        self.register_buffer('noise_weight',
            init.expand_to_neurons('noise_weight'),
            persistent = False)
        self.noisy = torch.any(
            (self.noise_weight > 0) * (noise_N[0,:] > 0) * (noise_p > 0))

        # States
        self.register_buffer('V', torch.zeros(shape))
        self.register_buffer('ref', torch.zeros(shape), dtype=torch.int8)

        # Init
        torch.nn.init.uniform_(self.V)

    def noise(self):
        d = torch.distributions.Binomial(self.noise_N, self.noise_p)
        return d.sample() * self.noise_weight

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
