import torch
from cantata import util, init
import cantata.elements as ce


class Membrane(ce.Module):
    '''
    Leaky integrator membrane with a refractory mechanism
    Input: Instantaneous spikes; synaptic current
    Output: Membrane voltage
    Internal state: Voltage, refractory state
    '''

    def __init__(self, N, batch_size, alpha, tau_ref=None, noise=None):
        super().__init__()
        self.register_parabuf('alpha', alpha)
        self.register_buffer('V', torch.zeros(batch_size, N))
        if tau_ref is None:
            tau_ref = torch.zeros_like(self.V, dtype=torch.int16)
        self.register_buffer('tau_ref', tau_ref, persistent=False)
        self.register_buffer(
            'ref', torch.zeros(batch_size, N, dtype=tau_ref.dtype))
        self.noisy = noise is not None and noise.active
        if self.noisy:
            self.noise = noise
        self.reset()

    @classmethod
    def configured(cls, conf, batch_size, dt, train_tau_mem=False,
                   disable_training=False, **kwargs):
        N = init.get_N(conf)
        ref_dtype = torch.int16

        # Parameters
        tm = init.expand_to_neurons(conf, 'tau_mem')
        tmg = init.expand_to_neurons(conf, 'tau_mem_gamma') * 1.0
        G = torch.distributions.gamma.Gamma(tmg, tmg/tm)
        alpha = util.decayconst(G.sample(), dt)
        if train_tau_mem and not disable_training:
            alpha = torch.nn.Parameter(alpha)

        tau_ref = init.expand_to_neurons(conf, 'tau_ref')
        tau_ref = torch.round(tau_ref/dt).to(ref_dtype)
        tau_ref = torch.clip(tau_ref, min=1).expand(batch_size, N)

        # Models
        noise = ce.Noise(conf, batch_size, dt,
                         disable_training=disable_training, **kwargs)

        return cls(N, batch_size, alpha, tau_ref=tau_ref, noise=noise)

    def reset(self, keep_values=False):
        if keep_values:
            self.V = self.V.detach()
            self.ref = self.ref.detach()
        else:
            self.V = torch.rand_like(self.V)
            self.ref = torch.zeros_like(self.ref)
        if isinstance(self.alpha, torch.nn.Parameter):
            self.alpha.data.clamp_(0., 1.)

    def forward(self, current, X=None):
        self.V = self.alpha*self.V + (1-self.alpha)*current
        if self.noisy:
            self.V = self.V + self.noise()

        if X is not None:
            with torch.no_grad():
                spiking = X > 0
                self.ref[spiking] = self.tau_ref[spiking]
                refractory = self.ref > 0
                self.V[spiking + refractory] = 0
                self.ref[refractory] -= 1

        return self.V
