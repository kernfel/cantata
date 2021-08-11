import torch
from cantata import util, init, elements as ce


class SynCurrent(ce.Module):
    '''
    Synaptic current filter to turn a delta synapse into a current synapse.
    Input: (Post-) Synaptic inputs in the form of current impulses
    Output: Decaying synaptic currents
    '''

    def __init__(self, N, batch_size, alpha):
        super().__init__()
        self.active = N > 0
        if not self.active:
            return
        self.alpha = alpha
        self.register_buffer('I', torch.zeros(batch_size, N))

    @classmethod
    def configured(cls, conf, batch_size, dt, train_tau_syn=False,
                   disable_training=False, **kwargs):
        active = conf.tau_I > 0
        if not active:
            return cls(0, None, None)
        N = init.get_N(conf)
        alpha = util.decayconst(conf.tau_I, dt)
        if train_tau_syn and not disable_training:
            alpha = torch.nn.Parameter(torch.tensor(alpha))
        return cls(N, batch_size, alpha)

    def reset(self, keep_values=False):
        if not self.active:
            return
        if keep_values:
            self.I = self.I.detach()
        else:
            self.I = torch.zeros_like(self.I)
        if isinstance(self.alpha, torch.nn.Parameter):
            self.alpha.data.clamp_(0., 1.)

    def forward(self, impulses):
        '''
        impulses: (batch, post)
        Output: Current (batch, post)
        '''
        if not self.active:
            return impulses
        self.I = self.alpha*self.I + (1-self.alpha)*impulses
        return self.I
