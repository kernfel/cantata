import torch
from cantata import util, init, elements as ce


class SynCurrent(ce.Module):
    '''
    Synaptic current filter to turn a delta synapse into a current synapse.
    Input: (Post-) Synaptic inputs in the form of current impulses
    Output: Decaying synaptic currents
    '''

    def __init__(self, conf, batch_size, dt, train_tau=False,
                 disable_training=False, **kwargs):
        super(SynCurrent, self).__init__()
        self.active = conf.tau_I > 0
        if not self.active:
            return
        N = init.get_N(conf)
        alpha = util.decayconst(conf.tau_I, dt)
        if train_tau and not disable_training:
            self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        else:
            self.alpha = alpha  # single value
        self.register_buffer('I', torch.zeros(batch_size, N))

    def reset(self):
        if not self.active:
            return
        self.I = torch.zeros_like(self.I)

    def forward(self, impulses):
        '''
        impulses: (batch, post)
        Output: Current (batch, post)
        '''
        if not self.active:
            return impulses
        self.I = self.alpha*self.I + (1-self.alpha)*impulses
        return self.I
