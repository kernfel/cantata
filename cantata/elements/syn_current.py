import torch
from cantata import util, init

class SynCurrent(torch.nn.Module):
    '''
    Synaptic current filter to turn a delta synapse into a current synapse.
    Input: (Post-) Synaptic inputs in the form of current impulses
    Output: Decaying synaptic currents
    '''
    def __init__(self, conf, batch_size, dt, **kwargs):
        super(SynCurrent, self).__init__()
        self.active = conf.tau_I > 0
        if not self.active:
            return
        N = init.get_N(conf)
        self.alpha = util.decayconst(conf.tau_I, dt)
        self.register_buffer('I', torch.zeros(batch_size, N))

    def reset(self):
        if not self.active:
            return
        self.I.zero_()

    def forward(self, impulses):
        '''
        impulses: (batch, post)
        Output: Current (batch, post)
        '''
        if not self.active:
            return impulses
        self.I = self.alpha*self.I + (1-self.alpha)*impulses
        return self.I
