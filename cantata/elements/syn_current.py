import torch
from cantata import util, init
import cantata.elements as ce

class CurrentSynapse(torch.nn.Module):
    '''
    Current synapse, wrapper around a DeltaSynapse.
    Input: Delayed presynaptic spikes; postsynaptic spikes; postsynaptic voltage
    Output: Synaptic currents
    Internal state: Current I
    '''
    def __init__(self, conf_pre, batch_size, dt,
                 conf_post = None, name_post = None,
                 STDP = ce.Abbott, shared_weights = True):
        super(type(self), self).__init__()
        self.syn = ce.DeltaSynapse(
            conf_pre, batch_size, dt,
            conf_post, name_post,
            STDP, shared_weights)
        self.active = self.syn.active
        if not self.active:
            return

        conf = conf_pre if conf_post is None else conf_post
        N = init.get_N(conf)

        # Current
        tau_I = conf.tau_I
        self.alpha_I = util.decayconst(conf.tau_I, dt)
        self.register_buffer('I', torch.zeros(batch_size, N))

    def reset(self):
        if self.active:
            self.I = torch.zeros_like(self.I)
            self.syn.reset()

    def forward(self, Xd, X, Vpost):
        '''
        Xd: (delay, batch, pre)
        X: (batch, post)
        Vpost: (batch, post)
        Output: Current (batch, post)
        '''
        if not self.active:
            return torch.zeros_like(Vpost)
        increment = self.syn.forward(Xd, X, Vpost)
        self.I = self.I * self.alpha_I + increment*(1-self.alpha_I)
        return self.I
