import torch
from cantata import util, init, cfg

class Abbott(torch.nn.Module):
    '''
    Abbott STDP model, asymmetric
    Input: Presynaptic spikes arriving at the terminals; postsynaptic spikes
    Output: Appropriately weighted inputs to the postsynaptic neurons
    Internal state: Weights, pre- and postsynaptic activity traces
    '''
    def __init__(self, projections):
        super(Abbott, self).__init__()
        N = init.get_N()
        shape = (cfg.batch_size, N)

        # Parameters
        self.alpha_p = util.decayconst(cfg.model.tau_p)
        self.alpha_d = util.decayconst(cfg.model.tau_d)
        self.wmax = cfg.model.wmax
        A_p = init.expand_to_synapses('A_p', projections)
        A_d = init.expand_to_synapses('A_d', projections)
        self.active = torch.any(A_p != 0) or torch.any(A_d != 0)
        if self.active:
            self.register_buffer('A_p', A_p, persistent = False)
            self.register_buffer('A_d', A_d, persistent = False)

        # State
        if self.active:
            self.register_buffer('xbar_pre', torch.zeros(shape))
            self.register_buffer('xbar_post', torch.zeros(shape))
        self.register_buffer('W', torch.zeros(shape + (N,)))

    def forward(self, Xpre, Xpost):
        '''
        Xpre: (batch, pre, post)
        Xpost: (batch, post)
        Output: Synaptic weight before the update (batch, pre, post)
        '''
        W = self.W.clone()
        if self.active:
            dW_pot = torch.einsum('bo,   beo,           eo      ->beo',
                                  Xpost, self.xbar_pre, self.A_p)
            dW_dep = torch.einsum('beo, bo,             eo      ->beo',
                                  Xpre, self.xbar_post, self.A_d)
            self.xbar_pre = util.expfilt(Xpre, self.xbar_pre, self.alpha_p)
            self.xbar_post = util.expfilt(Xpost, self.xbar_post, self.alpha_d)
            self.W = torch.clamp(W + dW_pot - dW_dep, 0, self.wmax)
        return W
