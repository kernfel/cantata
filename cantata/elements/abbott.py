import torch
from cantata import util, init, cfg
import weakref

class Abbott(torch.nn.Module):
    '''
    Abbott STDP model, asymmetric
    Input: Delayed presynaptic spikes; postsynaptic spikes
    Output: Appropriately weighted inputs to the postsynaptic neurons
    Internal state: Weights, pre- and postsynaptic activity traces
    '''
    def __init__(self, projections, dmap_host):
        super(Abbott, self).__init__()

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
        self.dmap_host = weakref.ref(dmap_host)
        N = init.get_N()
        d,b = self.dmap_host().delaymap.shape[0], cfg.batch_size
        if self.active:
            self.register_buffer('xbar_pre', torch.zeros(d,b,N))
            self.register_buffer('xbar_post', torch.zeros(b,N))
        self.register_buffer('W', torch.zeros(b,N,N))

    def forward(self, Xd, Xpost, *args):
        '''
        Xd: (delay, batch, pre)
        Xpost: (batch, post)
        Output: Synaptic weight before the update (batch, pre, post)
        '''
        W = self.W.clone()
        if self.active:
            dmap = self.dmap_host().delaymap
            dW_pot = torch.einsum(
                'bo,   dbe,           deo,  eo,     ->beo',
                Xpost, self.xbar_pre, dmap, self.A_p)
            dW_dep = torch.einsum(
                'dbe, bo,             deo,  eo      ->beo',
                Xd,   self.xbar_post, dmap, self.A_d)
            self.xbar_pre = util.expfilt(Xd, self.xbar_pre, self.alpha_p)
            self.xbar_post = util.expfilt(Xpost, self.xbar_post, self.alpha_d)
            self.W = torch.clamp(W + dW_pot - dW_dep, 0, self.wmax)
        return W
