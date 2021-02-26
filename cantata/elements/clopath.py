import torch
from cantata import util, init, cfg
import weakref
from torch.nn.functional import relu

class Clopath(torch.nn.Module):
    '''
    Clopath STDP model
    Input: Delayed presynaptic spikes; postsynaptic spikes
    Output: Appropriately weighted inputs to the postsynaptic neurons
    Internal state: Weights, pre- and postsynaptic activity traces
    '''
    def __init__(self, projections, dmap_host):
        super(Clopath, self).__init__()

        # Parameters
        self.alpha_p = util.decayconst(cfg.model.tau_p)
        self.alpha_d = util.decayconst(cfg.model.tau_d)
        self.alpha_x = util.decayconst(cfg.model.tau_x)
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
            self.register_buffer('u_pot', torch.zeros(b,N))
            self.register_buffer('u_dep', torch.zeros(b,N))
        self.register_buffer('W', torch.zeros(b,N,N))

    def forward(self, Xd, Xpost, Vpost):
        '''
        Xd: (delay, batch, pre)
        Xpost: (batch, post)
        Vpost: (batch, post)
        Output: Synaptic weight before the update (batch, pre, post)
        '''
        out = self.W.clone()
        if self.active:
            dmap = self.dmap_host().delaymap
            dW_pot = torch.einsum(
                'dbe,          deo,  eo,       bo                    ->beo',
                self.xbar_pre, dmap, self.A_p, Xpost*relu(self.u_pot))
            dW_dep = torch.einsum(
                'dbe, deo,  eo,       bo              ->beo',
                Xd,   dmap, self.A_d, relu(self.u_dep))
            self.xbar_pre = util.expfilt(Xd, self.xbar_pre, self.alpha_x)
            self.u_pot = util.expfilt(Vpost, self.u_pot, self.alpha_p)
            self.u_dep = util.expfilt(Vpost, self.u_dep, self.alpha_d)
            self.W = torch.clamp(self.W + dW_pot - dW_dep, 0, self.wmax)
        return out
