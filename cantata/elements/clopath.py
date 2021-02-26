import torch
from cantata import util, init
import weakref
from torch.nn.functional import relu

class Clopath(torch.nn.Module):
    '''
    Clopath STDP model
    Input: Delayed presynaptic spikes; postsynaptic spikes
    Output: Appropriately weighted inputs to the postsynaptic neurons
    Internal state: Weights, pre- and postsynaptic activity traces
    '''
    def __init__(self, projections, dmap_host, conf, batch_size, N, dt):
        super(Clopath, self).__init__()

        # Parameters
        self.alpha_p = util.decayconst(conf.tau_p, dt)
        self.alpha_d = util.decayconst(conf.tau_d, dt)
        self.alpha_x = util.decayconst(conf.tau_x, dt)
        self.wmax = conf.wmax
        A_p = init.expand_to_synapses(projections, N, N, 'A_p')
        A_d = init.expand_to_synapses(projections, N, N, 'A_d')
        self.active = torch.any(A_p != 0) or torch.any(A_d != 0)
        if self.active:
            self.register_buffer('A_p', A_p, persistent = False)
            self.register_buffer('A_d', A_d, persistent = False)

        # State
        self.dmap_host = weakref.ref(dmap_host)
        d,b = self.dmap_host().delaymap.shape[0], batch_size
        if self.active:
            self.register_buffer('xbar_pre', torch.zeros(d,b,N))
            self.register_buffer('u_pot', torch.zeros(b,N))
            self.register_buffer('u_dep', torch.zeros(b,N))
        self.register_buffer('W', torch.zeros(b,N,N))

    def reset(self, W):
        if self.active:
            torch.nn.init.zeros_(self.xbar_pre)
            torch.nn.init.zeros_(self.u_pot)
            torch.nn.init.zeros_(self.u_dep)
        self.W = W.clone().expand_as(self.W)

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
