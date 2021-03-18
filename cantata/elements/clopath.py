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
    def __init__(self, projections, host, conf, batch_size, nPre, nPost, dt):
        super(Clopath, self).__init__()

        # Parameters
        self.alpha_p = util.decayconst(conf.tau_p, dt)
        self.alpha_d = util.decayconst(conf.tau_d, dt)
        self.alpha_x = util.decayconst(conf.tau_x, dt)
        A_p = init.expand_to_synapses(projections, nPre, nPost, 'A_p')
        A_d = init.expand_to_synapses(projections, nPre, nPost, 'A_d')
        self.active = torch.any(A_p != 0) or torch.any(A_d != 0)
        if self.active:
            self.register_buffer('A_p', A_p, persistent = False)
            self.register_buffer('A_d', A_d, persistent = False)

        # State
        self.host = weakref.ref(host)
        d,b = self.host().delaymap.shape[0], batch_size
        if self.active:
            self.register_buffer('xbar_pre', torch.zeros(d,b,nPre))
            self.register_buffer('u_pot', torch.zeros(b,nPost))
            self.register_buffer('u_dep', torch.zeros(b,nPost))
        self.register_buffer('W', torch.zeros(b,nPre,nPost))

    def reset(self, W):
        if self.active:
            self.xbar_pre = torch.zeros_like(self.xbar_pre)
            self.u_pot = torch.zeros_like(self.u_pot)
            self.u_dep = torch.zeros_like(self.u_dep)
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
            host = self.host()
            dmap = host.delaymap
            dW_pot = torch.einsum(
                'dbe,          deo,  eo,       bo                    ->beo',
                self.xbar_pre, dmap, self.A_p, Xpost*relu(self.u_pot))
            dW_dep = torch.einsum(
                'dbe, deo,  eo,       bo              ->beo',
                Xd,   dmap, self.A_d, relu(self.u_dep))
            self.xbar_pre = util.expfilt(Xd, self.xbar_pre, self.alpha_x)
            self.u_pot = util.expfilt(Vpost, self.u_pot, self.alpha_p)
            self.u_dep = util.expfilt(Vpost, self.u_dep, self.alpha_d)
            self.W = torch.minimum(host.wmax, torch.maximum(host.wmin,
                self.W + dW_pot - dW_dep))
        return out
