import torch
from cantata import util, init
import weakref

class Abbott(torch.nn.Module):
    '''
    Abbott STDP model, asymmetric
    Input: Delayed presynaptic spikes; postsynaptic spikes
    Output: Appropriately weighted inputs to the postsynaptic neurons
    Internal state: Weights, pre- and postsynaptic activity traces
    '''
    def __init__(self, projections, conf, batch_size, nPre, nPost, dt):
        super(Abbott, self).__init__()

        # Parameters
        self.alpha_p = util.decayconst(conf.tau_p, dt)
        self.alpha_d = util.decayconst(conf.tau_d, dt)
        A_p = init.expand_to_synapses(projections, nPre, nPost, 'A_p')
        A_d = init.expand_to_synapses(projections, nPre, nPost, 'A_d')
        self.active = torch.any(A_p != 0) or torch.any(A_d != 0)
        if self.active:
            self.register_buffer('A_p', A_p, persistent = False)
            self.register_buffer('A_d', A_d, persistent = False)

        # State
        self.register_buffer('W', torch.zeros(batch_size,nPre,nPost))

    def reset(self, host):
        self.W = host.W.clone().expand_as(self.W)
        if self.active:
            self.host = weakref.ref(host)
            try:
                self.get_buffer('xbar_pre')
            except AttributeError:
                d = host.delaymap.shape[0]
                batch_size, nPre, nPost = self.W.shape
                self.register_buffer('xbar_pre', torch.empty(d,batch_size,nPre))
                self.register_buffer('xbar_post', torch.empty(batch_size,nPost))
            xbar_pre.zero_()
            xbar_post.zero_()

    def forward(self, Xd, Xpost, *args):
        '''
        Xd: (delay, batch, pre)
        Xpost: (batch, post)
        Output: Synaptic weight before the update (batch, pre, post)
        '''
        out = self.W.clone()
        if self.active:
            host = self.host()
            dmap = host.delaymap
            dW_pot = torch.einsum(
                'bo,   deo,  dbe,           eo      ->beo',
                Xpost, dmap, self.xbar_pre, self.A_p)
            dW_dep = torch.einsum(
                'dbe, deo,  bo,             eo      ->beo',
                Xd,   dmap, self.xbar_post, self.A_d)
            self.xbar_pre = util.expfilt(Xd, self.xbar_pre, self.alpha_p)
            self.xbar_post = util.expfilt(Xpost, self.xbar_post, self.alpha_d)
            self.W = torch.minimum(
                host.wmax, torch.clamp(self.W + dW_pot - dW_dep, 0))
        return out

# Note: In order to drop the tau parameters to population or projection level,
# alpha_p would need to filter presynaptic spikes by the postsynaptic target's
# time scale value; therefore, xbar_pre would need to be either (beo), or
# d (in xbar_pre, Xd & dmap) would need to index projections rather than delays.
# The latter solution scales favorably and does not require adjustments for
# other users of Xd/dmap, but it does need to be implemented at the dmap/delays
# source.
