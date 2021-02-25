import torch
from cantata import util, init, cfg
import cantata.elements as ce

class DeltaSynapse(torch.nn.Module):
    '''
    Delta synapse with short- and long-term plasticity submodules
    Input: Delayed presynaptic spikes; postsynaptic spikes
    Output: Synaptic currents
    Internal state: -
    '''
    def __init__(self, projections, delaymap):
        super(DeltaSynapse, self).__init__()
        N = init.get_N()

        # Parameters
        self.register_buffer('STDP_frac',
            init.expand_to_synapses('STDP_frac', projections),
            persistent = False)
        self.register_buffer('delaymap', delaymap, persistent=False)

        self.wmax = cfg.model.wmax
        w = init.build_connectivity(projections) * self.wmax
        self.W = torch.nn.Parameter(w)
        self.register_buffer('signs', torch.zeros_like(w), persistent = False)
        self.align_signs()

        # Models
        self.shortterm = ce.STP()
        if cfg.model.STDP_Clopath:
            self.longterm = ce.Clopath(projections)
        else:
            self.longterm = ce.Abbott(projections)

    def forward(self, Xd, X):
        '''
        Xd: (delay, batch, pre)
        X: (batch, post)
        Output: Current (batch, post)
        '''
        stp = self.shortterm(Xd)
        Xpre = torch.einsum('deo,dbe->beo', self.delaymap, Xd)
        Wlong = self.longterm(Xpre, X)
        W = self.signs * (self.W * (1-self.STDP_frac) + Wlong * self.STDP_frac)
        I = torch.einsum('beo,beo->bo', W, Xpre)
        return I

    def load_state_dict(self, *args, **kwargs):
        super(DeltaSynapse, self).load_state_dict(*args, **kwargs)
        self.align_signs()

    def align_signs(self):
        signs = init.expand_to_neurons('sign').unsqueeze(1).expand(N,N)
        signs = torch.where(self.W>0, signs, torch.zeros_like(signs))
        self.signs = signs.to(torch.int8)
