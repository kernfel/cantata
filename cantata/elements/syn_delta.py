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
        shortterm = ce.STP()
        self.has_STP = shortterm.active
        if self.has_STP:
            self.shortterm = shortterm

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
        Xpre = torch.einsum('deo,dbe->beo', self.delaymap, Xd)
        Wlong = self.longterm(Xpre, X)
        W = self.signs * (self.W * (1-self.STDP_frac) + Wlong * self.STDP_frac)
        if self.has_STP:
            Wshort = torch.einsum('deo,dbe->beo', self.shortterm(Xd)+1)
            I = torch.einsum('beo,beo,beo->bo', W, Xpre, Wshort)
        else:
            I = torch.einsum('beo,beo->bo', W, Xpre)
        return I

    def load_state_dict(self, *args, **kwargs):
        super(DeltaSynapse, self).load_state_dict(*args, **kwargs)
        self.align_signs()

    def align_signs(self):
        signs = init.expand_to_neurons('sign').unsqueeze(1).expand(N,N)
        signs = torch.where(self.W>0, signs, torch.zeros_like(signs))
        self.signs = signs.to(torch.int8)
