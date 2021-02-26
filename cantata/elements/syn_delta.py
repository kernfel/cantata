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

        self.register_buffer('delaymap', delaymap, persistent=False)

        # Weights
        self.wmax = cfg.model.wmax
        w = init.build_connectivity(projections) * self.wmax
        self.W = torch.nn.Parameter(w)
        self.register_buffer('signs', torch.zeros_like(w), persistent = False)

        # Short-term plasticity
        shortterm = ce.STP()
        self.has_STP = shortterm.active
        if self.has_STP:
            self.shortterm = shortterm

        # Long-term plasticity
        STDP_frac = init.expand_to_synapses('STDP_frac', projections)
        STDP_model = ce.Clopath if cfg.model.STDP_Clopath else ce.Abbott
        longterm = STDP_model(projections, self)
        self.has_STDP = torch.any(STDP_frac > 0) and longterm.active
        if self.has_STDP:
            self.register_buffer('STDP_frac', STDP_frac, persistent = False)
            self.longterm = longterm

        self.reset()

    def reset(self):
        self.align_signs()
        if self.has_STP:
            self.shortterm.reset()
        if self.has_STDP:
            self.longterm.reset(self.W)

    def forward(self, Xd, X, Vpost):
        '''
        Xd: (delay, batch, pre)
        X: (batch, post)
        Vpost: (batch, post)
        Output: Current (batch, post)
        '''
        if self.has_STDP:
            Wlong = self.longterm(Xd, X, Vpost)
            W = self.signs * \
                (self.W * (1-self.STDP_frac) + Wlong * self.STDP_frac)
        else:
            W = self.signs * self.W

        if self.has_STP:
            Wshort = torch.einsum('deo,dbe->beo', self.shortterm(Xd)+1)
            I = torch.einsum(
                'beo, dbe, deo,           beo   ->bo',
                 W,   Xd,  self.delaymap, Wshort)
        else:
            I = torch.einsum(
                'beo, dbe, deo,         ->bo',
                 W,   Xd,  self.delaymap)
        return I

    def load_state_dict(self, *args, **kwargs):
        super(DeltaSynapse, self).load_state_dict(*args, **kwargs)
        self.align_signs()

    def align_signs(self):
        signs = init.expand_to_neurons('sign').unsqueeze(1).expand(N,N)
        signs = torch.where(self.W>0, signs, torch.zeros_like(signs))
        self.signs = signs.to(torch.int8)
