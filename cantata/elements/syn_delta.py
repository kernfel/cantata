import torch
from cantata import util, init
import cantata.elements as ce

class DeltaSynapse(torch.nn.Module):
    '''
    Delta synapse with short- and long-term plasticity submodules
    Input: Delayed presynaptic spikes; postsynaptic spikes
    Output: Synaptic currents
    Internal state: -
    '''
    def __init__(self, projections, delaymap, conf, STDP, batch_size, N, dt):
        super(DeltaSynapse, self).__init__()

        self.register_buffer('delaymap', delaymap, persistent=False)

        # Weights
        self.wmax = conf.wmax
        w = init.build_connectivity(conf, projections, N, N) * self.wmax
        self.W = torch.nn.Parameter(w)
        signs = init.expand_to_neurons(conf, 'sign').to(torch.int8)
        self.register_buffer('signs_pre', signs, persistent = False)
        self.register_buffer('signs', torch.zeros_like(w), persistent = False)

        # Short-term plasticity
        shortterm = ce.STP(conf, delaymap.shape[0], batch_size, N, dt)
        self.has_STP = shortterm.active
        if self.has_STP:
            self.shortterm = shortterm

        # Long-term plasticity
        STDP_frac = init.expand_to_synapses(projections, N, N, 'STDP_frac')
        longterm = STDP(projections, self, conf, batch_size, N, dt)
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
            Wshort = self.shortterm(Xd)+1
            I = torch.einsum(
                'beo, dbe, deo,           dbe   ->bo',
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
        signs = self.signs_pre.unsqueeze(1).expand_as(self.signs)
        signs = torch.where(self.W>0, signs, torch.zeros_like(signs))
        self.signs = signs
