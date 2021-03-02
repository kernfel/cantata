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
    def __init__(self, conf_pre, STDP, batch_size, dt,
                 conf_post = None, name_post = None):
    # def __init__(self, projections, delaymap, conf_pre, conf_post, STDP,
    #              batch_size, nPre, nPost, dt):
        super(DeltaSynapse, self).__init__()
        nPre = sum([p.n for p in conf_pre.populations.values()])
        xarea = conf_post is not None
        if xarea:
            nPost = sum([p.n for p in conf_post.populations.values()])
            projections = init.build_projections_xarea(
                conf_pre, conf_post, name_post)
        else:
            nPost = nPre
            conf_post = conf_pre
            projections = init.build_projections(conf_pre)
        self.active = len(projections[0]) > 0
        if not self.active:
            return

        delaymap = init.get_delaymap(
            conf_pre, projections, nPre, nPost, dt, xarea)
        self.register_buffer('delaymap', delaymap, persistent=False)
        wmax = init.expand_to_synapses(projections, nPre, nPost, 'wmax')
        self.register_buffer('wmax', wmax, persistent=False)

        # Weights
        w = init.build_connectivity(projections, nPre, nPost) * self.wmax
        self.W = torch.nn.Parameter(w)
        signs = init.expand_to_neurons(conf_pre, 'sign').to(torch.int8)
        self.register_buffer('signs_pre', signs, persistent = False)
        self.register_buffer('signs', torch.zeros_like(w), persistent = False)

        # Short-term plasticity
        shortterm = ce.STP(conf_pre, delaymap.shape[0], batch_size, nPre, dt)
        self.has_STP = shortterm.active
        if self.has_STP:
            self.shortterm = shortterm

        # Long-term plasticity
        STDP_frac = init.expand_to_synapses(
            projections, nPre, nPost, 'STDP_frac')
        longterm = STDP(
            projections, self, conf_post, batch_size, nPre, nPost, dt)
        self.has_STDP = torch.any(STDP_frac > 0) and longterm.active
        if self.has_STDP:
            self.register_buffer('STDP_frac', STDP_frac, persistent = False)
            self.longterm = longterm

        self.reset()

    def reset(self):
        if self.active:
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
        if not self.active:
            return None
        if self.has_STDP:
            Wlong = self.longterm(Xd, X, Vpost)
            W = self.signs * \
                (self.W * (1-self.STDP_frac) + Wlong * self.STDP_frac)
            WD = 'beo'
        else:
            W = self.signs * self.W
            WD = 'eo'

        if self.has_STP:
            Wshort = self.shortterm(Xd)+1
            I = torch.einsum(
                f'{WD}, dbe, deo,           dbe   ->bo',
                 W,     Xd,  self.delaymap, Wshort)
        else:
            I = torch.einsum(
                f'{WD}, dbe, deo,         ->bo',
                 W,     Xd,  self.delaymap)
        return I

    def load_state_dict(self, *args, **kwargs):
        super(DeltaSynapse, self).load_state_dict(*args, **kwargs)
        self.align_signs()

    def align_signs(self):
        if not self.active:
            return
        signs = self.signs_pre.unsqueeze(1).expand_as(self.signs)
        signs = torch.where(self.W>0, signs, torch.zeros_like(signs))
        self.signs = signs
