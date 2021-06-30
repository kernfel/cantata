import torch
from cantata import util, init
import cantata.elements as ce

class Synapse(torch.nn.Module):
    '''
    Synapse with optional current, short- and long-term plasticity submodules
    Input: Delayed presynaptic spikes; postsynaptic spikes; postsynaptic voltage
    Output: Synaptic currents
    '''
    def __init__(self, projections, conf_pre, conf_post, batch_size, dt,
                 stp = None, ltp = None, current = None,
                 shared_weights = True, **kwargs):
        super(Synapse, self).__init__()
        self.active = len(projections[0]) > 0
        if not self.active:
            return

        nPre = init.get_N(conf_pre)
        nPost = init.get_N(conf_post)
        delaymap = init.get_delaymap(projections, dt, conf_pre, conf_post)
        self.register_buffer('delaymap', delaymap, persistent=False)
        wmax = init.expand_to_synapses(projections, nPre, nPost, 'wmax')
        wmin = init.expand_to_synapses(projections, nPre, nPost, 'wmin')
        self.register_buffer('wmin', wmin, persistent=False)
        self.register_buffer('wmax', wmax, persistent=False)

        # Weights
        bw = 0 if shared_weights else batch_size
        w = init.build_connectivity(projections, nPre, nPost, bw)
        w = torch.where(
            w==0, w, self.wmin + w * (self.wmax-self.wmin))
        self.W = torch.nn.Parameter(w)
        signs = init.expand_to_neurons(conf_pre, 'sign').to(torch.int8)
        self.register_buffer('signs_pre', signs, persistent = False)
        self.register_buffer('signs', torch.zeros_like(w), persistent = False)

        if ltp is not None:
            STDP_frac = init.expand_to_synapses(
                projections, nPre, nPost, 'STDP_frac')
            if not torch.any(STDP_frac > 0):
                ltp = None
            else:
                self.register_buffer('STDP_frac', STDP_frac, persistent = False)

        self.shortterm = stp
        self.longterm = ltp
        self.current = current

        self.reset()

    def reset(self):
        if self.active:
            self.align_signs()
            if self.shortterm is not None:
                self.shortterm.reset()
            if self.longterm is not None:
                self.longterm.reset(self)
            if self.current is not None:
                self.current.reset()

    def forward(self, Xd, X, Vpost):
        '''
        Xd: (delay, batch, pre)
        X: (batch, post)
        Vpost: (batch, post)
        Output: Current (batch, post)
        '''
        if not self.active:
            return torch.zeros_like(Vpost)

        # LTP
        if self.longterm is not None:
            Wlong = self.longterm(Xd, X, Vpost)
            W = self.signs * \
                (self.W * (1-self.STDP_frac) + Wlong * self.STDP_frac)
            WD = 'beo'
        else:
            W = self.signs * self.W
            WD = 'eo' if len(self.W.shape) == 2 else 'beo'

        # STP
        if self.shortterm is not None:
            Xd *= self.shortterm(Xd)+1 # dbe

        # Integrate
        I = internal_forward(WD, W, Xd)

        # Current filter
        if self.current is not None:
            I = self.current(I)

        return I

    def internal_forward(self, WD, W, Xd):
        return torch.einsum(
            f'{WD}, dbe, deo          ->bo',
             W,     Xd,  self.delaymap)

    def load_state_dict(self, *args, **kwargs):
        super(DeltaSynapse, self).load_state_dict(*args, **kwargs)
        self.align_signs()

    def align_signs(self):
        if not self.active:
            return
        signs = self.signs_pre.unsqueeze(1).expand_as(self.signs)
        signs = torch.where(self.W>0, signs, torch.zeros_like(signs))
        self.signs = signs

    def weight(self):
        return self.W