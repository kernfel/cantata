import torch
from cantata import init
import cantata.elements as ce


class Synapse(ce.Module):
    '''
    Synapse with optional current, short- and long-term plasticity submodules
    Input: Delayed presynaptic spikes; postsynaptic spikes; postsyn voltage
    Output: Synaptic currents
    '''

    def __init__(self, W, signs_pre, delaymap=None, wmin=None, wmax=None,
                 current=None, stp=None, ltp=None, STDP_frac=None):
        super().__init__()
        self.active = W is not None
        if not self.active:
            return
        self.register_parabuf('W', W)
        if delaymap is None:
            delaymap = torch.ones(1, *W.shape[-2:])
        self.register_buffer('delaymap', delaymap, persistent=False)
        self.register_buffer('signs_pre', signs_pre, persistent=False)
        self.register_buffer('signs', torch.zeros_like(W), persistent=False)
        self.register_buffer('wmin', wmin, persistent=False)
        self.register_buffer('wmax', wmax, persistent=False)

        if ltp is not None:
            if not torch.any(STDP_frac > 0):
                ltp = None
            else:
                self.register_buffer('STDP_frac', STDP_frac, persistent=False)
        self.shortterm = stp
        self.longterm = ltp
        self.current = current

        self.reset()

    @classmethod
    def configured(cls, projections, conf_pre, conf_post, batch_size, dt,
                   stp=None, ltp=None, current=None,
                   shared_weights=True,
                   train_weight=True, disable_training=False, **kwargs):
        active = len(projections[0]) > 0
        if not active:
            ret = cls(None, None, None)
            ret.projections = projections
            return ret

        nPre = init.get_N(conf_pre)
        nPost = nPre if conf_post is None else init.get_N(conf_post)
        delaymap = init.get_delaymap(projections, dt, conf_pre, conf_post)
        wmax = init.expand_to_synapses(projections, nPre, nPost, 'wmax')
        wmin = init.expand_to_synapses(projections, nPre, nPost, 'wmin')

        # Weights
        bw = 0 if shared_weights else batch_size
        w = init.build_connectivity(projections, nPre, nPost, bw)
        w = torch.where(
            w == 0, w, wmin + w * (wmax-wmin))
        if train_weight and not disable_training:
            w = torch.nn.Parameter(w)
        signs_pre = init.expand_to_neurons(conf_pre, 'sign').to(torch.int8)

        if ltp is not None:
            STDP_frac = init.expand_to_synapses(
                projections, nPre, nPost, 'STDP_frac')
        else:
            STDP_frac = None

        ret = cls(w, signs_pre, delaymap=delaymap, wmin=wmin, wmax=wmax,
                  current=current, stp=stp, ltp=ltp, STDP_frac=STDP_frac)
        ret.projections = projections
        return ret

    def reset(self, keep_values=False):
        if self.active:
            self.align_signs()
            if self.shortterm is not None:
                self.shortterm.reset(keep_values)
            if self.longterm is not None:
                self.longterm.reset(self, keep_values)
            if self.current is not None:
                self.current.reset(keep_values)

    def forward(self, Xd, X=None, Vpost=None):
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
                (self.weight() * (1-self.STDP_frac) + Wlong * self.STDP_frac)
            WD = 'beo'
        else:
            W = self.signs * self.weight()
            WD = 'eo' if len(self.W.shape) == 2 else 'beo'

        # STP
        if self.shortterm is not None:
            Xd = Xd * (self.shortterm(Xd)+1)  # dbe

        # Integrate
        output = self.internal_forward(WD, W, Xd)

        # Current filter
        if self.current is not None:
            output = self.current(output)

        return output

    def internal_forward(self, WD, W, Xd):
        return torch.einsum(
            f'{WD}, dbe, deo          ->bo',
            W,      Xd,  self.delaymap)

    def load_state_dict(self, *args, **kwargs):
        super(Synapse, self).load_state_dict(*args, **kwargs)
        self.align_signs()

    def align_signs(self):
        if not self.active:
            return
        signs = self.signs_pre.unsqueeze(1).expand_as(self.signs)
        signs = torch.where(self.W != 0, signs, torch.zeros_like(signs))
        self.signs = signs

    def weight(self):
        return torch.abs(self.W)
