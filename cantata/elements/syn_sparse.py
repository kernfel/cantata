import torch
from cantata import util, init
import cantata.elements as ce

class DeltaSynapse(torch.nn.Module):
    '''
    Delta synapse with short- and long-term plasticity submodules
    Input: Delayed presynaptic spikes; postsynaptic spikes; postsynaptic voltage
    Output: Synaptic currents
    Internal state: Weights W
    '''
    def __init__(self, conf_pre, batch_size, dt,
                 conf_post = None, name_post = None,
                 STDP = ce.Abbott, shared_weights = True, **kwargs):
        super(DeltaSynapse, self).__init__()
        projections = init.build_projections(conf_pre, conf_post, name_post)
        self.active = len(projections[0]) > 0
        if not self.active:
            return

        xarea = conf_post is not None
        nPre = init.get_N(conf_pre)
        nPost = init.get_N(conf_post) if xarea else nPre
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

        # Short-term plasticity
        shortterm = ce.STP(conf_pre, delaymap.shape[0], batch_size, dt)
        self.has_STP = shortterm.active
        if self.has_STP:
            self.shortterm = shortterm

        # Long-term plasticity
        if STDP is None:
            self.has_STDP = False
        else:
            STDP_frac = init.expand_to_synapses(
                projections, nPre, nPost, 'STDP_frac')
            longterm = STDP(
                projections, self, conf_post if xarea else conf_pre,
                batch_size, nPre, nPost, dt)
            self.has_STDP = torch.any(STDP_frac > 0) and longterm.active
        if self.has_STDP:
            self.register_buffer('STDP_frac', STDP_frac, persistent = False)
            self.longterm = longterm

        self.reset()

    # def reset(self):
    #     if self.active:
    #         self.align_signs()
    #         if self.has_STP:
    #             self.shortterm.reset()
    #         if self.has_STDP:
    #             self.longterm.reset(self.W)

    def __init__(self, conf_pre, batch_size, dt,
                 conf_post = None, name_post = None,
                 STDP = ce.Abbott, shared_weights = True):
        super(DeltaSynapse, self).__init__()
        projections = init.build_projections(conf_pre, conf_post, name_post)
        self.active = len(projections[0]) > 0
        if not self.active:
            return
        self.shared_weights = shared_weights

        iPre, iPost, iDelay = [], [], []
        for idx, p in zip(*projections):
            iPre.append(torch.tensor(idx[0]).flatten())
            iPost.append(torch.tensor(idx[1]).flatten())
            d = init.get_delay(p.delay, dt, False)
            iDelay.append(torch.ones(idx[0].size, dtype=torch.int64) * d)
        self.indices = dict(
            pre = torch.cat(iPre),
            post = torch.cat(iPost),
            delay = torch.cat(iDelay),
            batch = torch.arange(batch_size).reshape(-1,1)
        )

        # Weights
        xarea = conf_post is not None
        nPre = init.get_N(conf_pre)
        nPost = init.get_N(conf_post) if xarea else nPre
        wmax = init.expand_to_synapses(projections, nPre, nPost, 'wmax')
        wmin = init.expand_to_synapses(projections, nPre, nPost, 'wmin')
        # self.register_buffer('wmin', wmin, persistent=False)
        # self.register_buffer('wmax', wmax, persistent=False)

        bw = 0 if shared_weights else batch_size
        w = init.build_connectivity(projections, nPre, nPost, bw)
        w = torch.where(
            w==0, w, self.wmin + w * (self.wmax-self.wmin))
        if shared_weights:
            w.unsqueeze_(0)

        offset = torch.zeros(2,1, dtype=torch.long)
        W_values, W_indices = [],[]
        for batch in range(1 if shared_weights else batch_size)
            batch_tpl = () if shared_weights else (batch,)
            for idx, p in zip(*projections):
                block = w[batch_tpl + idx].to_sparse()
                W_values.append(block.values())
                W_indices.append(block.indices() + offset)
                offset[0] += idx[0].size
                offset[1] += idx[1].size
        W_values = torch.cat(W_values)
        W_indices = torch.cat(W_indices, dim=-1)
        # init.build_connectivity gives W as ([b,]pre,post), but we need it
        # as (post, pre), so:
        W_indices = torch.stack([W_indices[1], W_indices[0]])
        W_sparse = torch.sparse_coo_tensor(W_indices, W_values, tuple(offset))


        self.W = torch.nn.Parameter(w)
        signs = init.expand_to_neurons(conf_pre, 'sign').to(torch.int8)
        self.register_buffer('signs_pre', signs, persistent = False)
        self.register_buffer('signs', torch.zeros_like(w), persistent = False)

    def forward(self, Xd, X, Vpost):
        if not self.active:
            return torch.zeros_like(Vpost)

        # Step 1: Pick from Xd (d,b,pre) to Xpre (b,nq)
        Xpre = Xd[
            self.indices['delay'],
            self.indices['batch'],
            self.indices['pre']
        ]
        # Step 2: Apply STP
        Spre = shortterm(Xpre) # TODO
        # Step 3: Apply STDP
        if has_STDP:
            W = self.longterm(Xpre, X, Vpost) # TODO
        else
            W = self.W_sparse
        # Step 4: Compute synaptic impulses
        if self.shared_weights:
            Spost = torch.sparse.mm(W, Spre.T).T
        else:
            Spost = torch.sparse.mm(W, Spre.reshape(-1,1))
            Spost = Spost.reshape(self.indices['batch'].shape[1], -1)
        # Step 5: Apply current filter
        Ipost = self.current(Spost) # TODO
        # Step 6: Recombine from Ipost (b,nq) to I (b,post)
        I = torch.zeros_like(Vpost)
        I.index_put_(
            [self.indices['batch'], self.indices['post']],
            Ipost, accumulate = True)
        return I

    # def forward(self, Xd, X, Vpost):
    #     '''
    #     Xd: (delay, batch, pre)
    #     X: (batch, post)
    #     Vpost: (batch, post)
    #     Output: Current (batch, post)
    #     '''
    #     if not self.active:
    #         return torch.zeros_like(Vpost)
    #     if self.has_STDP:
    #         Wlong = self.longterm(Xd, X, Vpost)
    #         W = self.signs * \
    #             (self.W * (1-self.STDP_frac) + Wlong * self.STDP_frac)
    #         WD = 'beo'
    #     else:
    #         W = self.signs * self.W
    #         WD = 'eo' if len(self.W.shape) == 2 else 'beo'
    #
    #     if self.has_STP:
    #         Wshort = self.shortterm(Xd)+1
    #         I = torch.einsum(
    #             f'{WD}, dbe, deo,           dbe   ->bo',
    #              W,     Xd,  self.delaymap, Wshort)
    #     else:
    #         I = torch.einsum(
    #             f'{WD}, dbe, deo          ->bo',
    #              W,     Xd,  self.delaymap)
    #     return I
    #
    # def load_state_dict(self, *args, **kwargs):
    #     super(DeltaSynapse, self).load_state_dict(*args, **kwargs)
    #     self.align_signs()
    #
    # def align_signs(self):
    #     if not self.active:
    #         return
    #     signs = self.signs_pre.unsqueeze(1).expand_as(self.signs)
    #     signs = torch.where(self.W>0, signs, torch.zeros_like(signs))
    #     self.signs = signs
