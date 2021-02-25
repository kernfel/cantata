import torch
from cantata import util, init, cfg
import cantata.elements as ce

class SNN(torch.nn.Module):
    '''
    SNN layer.
    Input: Feedforward spikes
    Output: Output spikes
    Internal state: -
    '''
    def __init__(self):
        super(SNN, self).__init__()

        init.get_N(True) # force update
        self.p_names, self.p_idx = init.build_population_indices()
        projections = init.build_projections(self.p_names, self.p_idx)
        dmap, delays = init.build_delay_mapping(projections)

        self.spikes = ce.ALIFSpikes(delays)
        self.membrane = ce.Membrane()
        self.shortterm = ce.STP()
        if cfg.model.STDP_Clopath:
            self.longterm_int = ce.Clopath('int')
            self.longterm_ff = ce.Clopath('ff')
        else:
            self.longterm_int = ce.Abbott('int')
            self.longterm_ff = ce.Abbott('ff')
        self.synapses_int = ce.DeltaSynapse('int')
        self.synapses_ff = ce.DeltaSynapse('ff')

    def forward(self, FF):
        X, *Xd = self.spikes(self.membrane.V)
        w_short = self.shortterm(Xd)
        w_long_int = self.longterm_int(Xd, X)
        w_long_ff = self.longterm_ff(FF, X)
        current = (
            self.synapses_int(Xd, w_long_int, w_short)
            + self.synapses_ff(FF, w_long_ff)
        )
        self.membrane(X, current)
        return X
