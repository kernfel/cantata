import torch
from torch.nn.functional import relu
import numpy as np
from box import Box
from cantata import util, init, cfg

class SpikeLayer(torch.nn.Module):
    '''
    SNN layer.
    Input: Feedforward spikes
    Output: Output spikes
    Internal state: V, refractory
    '''
    def __init__(self):
        super(SpikeLayer, self).__init__()

        self.p_names, self.p_idx = init.build_population_indices()
        projections = init.build_projections(self.p_names, self.p_idx)
        dmap, delays = init.build_delay_mapping(projections)

        spikes = Spikes(delays)
        shortterm = STP() # mock
        if cfg.model.STDP_Clopath:
            longterm_int = Clopath('int') # mock
            longterm_ff = Clopath('ff') # mock
        else:
            longterm_int = Abbott('int') # mock
            longterm_ff = Abbott('ff') # mock
        synapses_int = Synapse('int') # mock
        synapses_ff = Synapse('ff') # mock

        self.modules = ModuleList([
            spikes,
            shortterm,
            longterm_int,
            longterm_ff,
            synapses_int,
            synapses_ff
        ])

        # TODO: V, refractory.

    def forward(self, FF):
        X, *Xd = self.spikes(self.V)
        w_short = self.shortterm(Xd)
        w_long_int = self.longterm_int(Xd, X)
        w_long_ff = self.longterm_ff(FF, X)
        current = (
            self.synapses_int(Xd, w_long_int, w_short)
            + self.synapses_ff(FF, w_long_ff)
        )
        self.integrate(X, current)
        return X

    def integrate(self, X, current):
        # TODO
