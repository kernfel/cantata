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
        self.synapses_int = ce.DeltaSynapse(projections, dmap)
        self.synapses_ff = ce.DeltaSynapse('TODO')

    def forward(self, FF):
        X, Xd = self.spikes(self.membrane.V)
        I_int = self.synapses_int(Xd, X)
        I_ff = self.synapses_ff(FF, X)
        current = I_int + I_ff
        self.membrane(X, current)
        return X
