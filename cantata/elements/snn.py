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

        self.reset()

    def reset(self):
        for m in self.children():
            m.reset()

    def forward(self, FF):
        V = self.membrane.V
        X, Xd = self.spikes(V)
        I_int = self.synapses_int(Xd, X, V)
        I_ff = self.synapses_ff(FF, X, V)
        current = I_int + I_ff
        self.membrane(X, current)
        return X
