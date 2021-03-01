import torch
from cantata import util, init
import cantata.elements as ce

class SNN(torch.nn.Module):
    '''
    SNN layer.
    Input: Feedforward spikes
    Output: Output spikes
    Internal state: -
    '''
    def __init__(self, conf, STDP, batch_size, dt, name):
        super(SNN, self).__init__()

        self.N = sum([p.n for p in conf.populations.values()])
        self.name = name
        self.p_names, self.p_idx = init.build_population_indices(conf)
        projections = init.build_projections(conf, self.p_names, self.p_idx)
        dmap, delays = init.build_delay_mapping(projections, self.N, self.N, dt)

        self.spikes = ce.ALIFSpikes(delays, conf, batch_size, self.N, dt)
        self.membrane = ce.Membrane(conf, batch_size, self.N, dt)
        self.synapses_int = ce.DeltaSynapse(
            projections, dmap, conf, STDP, batch_size, self.N, dt)
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
