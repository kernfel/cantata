import torch
from box import Box
from cantata import util, init
import cantata.elements as ce

class SNN(torch.nn.Module):
    '''
    SNN layer.
    Input: Feedforward spikes
    Output: Output spikes
    Internal state: -
    '''
    def __init__(self, conf, batch_size, dt,
                 input_areas = {}, name = '', Synapse = ce.DeltaSynapse,
                 **kwargs):
        super(SNN, self).__init__()

        self.N = init.get_N(conf)
        self.name = name
        self.p_names, self.p_idx = init.build_population_indices(conf)

        self.spikes = ce.ALIFSpikes(conf, batch_size, dt)
        self.membrane = ce.Membrane(conf, batch_size, dt)
        self.synapses_int = Synapse(conf, batch_size, dt, **kwargs)

        self.synapses_ext = [] # ModuleList complicates reset()
        for name_pre, conf_pre in input_areas.items():
            syn = Synapse(
                conf_pre, batch_size, dt, conf, self.name, **kwargs)
            if not syn.active or name_pre == self.name:
                self.synapses_ext.append(None)
            else:
                setattr(self, f'synapse_{name_pre}', syn)
                self.synapses_ext.append(syn)

        self.reset()

    def reset(self):
        X, Xd, Xd_xarea = self.spikes(self.membrane.V)
        for m in self.children():
            m.reset()
        return (
            torch.zeros_like(X),
            None if Xd_xarea is None else torch.zeros_like(Xd_xarea)
        )

    def forward(self, *Xext):
        V = self.membrane.V
        X, Xd, Xd_xarea = self.spikes(V)
        current = self.synapses_int(Xd, X, V)
        for X_ext, syn_ext in zip(Xext, self.synapses_ext):
            if syn_ext is not None:
                current += syn_ext(X_ext, X, V)
        self.membrane(X, current)
        return X, Xd_xarea
