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
    def __init__(self, conf_all, STDP, batch_size, dt, name):
        super(SNN, self).__init__()
        conf = conf_all.areas[name]

        self.N = init.get_N(conf)
        self.name = name
        self.p_names, self.p_idx = init.build_population_indices(conf)

        self.spikes = ce.ALIFSpikes(conf, batch_size, self.N, dt)
        self.membrane = ce.Membrane(conf, batch_size, self.N, dt)
        self.synapses_int = ce.DeltaSynapse(conf, STDP, batch_size, dt)

        self.synapses_ext = [] # ModuleList complicates reset()
        all_areas = Box({'__input__': conf_all.input}) + conf_all.areas
        for name_pre, conf_pre in all_areas.items():
            syn = ce.DeltaSynapse(
                conf_pre, STDP, batch_size, dt, conf, self.name)
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
        for i, syn_ext in enumerate(self.synapses_ext):
            if syn_ext is not None:
                current += syn_ext(Xext[i], X, V)
        self.membrane(X, current)
        return X, Xd_xarea
