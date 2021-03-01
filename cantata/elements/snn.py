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
    def __init__(self, sconf, STDP, batch_size, dt, name):
        super(SNN, self).__init__()
        conf = sconf[name]

        self.N = sum([p.n for p in conf.populations.values()])
        self.name = name
        self.p_names, self.p_idx = init.build_population_indices(conf)
        projections = init.build_projections(conf, self.p_names, self.p_idx)
        dmap, delays = init.build_delay_mapping(projections, self.N, self.N, dt)
        delays_xarea = init.get_delays_xarea(conf, dt)

        self.spikes = ce.ALIFSpikes(
            delays, delays_xarea, conf, batch_size, self.N, dt)
        self.membrane = ce.Membrane(conf, batch_size, self.N, dt)
        self.synapses_int = ce.DeltaSynapse(
            projections, dmap, conf, conf, STDP, batch_size, self.N, self.N, dt)

        self.synapses_ext = [] # ModuleList complicates reset()
        for area, aconf in sconf.items():
            proj_xarea = build_projections_xarea(sconf, area, self.name)
            if len(proj_xarea[0] > 0) or area == self.name:
                self.synapses_ext.append(None)
            else:
                delays_xarea = get_delays_xarea(aconf, dt)
                nPre = sum([p.n for p in aconf.populations.values()])
                dmap_xarea = get_delaymap_xarea(
                    proj_xarea, delays_xarea, nPre, self.N, dt)
                syn = ce.DeltaSynapse(
                    proj_xarea, dmap_xarea, aconf, conf, STDP,
                    batch_size, nPre, self.N, dt
                )
                setattr(self, f'synapse_{area}', syn)
                self.synapses_ext.append(syn)

        self.reset()

    def reset(self):
        for m in self.children():
            m.reset()

    def forward(self, *Xext):
        V = self.membrane.V
        X, Xd, Xd_xarea = self.spikes(V)
        current = self.synapses_int(Xd, X, V)
        for i, syn_ext in enumerate(self.synapses_ext):
            if syn_ext is not None:
                current += syn_ext(Xext[i], X, V)
        self.membrane(X, current)
        return X, Xd_xarea
