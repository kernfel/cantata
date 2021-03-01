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
        ext_delays = init.get_external_delays(conf, dt)

        self.spikes = ce.ALIFSpikes(
            delays, ext_delays, conf, batch_size, self.N, dt)
        self.membrane = ce.Membrane(conf, batch_size, self.N, dt)
        self.synapses_int = ce.DeltaSynapse(
            projections, dmap, conf, STDP, batch_size, self.N, dt)

        self.synapses_ext = [] # ModuleList complicates reset()
        self.buffers_ext = []
        for area, aconf in sconf.items():
            proj_xarea = build_projections_xarea(sconf, area, self.name)
            if len(proj_xarea[0] > 0) or area == self.name:
                self.synapses_ext.append(None)
                self.buffers_ext.append(None)
            else:
                delays_xarea = get_delays_xarea(aconf, dt)
                nPre = sum([p.n for p in aconf.populations.values()])
                dmap_xarea = get_delaymap_xarea(
                    proj_xarea, delays_xarea, nPre, self.N, dt)

                buffer_name = f'dmap_{area}'
                self.register_buffer(buffer_name, dmap_xarea)
                self.buffer_ext.append(buffer_name)

                model_name = f'synapse_{area}'
                syn = ce.DeltaSynapse(
                    # TODO
                )
                setattr(self, f'synapse_{area}', syn)
                self.synapses_ext.append(syn)

        self.reset()

    def reset(self):
        for m in self.children():
            m.reset()

    def forward(self, *Xext):
        V = self.membrane.V
        X, Xd = self.spikes(V)
        current = self.synapses_int(Xd, X, V)
        for syn_ext, buffer_name in zip(self.synapses_ext, self.buffers_ext):
            if syn_ext is not None:
                current += syn_ext(Xext, X, V)
        self.membrane(X, current)
        return X
