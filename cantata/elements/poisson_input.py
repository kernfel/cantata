import torch
from cantata import init
import cantata.elements as ce

class PoissonInput(torch.nn.Module):
    '''
    Input layer, transforms input rate to poisson spikes
    Input: Rates
    Output: Spikes
    Internal state: -
    '''
    def __init__(self, conf, batch_size, dt, name = 'Input'):
        super(PoissonInput, self).__init__()
        self.N = init.get_N(conf)
        self.dt, self.name = dt, name
        self.p_names, self.p_idx = init.build_population_indices(conf)

        # Add 1 to account for immediate delivery of truly delayed connections,
        # but remove minimal delay
        delays = [d+1 if d>1 else 0 for d in init.get_delays(conf, dt, True)]
        self.spike_buffer = ce.DelayBuffer((batch_size, self.N), delays)

        cmap = torch.zeros(conf.n_channels, self.N)
        for pname, pidx in zip(self.p_names, self.p_idx):
            cmap[conf.populations[pname].channel, pidx] = 1
        self.register_buffer('cmap', cmap, persistent = False)

    def forward(self, rates):
        '''
        rates, in Hz: (batch, channels)
        Output, in spikes: (delays, batch, neurons)
        '''
        norm_rates = torch.clip(rates * self.dt, 0, 1)
        neuron_rates = torch.matmul(norm_rates, self.cmap) # bc,cn->bn
        Xd, = self.spike_buffer(torch.bernoulli(neuron_rates))
        return Xd
