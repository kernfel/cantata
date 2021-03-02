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
    def __init__(self, conf, dt, name = 'Input'):
        super(PoissonInput, self).__init__()
        self.N = init.get_N(conf)
        self.dt, self.name = dt, name
        self.p_names, self.p_idx = init.build_population_indices(conf)

        # Add 1 to account for immediate delivery of truly delayed connections.
        delays = [d+1 for d in init.get_delays(conf, dt, True)]
        if delays[0] == 2: # Remove minimal delay, deliver it immediately
            delays = delays[1:]
        self.buffered = len(delays)
        if self.buffered:
            self.rate_buffer = ce.DelayBuffer((batch_size,conf.n_channels), delays)

        cmap = torch.zeros(conf.n_channels, self.N)
        for pname, pidx in zip(self.p_names, self.p_idx):
            cmap[conf.populations[pname].channel, pidx] = 1
        self.register_buffer('cmap', cmap, persistent = False)

    def forward(self, rates):
        '''
        rates, in Hz: (batch, channels)
        Output, in spikes: (1, batch, neurons)
        '''
        norm_rates = torch.clip(rates * self.dt, 0, 1)
        if self.buffered:
            delayed_rates, = self.rate_buffer(norm_rates)
            norm_rates = torch.cat((
                norm_rates.unsqueeze(0),
                delayed_rates
            ), dim=0)
        else:
            norm_rates = norm_rates.unsqueeze(0)
        neuron_rates = torch.matmul(norm_rates, self.cmap) # dbc,cn->dbn
        return torch.bernoulli(neuron_rates)
