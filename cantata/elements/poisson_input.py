import torch
from cantata import util, init, cfg

class PoissonInput(torch.nn.Module):
    '''
    Input layer, transforms input rate to poisson spikes
    Input: Rates
    Output: Spikes
    Internal state: -
    '''
    def __init__(self, conf, dt, name = 'Input'):
        super(PoissonInput, self).__init__()
        conf.N = sum([p.n for p in conf.populations.values()])
        self.dt, self.name = dt, name
        self.p_names, self.p_idx = init.build_population_indices(conf)
        cmap = torch.zeros(conf.n_channels, conf.N)
        for pname, pidx in zip(self.p_names, self.p_idx):
            cmap[conf.populations.pname.channel, pidx] = 1
        self.register_buffer('cmap', cmap, persistent = False)

    def forward(self, rates):
        '''
        rates, in Hz: (batch, channels)
        Output, in spikes: (batch, neurons)
        '''
        norm_rates = torch.clip(rates * self.dt, 0, 1)
        neuron_rates = norm_rates @ self.cmap # bc,cn->bn
        return torch.bernoulli(neuron_rates)
