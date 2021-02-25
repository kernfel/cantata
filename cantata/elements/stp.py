import torch
from cantata import util, init, cfg

class STP(torch.nn.Module):
    '''
    Short-term plasticity
    Input: Delayed spikes
    Output: Plasticity factors
    Internal state: Activity traces
    '''
    def __init__(self, n_delays):
        super(STP, self).__init__()
        N = init.get_N()

        # Parameters
        self.register_buffer('p', init.expand_to_neurons('p'), persistent = False)
        self.alpha = util.decayconst(cfg.model.tau_r)

        # State
        self.register_buffer('Ws', torch.zeros(n_delays, cfg.batch_size, N))

    def forward(self, Xd):
        '''
        Xd: (delay, batch, pre)
        Output: Plasticity factor before the update (delay, batch, pre)
        '''
        out = self.Ws.clone()
        dW = Xd * self.p * (1 + self.depr_mask*self.Ws)
        self.Ws = self.Ws * self.alpha + dW
        return out
