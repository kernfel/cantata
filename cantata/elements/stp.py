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
        p = init.expand_to_neurons('p')
        self.alpha = util.decayconst(cfg.model.tau_r)
        self.active = torch.any(p != 0)
        if self.active:
            self.register_buffer('p', p, persistent = False)

        # State
        self.register_buffer('Ws', torch.zeros(n_delays, cfg.batch_size, N))

    def forward(self, Xd):
        '''
        Xd: (delay, batch, pre)
        Output: Plasticity factor before the update (delay, batch, pre)
        '''
        out = self.Ws.clone()
        if self.active:
            dW = Xd * self.p * (1 + self.depr_mask*self.Ws)
            self.Ws = self.Ws * self.alpha + dW
        return out
