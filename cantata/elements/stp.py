import torch
from cantata import util, init

class STP(torch.nn.Module):
    '''
    Short-term plasticity
    Input: Delayed spikes
    Output: Plasticity factors
    Internal state: Activity traces
    '''
    def __init__(self, conf, n_delays, batch_size, N, dt):
        super(STP, self).__init__()

        # Parameters
        p = init.expand_to_neurons(conf, 'p')
        self.alpha = util.decayconst(conf.tau_r, dt)
        self.active = torch.any(p != 0)
        if self.active:
            self.register_buffer('p', p, persistent = False)

        # State
        self.register_buffer('Ws', torch.zeros(n_delays, batch_size, N))

    def reset(self):
        if self.active:
            torch.nn.init.zeros_(self.Ws)

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
