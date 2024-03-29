import torch
from cantata import util, init, elements as ce


class STP(ce.Module):
    '''
    Short-term plasticity
    Input: Delayed spikes
    Output: Plasticity factors
    Internal state: Activity traces
    '''

    def __init__(self, conf, n_delays, batch_size, dt):
        super(STP, self).__init__()

        # Parameters
        p = init.expand_to_neurons(conf, 'p')
        self.active = torch.any(p != 0)
        if self.active:
            self.register_buffer('p', p, persistent=False)
            alpha = util.decayconst(init.expand_to_neurons(conf, 'tau_r'), dt)
            self.register_buffer('alpha', alpha, persistent=False)

        # State
        self.register_buffer(
            'Ws', torch.zeros(n_delays, batch_size, init.get_N(conf)))

    def reset(self, keep_values=False):
        if self.active:
            if keep_values:
                self.Ws = self.Ws.detach()
            else:
                self.Ws = torch.zeros_like(self.Ws)

    def forward(self, Xd):
        '''
        Xd: (delay, batch, pre)
        Output: Plasticity factor before the update (delay, batch, pre)
        '''
        out = self.Ws.clone()
        if self.active:
            dW = Xd * self.p * (1 + (self.p < 0)*self.Ws)
            self.Ws = self.Ws * self.alpha + dW
        return out
