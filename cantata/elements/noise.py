import torch
from cantata import util, init

class Noise(torch.nn.Module):
    '''
    Poisson background noise
    Input: -
    Output: Current
    Internal state: -
    '''
    def __init__(self, conf, batch_size, N, dt):
        super(Noise, self).__init__()

        # Parameters
        Ns = init.expand_to_neurons(conf, 'N').expand(batch_size, N)
        p = (init.expand_to_neurons(conf, 'rate') * dt).clamp(0,1)
        W = init.expand_to_neurons(conf, 'weight')
        self.active = torch.any((W > 0) * (Ns[0,:] > 0) * (p > 0))
        if self.active:
            self.register_buffer('N', Ns, persistent = False)
            self.register_buffer('p', p, persistent = False)
            self.register_buffer('W', W, persistent = False)
        else:
            self.register_buffer('N', torch.zeros_like(Ns), persistent=False)

    def forward(self):
        if self.active:
            return torch.binomial(self.N, self.p) * self.W
        else:
            return self.N
