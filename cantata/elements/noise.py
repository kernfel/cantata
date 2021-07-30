import torch
from cantata import init, elements as ce


class Noise(ce.Module):
    '''
    Poisson background noise
    Input: -
    Output: Current
    Internal state: -
    '''

    def __init__(self, conf, batch_size, dt, train_weight=False,
                 disable_training=False, **kwargs):
        super(Noise, self).__init__()

        # Parameters
        N = init.expand_to_neurons(conf, 'noise_N').expand(
            batch_size, init.get_N(conf)).to(torch.get_default_dtype())
        p = (init.expand_to_neurons(conf, 'noise_rate') * dt).clamp(0, 1)
        W = init.expand_to_neurons(conf, 'noise_weight')
        self.active = torch.any((W > 0) * (N[0, :] > 0) * (p > 0))
        if self.active:
            self.register_buffer('N', N, persistent=False)
            self.register_buffer('p', p, persistent=False)
            if train_weight and not disable_training:
                self.W = torch.nn.Parameter(W)
            else:
                self.register_buffer('W', W, persistent=False)
        else:
            self.register_buffer('N', torch.zeros_like(N), persistent=False)

    def forward(self):
        if self.active:
            current = torch.binomial(self.N, self.p.expand_as(self.N))
            return current * self.W
        else:
            return self.N
