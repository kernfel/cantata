import torch
from torch.nn.functional import relu
import numpy as np
from box import Box
from cantata import util, init, cfg

class Spikes(torch.nn.Module):
    '''
    Surrogate gradient spike function.
    Input: Membrane potentials
    Output: List of present and delayed spikes: [X_t, X_d_0, ..., X_d_i]
    Internal state: threshold, delay_{d}
    '''
    def __init__(self, delays):
        super(Spikes, self).__init__()
        shape = (cfg.batch_size, init.get_N())

        amplitude = init.expand_to_neurons('th_ampl')
        self.adaptive = torch.any(amplitude>0)
        if self.adaptive:
            tau = init.expand_to_neurons('th_tau')
            self.alpha = util.decayconst(tau)
            self.amplitude = amplitude
            self.register_buffer('threshold', torch.zeros(shape))

        self.delay = len(delays) > 0
        if self.delay:
            self.t = 0
            self.delays = delays
            self.max_delay = max(delays)
            for d in range(self.max_delay):
                self.register_buffer(f'delay_{d}', torch.zeros(shape))

    def forward(self, mem):
        if self.adaptive:
            mthr = mem - (self.threshold + 1)
            X = SurrGradSpike.apply(mthr)
            self.threshold = self.threshold * self.alpha + X * self.amplitude
        else:
            mthr = mem - 1
            X = SurrGradSpike.apply(mthr)

        out = [X]
        if self.delay:
            for d in self.delays:
                out.append(getattr(self, f'delay_{(self.t-d) % self.max_delay}'))
            setattr(self, f'delay_{self.t % self.max_delay}', X)
            self.t = self.t + 1
        return out



class SurrGradSpike(torch.autograd.Function):
    scale = 100.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input >= 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
