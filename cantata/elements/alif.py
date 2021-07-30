import torch
from cantata import util, init
import cantata.elements as ce


class ALIFSpikes(ce.Module):
    '''
    Surrogate gradient spike function with an adaptive threshold
    Input: Membrane potentials
    Output: Tuple of present and delayed spikes: (X(t), X(d_0, .. d_i))
    Internal state: threshold, delay buffers
    '''

    def __init__(self, conf, batch_size, dt,
                 train_tau=False, train_amplitude=False,
                 disable_training=False):
        super(ALIFSpikes, self).__init__()
        N = init.get_N(conf)
        amplitude = init.expand_to_neurons(conf, 'th_ampl')
        self.adaptive = torch.any(amplitude > 0)
        if self.adaptive:
            tau = init.expand_to_neurons(conf, 'th_tau')
            alpha = util.decayconst(tau, dt)
            if train_tau and not disable_training:
                self.alpha = torch.nn.Parameter(alpha)
            else:
                self.register_buffer('alpha', alpha, persistent=False)
            if train_amplitude and not disable_training:
                self.amplitude = torch.nn.Parameter(amplitude)
            else:
                self.register_buffer('amplitude', amplitude, persistent=False)
            self.register_buffer('threshold', torch.zeros(batch_size, N))

        delays = init.get_delays(conf, dt, False)
        self.spike_buffer = ce.DelayBuffer((batch_size, N), delays)

    def reset(self):
        if self.adaptive:
            self.threshold = torch.zeros_like(self.threshold)
        self.spike_buffer.reset()

    def forward(self, V):
        '''
        V: (batch, pre)
        Output:
            X: (batch, pre)
            Xd: (delay, batch, pre)
        '''
        if self.adaptive:
            mthr = V - (self.threshold + 1)
            X = SurrGradSpike.apply(mthr)
            self.threshold = self.threshold * self.alpha + X * self.amplitude
        else:
            mthr = V - 1
            X = SurrGradSpike.apply(mthr)
        Xd, = self.spike_buffer(X)
        return X, Xd


class SurrGradSpike(torch.autograd.Function):
    '''
    Fast-sigmoid surrogate gradient as in SuperSpike
        Zenke & Ganguli 2018
        Zenke & Vogels 2020
    '''
    scale = 10.0

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
