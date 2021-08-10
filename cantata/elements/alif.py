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
                 train_tau_th=False, train_amplitude=False,
                 disable_training=False):
        super(ALIFSpikes, self).__init__()
        N = init.get_N(conf)
        amplitude = init.expand_to_neurons(conf, 'th_ampl')
        self.adaptive = torch.any(amplitude > 0)
        if self.adaptive:
            tau = init.expand_to_neurons(conf, 'th_tau')
            alpha = util.decayconst(tau, dt)
            if train_tau_th and not disable_training:
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
        if isinstance(self.alpha, torch.nn.Parameter):
            self.alpha.data.clamp_(0., 1.)

    def forward(self, V):
        '''
        V: (batch, pre)
        Output:
            X: (batch, pre)
            Xd: (delay, batch, pre)
        '''
        if self.adaptive:
            X = SurrGradSpike.apply(V, (1 + self.threshold))
            self.threshold = \
                self.alpha * self.threshold \
                + (1-self.alpha) * X * self.amplitude
        else:
            X = SurrGradSpike.apply(V)
        Xd, = self.spike_buffer(X)
        return X, Xd


class SurrGradSpike(torch.autograd.Function):
    '''
    Fast-sigmoid surrogate gradient as in SuperSpike
        Zenke & Ganguli 2018
        Zenke & Vogels 2020
    Optional variable threshold for appropriately scaled backward pass, cf.
        Salaj, ..., Maass 2021
    '''
    scale = 10.0

    @staticmethod
    def forward(ctx, voltage, threshold=None):
        ctx.save_for_backward(voltage, threshold)
        out = torch.zeros_like(voltage)
        if threshold is None:
            threshold = 1
        out[voltage >= threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        has_threshold = threshold is not None
        if not has_threshold:
            threshold = 1
        V_norm = (input - threshold) / threshold
        grad = grad_input/(SurrGradSpike.scale*torch.abs(V_norm)+1.0)**2
        return (grad, -grad) if has_threshold else grad
