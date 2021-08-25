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

    def __init__(self, N, batch_size, delays=None, amplitude=None, alpha=None):
        super().__init__()
        self.adaptive = amplitude is not None and torch.any(amplitude > 0)
        if self.adaptive:
            self.surrogate = AdaptiveSurrGradSpike
            self.register_parabuf('alpha', alpha, persistent=False)
            self.register_parabuf('amplitude', amplitude, persistent=False)
            self.register_buffer('threshold', torch.zeros(batch_size, N))
        else:
            self.surrogate = SurrGradSpike
        if delays is None:
            delays = [0]
        self.spike_buffer = ce.DelayBuffer((batch_size, N), delays)

    @classmethod
    def configured(cls, conf, batch_size, dt,
                   train_tau_th=False, train_amplitude=False,
                   disable_training=False):
        N = init.get_N(conf)
        delays = init.get_delays(conf, dt, False)
        amplitude = init.expand_to_neurons(conf, 'th_ampl')
        adaptive = torch.any(amplitude > 0)
        if adaptive:
            tau = init.expand_to_neurons(conf, 'th_tau')
            alpha = util.decayconst(tau, dt)
            if train_tau_th and not disable_training:
                alpha = torch.nn.Parameter(alpha)
            if train_amplitude and not disable_training:
                amplitude = torch.nn.Parameter(amplitude)
            return cls(N, batch_size, delays=delays,
                       amplitude=amplitude, alpha=alpha)
        else:
            return cls(N, batch_size, delays=delays)

    def reset(self, keep_values=False):
        if self.adaptive:
            if keep_values:
                self.threshold = self.threshold.detach()
            else:
                self.threshold = torch.zeros_like(self.threshold)
        self.spike_buffer.reset(keep_values)
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
            X = self.surrogate.apply(V, (1 + self.threshold))
            self.threshold = self.threshold * self.alpha + X * self.amplitude
        else:
            X = self.surrogate.apply(V-1)
        Xd, = self.spike_buffer(X)
        return X, Xd


class SurrGradSpike(torch.autograd.Function):
    '''
    Fast-sigmoid surrogate gradient as in SuperSpike
        Zenke & Ganguli 2018
        Zenke & Vogels 2020
    Note: Zenke & Vogels 2020 use the term "scale" in two different ways.
        Here, `scale` refers to their main usage (cf. Fig 3), that is, to how
        quickly the gradient drops off as the voltage recedes from the
        threshold (cf. Fig 3).
        Conversely, `gain` refers to the magnitude of the transmitted gradient,
        as well as the maximum gradient (at V==threshold), cf. Fig 5.
        In other words, `gain` scales the gradient, whereas `scale` scales the
        voltage.
    '''
    scale = 10.0
    gain = 0.5

    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        out = torch.zeros_like(voltage)
        out[voltage >= 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = SurrGradSpike.gain * grad_input \
            / (SurrGradSpike.scale*torch.abs(input - 1)+1.0)**2
        return grad


class AdaptiveSurrGradSpike(torch.autograd.Function):
    '''
    Fast-sigmoid surrogate gradient with a variable threshold
        Zenke & Ganguli 2018
        Zenke & Vogels 2020
        Salaj, ..., Maass 2021
    '''
    scale = 10.0
    gain = 0.5

    @staticmethod
    def forward(ctx, voltage, threshold):
        ctx.save_for_backward(voltage, threshold)
        out = torch.zeros_like(voltage)
        out[voltage >= threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        V_norm = (input - threshold) / threshold
        grad = AdaptiveSurrGradSpike.gain * grad_input \
            / (AdaptiveSurrGradSpike.scale*torch.abs(V_norm)+1.0)**2
        return grad, -grad
