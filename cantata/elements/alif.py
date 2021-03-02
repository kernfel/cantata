import torch
from cantata import util, init

class ALIFSpikes(torch.nn.Module):
    '''
    Surrogate gradient spike function with an adaptive threshold
    Input: Membrane potentials
    Output: Tuple of present and delayed spikes: (X(t), X(d_0, .. d_i))
    Internal state: threshold, delay buffers
    '''
    def __init__(self, delays, delays_xarea, conf, batch_size, N, dt):
        super(ALIFSpikes, self).__init__()

        amplitude = init.expand_to_neurons(conf, 'th_ampl')
        self.adaptive = torch.any(amplitude>0)
        if self.adaptive:
            tau = init.expand_to_neurons(conf, 'th_tau')
            self.alpha = util.decayconst(tau, dt)
            self.amplitude = amplitude
            self.register_buffer('threshold', torch.zeros(batch_size, N))

        self.t = 0
        self.delays = delays
        self.delays_xarea = delays_xarea
        self.max_delay = max(delays + delays_xarea)
        for d in range(self.max_delay):
            self.register_buffer(
                f'delay_{d}', torch.zeros(batch_size, N))

    def reset(self):
        if self.adaptive:
            self.threshold = torch.zeros_like(self.threshold)
        self.t = 0
        for d in range(self.max_delay):
            setattr(self, f'delay_{d}', torch.zeros_like(self.delay_0))

    def forward(self, V):
        '''
        V: (batch, pre)
        Output:
            X: (batch, pre)
            Xd: (delay, batch, pre)
            Xd_xarea: (delay, batch, pre)
        '''
        if self.adaptive:
            mthr = V - (self.threshold + 1)
            X = SurrGradSpike.apply(mthr)
            self.threshold = self.threshold * self.alpha + X * self.amplitude
        else:
            mthr = V - 1
            X = SurrGradSpike.apply(mthr)

        Xd = self.get_delayed_spikes(self.delays)
        Xd_xarea = self.get_delayed_spikes(self.delays_xarea)

        setattr(self, f'delay_{self.t % self.max_delay}', X)
        self.t = self.t + 1

        return X, Xd, Xd_xarea

    def get_delayed_spikes(self, delays):
        Xd = []
        for d in delays:
            Xd.append(getattr(self, f'delay_{(self.t-d) % self.max_delay}'))
        return torch.stack(Xd, dim=0) if len(Xd) > 0 else None



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
