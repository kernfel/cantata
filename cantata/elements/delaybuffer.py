import torch

class DelayBuffer(torch.nn.Module):
    def __init__(self, shape, *delay_lists):
        super(DelayBuffer, self).__init__()
        self.t = 0
        self.max_delay = max([i for part in delay_lists for i in part])
        self.delay_lists = delay_lists
        for d in range(self.max_delay):
            self.register_buffer(
                f'delay_{d}', torch.zeros(shape))

    def reset(self):
        self.t = 0
        for d in range(self.max_delay):
            setattr(self, f'delay_{d}', torch.zeros_like(self.delay_0))

    def forward(self, input):
        out = [self.get_buffer(d) for d in self.delay_lists]
        setattr(self, f'delay_{self.t % self.max_delay}', input)
        self.t = self.t + 1
        return out

    def get_buffer(self, delays):
        out = []
        for d in delays:
            out.append(getattr(self, f'delay_{(self.t-d) % self.max_delay}'))
        return torch.stack(out, dim=0) if len(out) > 0 else None
