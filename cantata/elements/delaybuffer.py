import torch
import cantata.elements as ce


class DelayBuffer(ce.Module):
    def __init__(self, shape, *delay_lists):
        super(DelayBuffer, self).__init__()
        self.delay_lists = delay_lists
        alldelays = [i for part in delay_lists for i in part]
        for i in alldelays:
            if i < 0:
                raise ValueError(f'Invalid delay: {i}')
        self.active = len(alldelays) > 0
        if self.active:
            self.t = 0
            self.max_delay = max(alldelays)
            for d in range(self.max_delay):
                self.register_buffer(
                    f'delay_{d}', torch.zeros(shape))

    def reset(self, keep_values=False):
        if self.active:
            if not keep_values:
                self.t = 0
            for d in range(self.max_delay):
                if keep_values:
                    setattr(self, f'delay_{d}',
                            getattr(self, f'delay_{d}').detach())
                else:
                    setattr(self, f'delay_{d}',
                            torch.zeros_like(self.delay_0))

    def forward(self, input):
        out = [self.get_buffer(d, input) for d in self.delay_lists]
        if self.active and self.max_delay > 0:
            setattr(self, f'delay_{self.t % self.max_delay}', input.clone())
            self.t = self.t + 1
        return out

    def get_buffer(self, delays, input):
        if not self.active or len(delays) == 0:
            return None
        out = []
        for d in delays:
            if d == 0:
                out.append(input.clone())
            else:
                out.append(
                    getattr(self, f'delay_{(self.t-d) % self.max_delay}'))
        return torch.stack(out, dim=0)
