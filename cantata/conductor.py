import torch
import cantata.elements as ce
from box import Box

class Conductor(torch.nn.Module):
    '''
    Putting it all together.
    Input: Rates
    Output: Output spikes
    Internal state: Cross-area spikes
    '''
    def __init__(self, conf, batch_size, dt, **kwargs):
        super(Conductor, self).__init__()

        self.input = ce.PoissonInput(conf.input, batch_size, dt)
        self.areas = torch.nn.ModuleList()
        all_areas = Box({'__input__': conf.input}) + conf.areas
        for name, area in conf.areas.items():
            m = ce.SNN(area, batch_size, dt, **kwargs,
                name=name, input_areas=all_areas)
            self.areas.append(m)
            self.register_buffer(f'Xd_{m.name}', torch.empty(0))

        self.reset()

    def reset(self):
        for m in self.areas:
            X, Xd = m.reset()
            setattr(self, f'Xd_{m.name}', Xd)

    def forward(self, rates):
        '''
        rates: (t, batch, channels) in Hz
        Output spikes, by area (including input), as (t, batch, N)
        '''
        Xd_returned = []
        for m in self.areas:
            Xd_returned.append(getattr(self, f'Xd_{m.name}'))
        inputs = []
        outputs = [[] for m in self.areas]

        for rate_t in rates:
            Xd_prev, Xd_returned = Xd_returned, []
            Xi = self.input(rate_t)
            inputs.append(Xi[0])
            for i, area in enumerate(self.areas):
                X, Xd = area(Xi, *Xd_prev)
                outputs[i].append(X)
                Xd_returned.append(Xd)

        for i, m in enumerate(self.areas):
            setattr(self, f'Xd_{m.name}', Xd_returned[i])

        return (
            torch.stack(inputs),
            *(None if X is None else torch.stack(X) for X in outputs)
        )
