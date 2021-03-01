import torch
import cantata.elements as ce

class Conductor(torch.nn.Module):
    '''
    Putting it all together.
    Input: Rates
    Output: Output spikes
    Internal state: Cross-area spikes
    '''
    def __init__(self, conf, batch_size, dt, n_steps):
        super(Conductor, self).__init__()

        self.input = ce.PoissonInput(conf.input, dt)
        self.areas = torch.ModuleList()
        for name, area in conf.areas.items():
            m = ce.SNN(area, dt, name)
            areas.append(m)
            self.register_buffer(f'Xd_{m.name}', torch.empty(0))

        self.reset()

    def reset(self):
        self.Xd_prev = []
        for m in self.areas:
            X, Xd = m.reset()
            setattr(self, f'Xd_{m.name}', Xd)

    def forward(self, rates):
        Xd_returned = []
        for m in self.areas:
            Xd_returned.append(getattr(self, f'Xd_{m.name}'))
        outputs = ([],) + ([] for m in self.areas)

        for rate_t in rates:
            Xd_prev, Xd_returned = Xd_returned, []
            Xi = self.input(rate_t)
            outputs[0].append(Xi)
            for i, area in enumerate(self.areas):
                X, Xd = area(Xi, *Xd_prev)
                outputs[i+1].append(X)
                Xd_returned.append(Xd)

        for i, m in enumerate(self.areas):
            setattr(self, f'Xd_{m.name}', Xd_returned[i])

        return (torch.stack(X, dim=0) for X in outputs)
