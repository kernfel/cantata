import torch
import cantata.elements as ce

class Conductor(torch.nn.Module):
    '''
    Putting it all together.
    Input: Rates
    Output: Output spikes
    Internal state: Cross-area spikes
    '''
    def __init__(self, conf, STDP, batch_size, dt):
        super(Conductor, self).__init__()

        self.input = ce.PoissonInput(conf.input, dt)
        self.areas = torch.nn.ModuleList()
        for name, area in conf.areas.items():
            m = ce.SNN(conf, STDP, batch_size, dt, name)
            self.areas.append(m)
            self.register_buffer(f'Xd_{m.name}', torch.empty(0))

        self.reset()

    def reset(self):
        for m in self.areas:
            X, Xd = m.reset()
            setattr(self, f'Xd_{m.name}', Xd)

    def forward(self, rates):
        Xd_returned = []
        for m in self.areas:
            Xd_returned.append(getattr(self, f'Xd_{m.name}'))
        inputs = []
        outputs = [[] for m in self.areas]

        for rate_t in rates:
            Xd_prev, Xd_returned = Xd_returned, []
            Xi = self.input(rate_t)
            inputs.append(Xi)
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
