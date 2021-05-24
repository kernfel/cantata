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
    def __init__(self, conf, batch_size, dt, out_dtype = torch.float, **kwargs):
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

        self.out_dtype = out_dtype

    def reset(self):
        self.shapes = [0]*len(self.areas)
        for i,m in enumerate(self.areas):
            X, Xd = m.reset()
            setattr(self, f'Xd_{m.name}', Xd)
            self.shapes[i] = X.shape

    def forward(self, rates):
        '''
        rates: (t, batch, channels) in Hz
        Output spikes, by area (including input), as (t, batch, N)
        '''
        Xd_returned = []
        for m in self.areas:
            Xd_returned.append(getattr(self, f'Xd_{m.name}'))

        inputs = None # Shape not known a priori, init at t==0 below
        outputs = [torch.zeros(len(rates), *shape, dtype = self.out_dtype)
                   for shape in self.shapes]

        # Main loop
        for t, rate_t in enumerate(rates):
            Xd_prev, Xd_returned = Xd_returned, []
            Xi = self.input(rate_t)
            if t == 0:
                inputs = torch.zeros(
                    len(rates), *Xi[0].shape, dtype = self.out_dtype)
            inputs[t] = Xi[0]
            for i, area in enumerate(self.areas):
                X, Xd = area(Xi, *Xd_prev)
                outputs[i][t] = X
                Xd_returned.append(Xd)

        # Resumability
        for i, m in enumerate(self.areas):
            setattr(self, f'Xd_{m.name}', Xd_returned[i])

        return (inputs, *outputs)
