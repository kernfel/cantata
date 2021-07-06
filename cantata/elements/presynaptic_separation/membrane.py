import torch
import cantata.elements as ce

class Membrane(ce.Membrane):
    def __init__(self, conf, batch_size, dt, **kwargs):
        N = init.get_N(conf)
        self.nPre = kwargs.get('nTotal')
        self.register_buffer('V_separate', torch.zeros(batch_size, nPre, N))
        super(Membrane, self).__init__(conf, batch_size, dt, **kwargs)

    def reset(self):
        super(Membrane, self).reset()
        # Divide the random reset evenly:
        self.V_separate = self.V.unsqueeze(1).expand_as(self.V_separate) / self.nPre

    def forward(self, X, current):
        self.V_separate = self.V_separate * self.alpha.unsqueeze(1) + current
        super(Membrane, self).forward(X, current.sum(dim=1))
        with torch.no_grad():
            self.V_separate[(self.V == 0).unsqueeze(1)] = 0 # Can't use self.ref!
        return self.V
