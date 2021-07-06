import torch
from cantata import init
import cantata.elements as ce

class Membrane(ce.Membrane):
    def __init__(self, conf, batch_size, dt, **kwargs):
        N = init.get_N(conf)
        self.nPre = kwargs.get('nTotal')
        self.V_separate = torch.zeros(batch_size, self.nPre, N)
        super(Membrane, self).__init__(conf, batch_size, dt, **kwargs)
        Vsep = self.V_separate
        del self.V_separate
        self.register_buffer('V_separate', Vsep)

    def reset(self):
        super(Membrane, self).reset()
        # Divide the random reset evenly:
        self.V_separate = self.V.unsqueeze(1).expand_as(self.V_separate) / self.nPre

    def forward(self, X, current):
        self.V_separate = self.V_separate * self.alpha + current
        super(Membrane, self).forward(X, current.sum(dim=1))
        with torch.no_grad():
            self.V_separate[ # Can't use self.ref, it's half a step ahead!
                (self.V == 0).unsqueeze(1).expand_as(self.V_separate)
            ] = 0
        return self.V
