import torch
import cantata.elements as ce

class Membrane(ce.Membrane):
    def __init__(self, conf, batch_size, dt, nPre):
        super(Membrane, self).__init__(conf, batch_size, dt)
        N = init.get_N(conf)
        self.nPre = nPre
        self.register_buffer('V_separate', torch.zeros(batch_size, nPre, N))

    def reset(self):
        super(Membrane, self).reset()
        # Divide the random reset evenly:
        self.V_separate = self.V.unsqueeze(1).expand_as(self.V_separate) / self.nPre

    def forward(self, X, current):
        self.V_separate = self.V_separate * self.alpha.unsqueeze(1) + current
        return super(Membrane, self).forward(X, current.sum(dim=1))
