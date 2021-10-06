import torch
from cantata import init, util
import cantata.elements as ce


class Membrane(ce.Membrane):
    def __init__(self, N, batch_size, alpha,
                 tau_ref=None, noise=None, nTotal=0, **kwargs):
        self.nPre = nTotal
        # ce.Membrane.__init__ calls reset(), so V_separate needs to be present
        # but before torch.nn.Module.__init__(), there is no register_buffer.
        self.V_separate = torch.zeros(batch_size, self.nPre, N)

        super().__init__(N, batch_size, alpha, tau_ref, noise)

        Vsep = self.V_separate
        del self.V_separate
        self.register_buffer('V_separate', Vsep)

    @classmethod
    def configured(cls, *args, **kwargs):
        return super().configured(*args, **kwargs)

    def reset(self, keep_values=False):
        super().reset(keep_values)
        # Divide the random reset evenly:
        self.V_separate = self.V.unsqueeze(
            1).expand_as(self.V_separate) / self.nPre

    def forward(self, current, X=None):
        self.V_separate = self.V_separate * self.alpha + current

        super().forward(current.sum(dim=1), X)

        with torch.no_grad():
            self.V_separate[  # Can't use self.ref, it's half a step ahead!
                (self.V == 0).unsqueeze(1).expand_as(self.V_separate)
            ] = 0
        return self.V
