import torch
from cantata import init
import cantata.elements as ce


class SynCurrent(ce.SynCurrent):
    def __init__(self, N, batch_size, alpha, nPre, **kwargs):
        super().__init__(N, batch_size, alpha)
        if not self.active:
            return
        self.I = self.I.unsqueeze(1).expand(batch_size, nPre, N)

    @classmethod
    def configured(cls, *args, **kwargs):
        return super().configured(*args, **kwargs)
