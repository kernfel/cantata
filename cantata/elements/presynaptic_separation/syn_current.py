import torch
from cantata import init
import cantata.elements as ce

class SynCurrent(ce.SynCurrent):
    def __init__(self, conf, batch_size, dt, **kwargs):
        super(SynCurrent, self).__init__(conf, batch_size, dt, **kwargs)
        if not self.active:
            return
        N = init.get_N(conf)
        nPre = kwargs.get('nPre')
        self.I = self.I.unsqueeze(1).expand(batch_size, nPre, N)
