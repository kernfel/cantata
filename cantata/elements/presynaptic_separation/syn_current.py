import torch
import cantata.elements as ce

class SynCurrent(ce.SynCurrent):
    def __init__(self, conf, batch_size, dt, nPre):
        super(SynCurrent, self).__init__(conf, batch_size, dt)
        if not self.active:
            return
        N = init.get_N(conf)
        self.I = self.I.unsqueeze(1).expand(batch_size, nPre, N)
