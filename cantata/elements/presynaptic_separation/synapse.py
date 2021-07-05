import torch
import cantata.elements as ce

class Synapse(ce.Synapse):
    def __init__(self, *args, **kwargs):
        super(Synapse, self).__init__(*args, **kwargs)

    def forward(self, Xd, X, Vpost):
        if not self.active:
            shape = Vpost.shape[0], Xd.shape[-1], Vpost.shape[-1]
            return torch.zeros(shape, device=Vpost.device, dtype=Vpost.dtype)
        else:
            return super(Synapse, self).forward(Xd, X, Vpost)

    def internal_forward(self, WD, W, Xd):
        return torch.einsum(
            f'{WD}, dbe, deo          ->beo',
             W,     Xd,  self.delaymap)
