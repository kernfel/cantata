import torch
import cantata.elements as ce


class Synapse(ce.Synapse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def configured(cls, *args, **kwargs):
        return super().configured(*args, **kwargs)

    def forward(self, Xd, X=None, Vpost=None):
        if not self.active:
            shape = Vpost.shape[0], Xd.shape[-1], Vpost.shape[-1]
            return torch.zeros(shape, device=Vpost.device, dtype=Vpost.dtype)
        else:
            return super().forward(Xd, X, Vpost)

    def internal_forward(self, WD, W, Xd):
        return torch.einsum(
            f'{WD}, dbe, deo          ->beo',
            W,      Xd,  self.delaymap)
