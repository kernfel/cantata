import torch
import cantata.elements as ce

class SNN(ce.SNN):
    def __init__(self, *args, **kwargs):
        super(SNN, self).__init__(*args, **kwargs)

    def forward(self, X_input):
        V = self.membrane.V
        X, Xd = self.spikes(V)
        current_internal = self.syn_internal(Xd, X, V)
        current_input = self.syn_input(X_input, X, V)

        # concatenate (b,e,o) over separate e
        current = torch.cat([current_input, current_internal], dim=1)

        self.membrane(X, current)
        return X
