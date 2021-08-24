from cantata import init
import cantata.elements as ce


class SNN(ce.Module):
    '''
    SNN layer.
    Input: Feedforward spikes
    Output: Output spikes
    Internal state: -
    '''

    def __init__(self, batch_size, dt,
                 membrane, spikes, syn_input, syn_internal,
                 conf=None):
        super(SNN, self).__init__()
        if conf is not None:
            self.N = init.get_N(conf)
            self.p_names, self.p_idx = init.build_population_indices(conf)
        self.membrane = membrane
        self.spikes = spikes
        self.syn_input = syn_input
        self.syn_internal = syn_internal
        self.reset()

    def reset(self, keep_values=False):
        for m in self.children():
            m.reset(keep_values=keep_values)

    def forward(self, X_input):
        V = self.membrane.V
        X, Xd = self.spikes(V)
        current = self.syn_internal(Xd, X, V) + self.syn_input(X_input, X, V)
        self.membrane(current, X)
        return X
