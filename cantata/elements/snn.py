from cantata import init
import cantata.elements as ce


class SNN(ce.Module):
    '''
    SNN layer.
    Input: Feedforward spikes
    Output: Output spikes
    Internal state: -
    '''

    def __init__(self, conf, batch_size, dt,
                 membrane, spikes, syn_input, syn_internal):
        super(SNN, self).__init__()
        self.N = init.get_N(conf)
        self.p_names, self.p_idx = init.build_population_indices(conf)
        self.membrane = membrane
        self.spikes = spikes
        self.syn_input = syn_input
        self.syn_internal = syn_internal
        self.reset()

    def reset(self):
        for m in self.children():
            m.reset()

    def forward(self, X_input):
        V = self.membrane.V
        X, Xd = self.spikes(V)
        current = self.syn_internal(Xd, X, V) + self.syn_input(X_input, X, V)
        self.membrane(X, current)
        return X
