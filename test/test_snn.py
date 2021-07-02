from cantata import config, util, init
import cantata.elements as ce
import torch

class Mock_Membrane(torch.nn.Module):
    def __init__(self, *shape):
        super(Mock_Membrane, self).__init__()
        self.register_buffer('V', torch.rand(shape))
    def reset(self):
        pass
    def forward(self, X, current):
        return self.V

class Mock_Spikes(torch.nn.Module):
    def __init__(self, delay):
        super(Mock_Spikes, self).__init__()
        self.delay = delay
        self.register_buffer('X', torch.zeros(1))
        self.register_buffer('Xd', torch.zeros(1))
    def reset(self):
        pass
    def forward(self, V):
        self.X = torch.rand_like(V)
        self.Xd = torch.rand(self.delay, *V.shape)
        return self.X, self.Xd

class Mock_Synapse(torch.nn.Module):
    def __init__(self):
        super(Mock_Synapse, self).__init__()
        self.register_buffer('I', torch.zeros(1))
    def reset(self):
        pass
    def forward(self, Xd, X, Vpost):
        I = torch.rand_like(X)
        return self.I

def test_SNN_does_not_modify_children(module_tests, model1, spikes):
    d,batch_size,e = 3,32,5
    dt = .001
    membrane = Mock_Membrane(batch_size, 5)
    lif = Mock_Spikes(d)
    syn_input = Mock_Synapse()
    syn_internal = Mock_Synapse()
    m = ce.SNN(model1.areas.A1, batch_size, dt,
        membrane, lif, syn_input, syn_internal)
    X = spikes(d,batch_size,e)
    module_tests.check_no_child_modification(m, X)
