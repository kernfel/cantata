import pytest
from cantata import config, util, init
import cantata.elements as ce
import torch
import numpy as np

class Mock_STDP(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Mock_STDP, self).__init__()
        self.did_reset = None
        self.active = True
        self.init_args = args, kwargs

    def reset(self, W):
        self.did_reset = W.clone()

    def forward(self, Xd, X, *args):
        # Xd is dbe; X is bo; return is beo.
        d,b,e = Xd.shape
        b,o = X.shape
        self.mock_weights = torch.arange(b*e*o).reshape(b,e,o).to(X) / (b*e*o)
        return self.mock_weights.clone()

@pytest.fixture(params = [True, False], ids = ['xarea', 'internal'])
def constructor(model1, batch_size, dt, request):
    defaults = model1.areas.A1, Mock_STDP, batch_size, dt
    if request.param:
        return *defaults, model1.areas.A2, "A2"
    else:
        return defaults

def test_DeltaSynapse_can_change_device(model1, spikes):
    batch_size, dt = 32, 1e-3
    m = ce.DeltaSynapse(model1.areas.A1, Mock_STDP, batch_size, dt)
    for device in [torch.device('cuda'), torch.device('cpu')]:
        Xd = spikes(3,batch_size,5).to(device)
        X = spikes(batch_size,5).to(device)
        V = torch.rand_like(X)
        m.to(device)
        I = m(Xd, X, V)
        assert I.device == X.device

@pytest.mark.parametrize('STDP',
    [pytest.param(ce.Abbott),
     pytest.param(ce.Clopath)])
def test_DeltaSynapse_constructs_STDP(constructor, STDP):
    m = ce.DeltaSynapse(*constructor)
    L = STDP(*m.longterm.init_args[0], **m.longterm.init_args[1])
    assert L.active

def test_DeltaSynapse_state(constructor, module_tests):
    nPost = 5 if len(constructor) == 4 else 6
    module_tests.check_state(
        ce.DeltaSynapse(*constructor),
        ['W'],
        [(5, nPost)]
    )

def test_DeltaSynapse_deactivates_with_no_connections(model1, batch_size, dt):
    m1 = ce.DeltaSynapse(model1.areas.A2, Mock_STDP, batch_size, dt)
    m2 = ce.DeltaSynapse(model1.areas.A1, Mock_STDP, batch_size, dt,
        model1.areas.A2, 'NotA2')
    assert not m1.active
    assert not m2.active

def test_DeltaSynapse_disables_STDP_when_possible(constructor):
    if len(constructor) == 4:
        constructor[0].populations.Exc.targets.Exc.STDP_frac = 0.
        constructor[0].populations.Exc.targets.Inh.STDP_frac = 0.
    else:
        constructor[0].populations.Exc.targets['A2.deadend'].STDP_frac = 0.
    m = ce.DeltaSynapse(*constructor)
    assert not m.has_STDP

def test_DeltaSynapse_disables_STP_when_possible(constructor):
    constructor[0].populations.Exc.p = 0.
    constructor[0].populations.Inh.p = 0.
    m = ce.DeltaSynapse(*constructor)
    assert not m.has_STP

def test_DeltaSynapse_resets_on_construction(constructor):
    m = ce.DeltaSynapse(*constructor)
    assert m.longterm.did_reset is not None

def test_DeltaSynapse_reset_aligns_signs(constructor):
    m = ce.DeltaSynapse(*constructor)
    W = torch.nn.functional.relu(torch.rand_like(m.W) - 0.5)
    m.W = torch.nn.Parameter(W)
    nonzero = m.W > 0
    m.reset()
    assert torch.all(m.signs[~nonzero] == 0)
    assert torch.all(m.signs[:2][nonzero[:2]] == 1)
    assert torch.all(m.signs[2:][nonzero[2:]] == -1)

def test_DeltaSynapse_aligns_signs_on_load(constructor):
    m1 = ce.DeltaSynapse(*constructor)
    W_orig = m1.W.clone()
    W = torch.nn.functional.relu(torch.rand_like(m1.W) - 0.5)
    m1.W = torch.nn.Parameter(W)
    nonzero = m1.W > 0
    state = m1.state_dict()
    del m1
    m2 = ce.DeltaSynapse(*constructor)
    m2.load_state_dict(state)
    assert torch.all(m2.signs[~nonzero] == 0)
    assert torch.all(m2.signs[:2][nonzero[:2]] == 1)
    assert torch.all(m2.signs[2:][nonzero[2:]] == -1)

def test_DeltaSynapse_reset_propagates_W_to_STDP(constructor):
    m = ce.DeltaSynapse(*constructor)
    expected = torch.rand_like(m.W)
    m.W = torch.nn.Parameter(expected)
    m.reset()
    assert torch.equal(m.longterm.did_reset, expected)

def test_DeltaSynapse_reset_maintains_W(constructor):
    m = ce.DeltaSynapse(*constructor)
    expected = torch.rand_like(m.W)
    m.W = torch.nn.Parameter(expected)
    m.reset()
    assert torch.equal(m.W, expected)

def test_DeltaSynapse_nonplastic_output(model1, batch_size, dt, spikes):
    m = ce.DeltaSynapse(model1.areas.A1, Mock_STDP, batch_size, dt)
    m.has_STP = m.has_STDP = False
    d,b,e,o = 3, batch_size, 5, 5
    Xd, X, V = spikes(d,b,e), spikes(b,o), torch.rand(b,o)
    expected = torch.zeros(b,o)
    for pre, post, delay, sign in zip(
        [range(2), range(2), range(2,5), range(2,5)],
        [range(2), range(2,5), range(2), range(2,5)],
        [1,        1,         2,         0],
        [1,        1,         -1,        -1]
    ):
        for i in pre:
            for j in post:
                for batch in range(b):
                    if Xd[delay, batch, i] > 0:
                        w = m.W[i,j] * sign
                        expected[batch, j] += w
    I = m(Xd, X, V)
    assert torch.allclose(I, expected)

def test_DeltaSynapse_scales_weight_with_STP(model1, batch_size, dt, spikes):
    m = ce.DeltaSynapse(model1.areas.A1, Mock_STDP, batch_size, dt)
    S = m.shortterm.Ws = torch.randn_like(m.shortterm.Ws)
    m.has_STDP = False
    d,b,e,o = 3, batch_size, 5, 5
    Xd, X, V = spikes(d,b,e), spikes(b,o), torch.rand(b,o)
    expected = torch.zeros(b,o)
    for pre, post, delay, sign in zip(
        [range(2), range(2), range(2,5), range(2,5)],
        [range(2), range(2,5), range(2), range(2,5)],
        [1,        1,         2,         0],
        [1,        1,         -1,        -1]
    ):
        for i in pre:
            for j in post:
                for batch in range(b):
                    if Xd[delay, batch, i] > 0:
                        w = m.W[i,j] * sign * (1 + S[delay, batch, i])
                        expected[batch, j] += w
    I = m(Xd, X, V)
    assert torch.allclose(I, expected)

def test_DeltaSynapse_interpolates_weight_with_STDP(model1, batch_size, dt,
                                                    spikes):
    m = ce.DeltaSynapse(model1.areas.A1, Mock_STDP, batch_size, dt)
    m.has_STP = False
    d,b,e,o = 3, batch_size, 5, 5
    Xd, X, V = spikes(d,b,e), spikes(b,o), torch.rand(b,o)
    m(Xd, X, V)
    L = m.longterm.mock_weights
    expected = torch.zeros(b,o)
    for pre, post, delay, sign, frac in zip(
        [range(2), range(2), range(2,5), range(2,5)],
        [range(2), range(2,5), range(2), range(2,5)],
        [1,        1,         2,         0],
        [1,        1,         -1,        -1],
        [0.5,      0.2,       0,         0]
    ):
        for i in pre:
            for j in post:
                for batch in range(b):
                    if Xd[delay, batch, i] > 0:
                        w = sign * ((1-frac) * m.W[i,j] + frac * L[batch, i, j])
                        expected[batch, j] += w
    I = m(Xd, X, V)
    assert torch.allclose(I, expected)
