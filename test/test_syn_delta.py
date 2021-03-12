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

@pytest.fixture(
    params = [True, False], ids = ['shared_weights', 'unique_weights'])
def shared_weights(request):
    return {'shared_weights': request.param}

@pytest.fixture(params = [True, False], ids = ['xarea', 'internal'])
def constructor(model1, batch_size, dt, request):
    defaults = model1.areas.A1, batch_size, dt
    if request.param:
        return *defaults, model1.areas.A2, "A2"
    else:
        return defaults

def test_DeltaSynapse_can_change_device(model1, spikes):
    batch_size, dt = 32, 1e-3
    m = ce.DeltaSynapse(model1.areas.A1, batch_size, dt, STDP=Mock_STDP)
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
def test_DeltaSynapse_constructs_STDP(constructor, STDP, shared_weights):
    m = ce.DeltaSynapse(*constructor, STDP=STDP, **shared_weights)
    assert m.longterm.active

def test_DeltaSynapse_state_with_shared_weights(constructor, module_tests):
    nPost = 5 if len(constructor) == 3 else 10
    module_tests.check_state(
        ce.DeltaSynapse(*constructor, STDP=Mock_STDP),
        ['W'],
        [(5, nPost)]
    )

def test_DeltaSynapse_state_with_unique_weights(constructor, module_tests):
    nPost = 5 if len(constructor) == 3 else 10
    batch_size = constructor[1]
    module_tests.check_state(
        ce.DeltaSynapse(*constructor, STDP=Mock_STDP, shared_weights = False),
        ['W'],
        [(batch_size, 5, nPost)]
    )

def test_DeltaSynapse_deactivates_with_no_connections(
    model1, batch_size, dt, shared_weights):
    m1 = ce.DeltaSynapse(model1.areas.A2, batch_size, dt,
        STDP=Mock_STDP, **shared_weights)
    m2 = ce.DeltaSynapse(model1.areas.A1, batch_size, dt,
        model1.areas.A2, 'NotA2',
        STDP=Mock_STDP, **shared_weights)
    assert not m1.active
    assert not m2.active

def test_DeltaSynapse_disables_STDP_when_possible(constructor):
    if len(constructor) == 3:
        constructor[0].populations.Exc.targets.Exc.STDP_frac = 0.
        constructor[0].populations.Exc.targets.Inh.STDP_frac = 0.
    else:
        constructor[0].populations.Exc.targets['A2.deadend'].STDP_frac = 0.
    m = ce.DeltaSynapse(*constructor, STDP=Mock_STDP)
    assert not m.has_STDP

def test_DeltaSynapse_disables_STP_when_possible(constructor):
    constructor[0].populations.Exc.p = 0.
    constructor[0].populations.Inh.p = 0.
    m = ce.DeltaSynapse(*constructor, STDP=Mock_STDP)
    assert not m.has_STP

def test_DeltaSynapse_resets_on_construction(constructor):
    m = ce.DeltaSynapse(*constructor, STDP=Mock_STDP)
    assert m.longterm.did_reset is not None

def test_DeltaSynapse_aligns_signs_with_pre_and_W(constructor, shared_weights):
    m = ce.DeltaSynapse(*constructor, STDP=Mock_STDP, **shared_weights)
    W = torch.nn.functional.relu(torch.rand_like(m.W) - 0.5)
    m.W = torch.nn.Parameter(W)
    nonzero = m.W > 0
    m.align_signs()
    assert torch.all(m.signs[~nonzero] == 0)
    if shared_weights['shared_weights']:
        exc = range(2)
        inh = range(2,5)
    else:
        exc = np.ix_(range(constructor[1]), range(2))
        inh = np.ix_(range(constructor[1]), range(2,5))
    print(nonzero.shape, m.signs.shape)
    assert torch.all(m.signs[exc][nonzero[exc]] == 1)
    assert torch.all(m.signs[inh][nonzero[inh]] == -1)

def test_DeltaSynapse_reset_aligns_signs(constructor, shared_weights):
    m = ce.DeltaSynapse(*constructor, STDP=Mock_STDP, **shared_weights)
    W = torch.nn.functional.relu(torch.rand_like(m.W) - 0.5)
    m.W = torch.nn.Parameter(W)
    nonzero = m.W > 0
    m.reset()
    signs_after_reset = m.signs.clone()
    m.align_signs()
    assert torch.equal(m.signs, signs_after_reset)

def test_DeltaSynapse_aligns_signs_on_load(constructor, shared_weights):
    m1 = ce.DeltaSynapse(*constructor, STDP=Mock_STDP, **shared_weights)
    W = torch.nn.functional.relu(torch.rand_like(m1.W) - 0.5)
    m1.W = torch.nn.Parameter(W)
    state = m1.state_dict()
    m2 = ce.DeltaSynapse(*constructor, STDP=Mock_STDP, **shared_weights)
    m2.load_state_dict(state)
    m1.align_signs()
    assert torch.equal(m2.signs, m1.signs)

def test_DeltaSynapse_reset_propagates_W_to_STDP(constructor, shared_weights):
    m = ce.DeltaSynapse(*constructor, STDP=Mock_STDP, **shared_weights)
    expected = torch.rand_like(m.W)
    m.W = torch.nn.Parameter(expected)
    m.reset()
    assert torch.equal(m.longterm.did_reset, expected)

def test_DeltaSynapse_reset_maintains_W(constructor, shared_weights):
    m = ce.DeltaSynapse(*constructor, STDP=Mock_STDP, **shared_weights)
    expected = torch.rand_like(m.W)
    m.W = torch.nn.Parameter(expected)
    m.reset()
    assert torch.equal(m.W, expected)

def test_DeltaSynapse_nonplastic_output(
    model1, batch_size, dt, spikes, shared_weights
):
    m = ce.DeltaSynapse(
        model1.areas.A1, batch_size, dt, STDP=Mock_STDP, **shared_weights)
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
                        if shared_weights['shared_weights']:
                            w = m.W[i,j]
                        else:
                            w = m.W[batch,i,j]
                        expected[batch, j] += w * sign
    I = m(Xd, X, V)
    assert torch.allclose(I, expected)

def test_DeltaSynapse_scales_weight_with_STP(
    model1, batch_size, dt, spikes, shared_weights
):
    m = ce.DeltaSynapse(
        model1.areas.A1, batch_size, dt, STDP=Mock_STDP, **shared_weights)
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
                        if shared_weights['shared_weights']:
                            w = m.W[i,j]
                        else:
                            w = m.W[batch,i,j]
                        w = w * sign * (1 + S[delay, batch, i])
                        expected[batch, j] += w
    I = m(Xd, X, V)
    assert torch.allclose(I, expected)

def test_DeltaSynapse_interpolates_weight_with_STDP(model1, batch_size, dt,
                                                    spikes, shared_weights):
    m = ce.DeltaSynapse(
        model1.areas.A1, batch_size, dt, STDP=Mock_STDP, **shared_weights)
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
                        if shared_weights['shared_weights']:
                            w = m.W[i,j]
                        else:
                            w = m.W[batch,i,j]
                        w = sign * ((1-frac) * w + frac * L[batch, i, j])
                        expected[batch, j] += w
    I = m(Xd, X, V)
    assert torch.allclose(I, expected)
