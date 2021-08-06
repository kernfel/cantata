import pytest
from cantata import init
import cantata.elements as ce
import torch
import numpy as np


class Mock_STDP(torch.nn.Module):
    def __init__(self):
        super(Mock_STDP, self).__init__()
        self.did_reset = None
        self.active = True
        self.register_buffer('mock_weights', None)

    def reset(self, host):
        self.did_reset = id(host)

    def forward(self, Xd, X, *args):
        d, b, e = Xd.shape
        b, o = X.shape
        self.mock_weights = torch.arange(
            b*e*o).reshape(b, e, o).to(X) / (b*e*o)
        return self.mock_weights


@pytest.fixture(params=[True, False], ids=['STDP', 'No_STDP'])
def STDP(request):
    return Mock_STDP() if request.param else None


class Mock_STP(torch.nn.Module):
    def __init__(self):
        super(Mock_STP, self).__init__()
        self.did_reset = False
        self.active = True
        self.register_buffer('mock_weights', None)

    def reset(self):
        self.did_reset = True

    def forward(self, Xd):
        d, b, e = Xd.shape
        self.mock_weights = torch.arange(
            d*b*e).reshape(d, b, e).to(Xd) / (d*b*e)
        return self.mock_weights


@pytest.fixture(params=[True, False], ids=['STP', 'No_STP'])
def STP(request):
    return Mock_STP() if request.param else None


class Mock_Current(torch.nn.Module):
    def __init__(self):
        super(Mock_Current, self).__init__()
        self.did_reset = False
        self.active = True
        self.register_buffer('output', None)

    def reset(self):
        self.did_reset = True

    def forward(self, impulse):
        b, o = impulse.shape
        self.mock_current = torch.arange(b*o).reshape(b, o).to(impulse) / (b*o)
        self.output = self.mock_current * impulse
        return self.output


@pytest.fixture(params=[True, False], ids=['Current', 'No_Current'])
def Current(request):
    return Mock_Current() if request.param else None


@pytest.fixture(
    params=[True, False], ids=['shared_weights', 'unique_weights'])
def shared_weights(request):
    return {'shared_weights': request.param}


@pytest.fixture(params=[True, False], ids=['xarea', 'internal'])
def constructor(model1, batch_size, dt, request):
    conf_pre = model1.areas.A1
    conf_post = model1.areas.A2 if request.param else None
    name_post = 'A2' if request.param else None
    projections = init.build_projections(conf_pre, conf_post, name_post)
    return (projections, conf_pre, conf_post, batch_size, dt)


@pytest.mark.parametrize('constructor', [False], indirect=True)
def test_Synapse_does_not_modify_children(module_tests, constructor, spikes,
                                          batch_size):
    stp = Mock_STP()
    ltp = Mock_STDP()
    current = Mock_Current()
    m = ce.Synapse.configured(*constructor, stp=stp, ltp=ltp, current=current)
    d, b, e, o = 3, batch_size, 5, 5
    Xd, X, V = spikes(d, b, e), spikes(b, o), torch.rand(b, o)
    module_tests.check_no_child_modification(m, Xd, X, V)


def test_Synapse_can_change_device(constructor, spikes, batch_size, dt):
    m = ce.Synapse.configured(*constructor)
    d = m.delaymap.shape[0]
    for device in [torch.device('cuda'), torch.device('cpu')]:
        Xd = spikes(d, batch_size, 5).to(device)
        X = spikes(batch_size, 5).to(device)
        V = torch.rand_like(X)
        m.to(device)
        out = m(Xd, X, V)
        assert out.device == X.device


def test_Synapse_state_with_shared_weights(constructor, module_tests):
    nPost = 5 if constructor[2] is None else 10
    module_tests.check_state(
        ce.Synapse.configured(*constructor, shared_weights=True),
        ['W'],
        [(5, nPost)]
    )


def test_Synapse_state_with_unique_weights(constructor, module_tests):
    nPost = 5 if constructor[2] is None else 10
    batch_size = constructor[3]
    module_tests.check_state(
        ce.Synapse.configured(*constructor, shared_weights=False),
        ['W'],
        [(batch_size, 5, nPost)]
    )


def test_Synapse_weight_limits(model2, batch_size, dt):
    a, b = np.random.rand(2)
    mini, maxi = (a, b) if a < b else (b, a)
    model2.areas.A1.populations.Exc1.targets.Exc1.wmin = mini
    model2.areas.A1.populations.Exc1.targets.Exc1.wmax = maxi
    projections = init.build_projections(model2.areas.A1)
    m = ce.Synapse.configured(
        projections, model2.areas.A1, None, batch_size, dt)
    w = m.W[:150, :150]
    assert torch.all((w > mini) + (w == 0))
    assert torch.all(w < maxi)


def test_Synapse_deactivates_with_no_connections(
    model1, batch_size, dt, shared_weights
):
    projections1 = init.build_projections(model1.areas.A2)
    m1 = ce.Synapse.configured(projections1, model1.areas.A2, None,
                               batch_size, dt, **shared_weights)
    projections2 = init.build_projections(model1.areas.A1, model1.areas.A2,
                                          'NotA2')
    m2 = ce.Synapse.configured(projections1, model1.areas.A1, model1.areas.A2,
                               batch_size, dt, **shared_weights)
    assert not m1.active
    assert not m2.active


def test_Synapse_resets_on_construction(constructor, STP, Current):
    ce.Synapse.configured(*constructor, stp=STP, current=Current)
    assert STP is None or STP.did_reset
    assert Current is None or Current.did_reset


def test_Synapse_reset_passes_self_to_STDP(constructor, STDP):
    m = ce.Synapse.configured(*constructor, ltp=STDP)
    assert STDP is None or STDP.did_reset == id(m)


def test_Synapse_aligns_signs_with_pre_and_W(constructor, shared_weights):
    m = ce.Synapse.configured(*constructor, **shared_weights)
    W = torch.nn.functional.relu(torch.rand_like(m.W) - 0.5)
    m.W = torch.nn.Parameter(W)
    nonzero = W > 0
    m.align_signs()
    if shared_weights['shared_weights']:
        exc = range(2)
        inh = range(2, 5)
    else:
        batch_size = constructor[3]
        exc = np.ix_(range(batch_size), range(2))
        inh = np.ix_(range(batch_size), range(2, 5))
    assert torch.all(m.signs[~nonzero] == 0)
    assert torch.all(m.signs[exc][nonzero[exc]] == 1)
    assert torch.all(m.signs[inh][nonzero[inh]] == -1)


def test_Synapse_reset_aligns_signs(constructor, shared_weights):
    m = ce.Synapse.configured(*constructor, **shared_weights)
    W = torch.nn.functional.relu(torch.rand_like(m.W) - 0.5)
    m.W = torch.nn.Parameter(W)
    m.reset()
    signs_after_reset = m.signs.clone()
    m.align_signs()
    assert torch.equal(m.signs, signs_after_reset)


def test_Synapse_aligns_signs_on_load(constructor, shared_weights):
    m1 = ce.Synapse.configured(*constructor, **shared_weights)
    W = torch.nn.functional.relu(torch.rand_like(m1.W) - 0.5)
    m1.W = torch.nn.Parameter(W)
    state = m1.state_dict()
    m2 = ce.Synapse.configured(*constructor, **shared_weights)
    m2.load_state_dict(state)
    m1.align_signs()
    assert m1.W is not m2.W
    assert torch.equal(m2.signs, m1.signs)


def test_Synapse_reset_maintains_W(constructor, shared_weights):
    m = ce.Synapse.configured(*constructor, **shared_weights)
    expected = torch.rand_like(m.W)
    m.W = torch.nn.Parameter(expected.clone())
    m.reset()
    assert torch.equal(m.W, expected)


@pytest.mark.parametrize('constructor', [False], indirect=True)
def test_Synapse_output_nosubmodules(constructor, shared_weights, spikes):
    *_, batch_size, dt = constructor
    m = ce.Synapse.configured(*constructor, **shared_weights)
    d, b, e, o = 3, batch_size, 5, 5  # model1, A1
    Xd, X, V = spikes(d, b, e), spikes(b, o), torch.rand(b, o)
    expected = torch.zeros(b, o)
    for pre, post, delay, sign in zip(
        # EE       EI        IE          II
        [range(2), range(2), range(2, 5), range(2, 5)],
        [range(2), range(2, 5), range(2), range(2, 5)],
        [1,        1,         2,         0],
        [1,        1,         -1,        -1]
    ):
        for i in pre:
            for j in post:
                for batch in range(b):
                    if Xd[delay, batch, i] > 0:
                        if shared_weights['shared_weights']:
                            w = m.W[i, j]
                        else:
                            w = m.W[batch, i, j]
                        expected[batch, j] += w * sign
    out = m(Xd, X, V)
    assert torch.allclose(out, expected, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('constructor', [False], indirect=True)
def test_Synapse_scales_weight_with_STP(constructor, shared_weights, spikes):
    projections, conf_pre, conf_post, batch_size, dt = constructor
    stp = Mock_STP()
    m = ce.Synapse.configured(*constructor, stp=stp, **shared_weights)
    m.W[m.W < .1]
    d, b, e, o = 3, batch_size, 5, 5
    Xd, X, V = spikes(d, b, e), spikes(b, o), torch.rand(b, o)
    m(Xd, X, V)
    S = stp.mock_weights.clone()
    expected = torch.zeros(b, o)
    for pre, post, delay, sign in zip(
        [range(2), range(2), range(2, 5), range(2, 5)],
        [range(2), range(2, 5), range(2), range(2, 5)],
        [1,        1,         2,         0],
        [1,        1,         -1,        -1]
    ):
        for i in pre:
            for j in post:
                for batch in range(b):
                    if Xd[delay, batch, i] > 0:
                        if shared_weights['shared_weights']:
                            w = m.W[i, j]
                        else:
                            w = m.W[batch, i, j]
                        w = w * sign * (1 + S[delay, batch, i])
                        expected[batch, j] += w
    out = m(Xd, X, V)
    assert torch.allclose(out, expected, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('constructor', [False], indirect=True)
def test_Synapse_interpolates_weight_with_STDP(constructor, shared_weights,
                                               spikes):
    projections, conf_pre, conf_post, batch_size, dt = constructor
    ltp = Mock_STDP()
    m = ce.Synapse.configured(*constructor, ltp=ltp, **shared_weights)
    d, b, e, o = 3, batch_size, 5, 5
    Xd, X, V = spikes(d, b, e), spikes(b, o), torch.rand(b, o)
    m(Xd, X, V)
    L = ltp.mock_weights.clone()
    expected = torch.zeros(b, o)
    for pre, post, delay, sign, frac in zip(
        [range(2), range(2), range(2, 5), range(2, 5)],
        [range(2), range(2, 5), range(2), range(2, 5)],
        [1,        1,         2,         0],
        [1,        1,         -1,        -1],
        [0.5,      0.2,       0,         0]
    ):
        for i in pre:
            for j in post:
                for batch in range(b):
                    if Xd[delay, batch, i] > 0:
                        if shared_weights['shared_weights']:
                            w = m.W[i, j]
                        else:
                            w = m.W[batch, i, j]
                        w = sign * ((1-frac) * w + frac * L[batch, i, j])
                        expected[batch, j] += w
    out = m(Xd, X, V)
    assert torch.allclose(out, expected, rtol=1e-03, atol=1e-05)


@pytest.mark.parametrize('constructor', [False], indirect=True)
def test_Synapse_filters_through_current(constructor, shared_weights, spikes):
    projections, conf_pre, conf_post, batch_size, dt = constructor
    current = Mock_Current()
    m = ce.Synapse.configured(*constructor, current=current, **shared_weights)
    d, b, e, o = 3, batch_size, 5, 5
    Xd, X, V = spikes(d, b, e), spikes(b, o), torch.rand(b, o)
    m(Xd, X, V)
    C = current.mock_current.clone()
    expected = torch.zeros(b, o)
    for pre, post, delay, sign in zip(
        [range(2), range(2), range(2, 5), range(2, 5)],
        [range(2), range(2, 5), range(2), range(2, 5)],
        [1,        1,         2,         0],
        [1,        1,         -1,        -1]
    ):
        for i in pre:
            for j in post:
                for batch in range(b):
                    if Xd[delay, batch, i] > 0:
                        if shared_weights['shared_weights']:
                            w = m.W[i, j]
                        else:
                            w = m.W[batch, i, j]
                        w = w * sign * C[batch, j]
                        expected[batch, j] += w
    out = m(Xd, X, V)
    assert torch.allclose(out, expected, rtol=1e-03, atol=1e-05)
