import pytest
from cantata import util
import cantata.elements as ce
import torch
import numpy as np


@pytest.fixture(params=[1, 3])
def constructor(model1, request, batch_size, dt):
    return model1.areas.A1, request.param, batch_size, dt


def test_STP_does_not_modify_children(module_tests, constructor, spikes):
    m = ce.STP(*constructor)
    shape = constructor[1], constructor[2], 5
    X = spikes(*shape)
    module_tests.check_no_child_modification(m, X)


def test_STP_can_change_device(constructor, spikes):
    m = ce.STP(*constructor)
    shape = constructor[1], constructor[2], 5
    for device in [torch.device('cuda'), torch.device('cpu')]:
        X = spikes(*shape).to(device)
        m.to(device)
        W = m(X)
        assert W.device == X.device


def test_STP_state(constructor, module_tests):
    shape = constructor[1], constructor[2], 5
    module_tests.check_state(
        ce.STP(*constructor),
        ['Ws'],
        [shape]
    )


def test_STP_reset_clears_weights(constructor, module_tests):
    module_tests.check_reset_clears(
        ce.STP(*constructor),
        'Ws'
    )


def test_STP_deactivates_when_not_required(constructor):
    conf = constructor[0]
    conf.populations.Exc.p = conf.populations.Inh.p = 0.
    m = ce.STP(*constructor)
    assert not m.active


def test_STP_returns_premodification_W(constructor, spikes):
    shape = constructor[1], constructor[2], 5
    m = ce.STP(*constructor)
    expected = torch.rand_like(m.Ws)
    m.Ws = expected.clone()
    W = m(spikes(*shape))
    assert torch.equal(W, expected)
    assert not torch.equal(m.Ws, expected)


def test_STP_decays_weights(constructor):
    shape = constructor[1], constructor[2], 5
    conf, dt = constructor[0], constructor[3]
    conf.populations.Exc.tau_r = tau_e = np.random.rand()
    conf.populations.Inh.tau_r = tau_i = np.random.rand()
    alpha_e = util.decayconst(tau_e, dt)
    alpha_i = util.decayconst(tau_i, dt)
    m = ce.STP(*constructor)
    m.Ws = torch.rand_like(m.Ws)
    expected = m.Ws.clone()
    expected[:, :, :2] *= alpha_e
    expected[:, :, 2:] *= alpha_i
    m(torch.zeros(shape))
    assert torch.allclose(m.Ws, expected)


def test_STP_adds_p_on_spikes(constructor, spikes):
    shape = constructor[1], constructor[2], 5
    m = ce.STP(*constructor)
    expected = m.Ws.clone()
    X = spikes(*shape)
    expected = X.clone()
    expected[:, :, :2] *= 0.1
    expected[:, :, 2:] *= -0.2
    m(X)
    assert torch.equal(m.Ws, expected)
