import pytest
from cantata import util
import cantata.elements as ce
import torch
import numpy as np


@pytest.fixture()
def constructor(model1, batch_size, dt):
    return model1.areas.A1, batch_size, dt


def test_SynCurrent_does_not_modify_children(module_tests, constructor, spikes):
    m = ce.SynCurrent.configured(*constructor)
    b, o = constructor[1], 5
    current = torch.rand(b, o)
    module_tests.check_no_child_modification(m, current)


def test_SynCurrent_can_change_device(constructor, spikes):
    m = ce.SynCurrent.configured(*constructor)
    b, o = constructor[1], 5
    for device in [torch.device('cuda'), torch.device('cpu')]:
        current = torch.rand(b, o, device=device)
        m.to(device)
        out = m(current)
        assert out.device == current.device


def test_SynCurrent_state(constructor, module_tests):
    b, o = constructor[1], 5
    module_tests.check_state(
        ce.SynCurrent.configured(*constructor),
        ['I'],
        [(b, o)]
    )


def test_SynCurrent_reset_clears_I(constructor, module_tests):
    module_tests.check_reset_clears(
        ce.SynCurrent.configured(*constructor),
        'I'
    )


def test_SynCurrent_deactivates_when_not_required(constructor):
    conf = constructor[0]
    conf.tau_I = 0.
    m = ce.SynCurrent.configured(*constructor)
    assert not m.active


def test_SynCurrent_decays_I(constructor):
    conf, batch_size, dt = constructor
    conf.tau_I = np.random.rand()
    alpha = util.decayconst(conf.tau_I, dt)
    m = ce.SynCurrent.configured(*constructor)
    m.I = torch.rand_like(m.I)
    expected = m.I * alpha
    assert torch.allclose(m(torch.zeros(batch_size, 5)), expected)


def test_SynCurrent_adds_impulses(constructor, spikes):
    conf, batch_size, dt = constructor
    o = 5
    conf.tau_I = np.random.rand()
    alpha = util.decayconst(conf.tau_I, dt)
    m = ce.SynCurrent.configured(*constructor)
    X = spikes(batch_size, o) * np.random.rand(batch_size, o)
    expected = X * (1-alpha)
    assert torch.allclose(m(X), expected)
