import pytest
from cantata import config, util, init
import cantata.elements as ce
import torch
import numpy as np

def test_Membrane_does_not_modify_children(module_tests, model1, spikes):
    batch_size, dt = 32, 1e-3
    m = ce.Membrane.configured(model1.areas.A1, batch_size, dt)
    X = spikes(batch_size,5)
    current = torch.rand_like(X)
    module_tests.check_no_child_modification(m, X, current)

def test_Membrane_can_change_device(model1, spikes):
    batch_size, dt = 32, 1e-3
    m = ce.Membrane.configured(model1.areas.A1, batch_size, dt)
    for device in [torch.device('cuda'), torch.device('cpu')]:
        X = spikes(batch_size,5).to(device)
        current = torch.zeros_like(X)
        m.to(device)
        V = m(X, current)
        assert V.device == X.device

def test_Membrane_state(model1, batch_size, dt, module_tests):
    module_tests.check_state(
        ce.Membrane.configured(model1.areas.A1, batch_size, dt),
        ['alpha', 'V', 'ref'],
        [(5,), (batch_size,5), (batch_size,5)]
    )

def test_Membrane_reset_clears_ref(model1, batch_size, dt, module_tests):
    module_tests.check_reset_clears(
        ce.Membrane.configured(model1.areas.A1, batch_size, dt), 'ref')

def test_Membrane_reset_clears_V(model1, batch_size, dt):
    m = ce.Membrane.configured(model1.areas.A1, batch_size, dt)
    V = m.V.clone()
    m.reset()
    assert not torch.equal(V, m.V)
    assert m.V.min() >= 0
    assert m.V.max() <= 1

def test_Membrane_reset_preserves_alpha(model1, batch_size, dt):
    m = ce.Membrane.configured(model1.areas.A1, batch_size, dt)
    alpha = m.alpha.clone()
    m.reset()
    assert torch.equal(alpha, m.alpha)

def test_Membrane_V_decays(model1, batch_size, dt):
    m = ce.Membrane.configured(model1.areas.A1, batch_size, dt)
    assert torch.all(m.alpha < 1)
    assert torch.all(m.alpha > 0)
    expected = m.V * m.alpha
    V_ret = m(torch.zeros(batch_size,5), torch.zeros(batch_size,5))
    assert torch.equal(V_ret, expected)
    assert torch.equal(m.V, expected)

def test_Membrane_adds_current(model1, batch_size, dt):
    m = ce.Membrane.configured(model1.areas.A1, batch_size, dt)
    current = torch.rand(batch_size, 5)
    expected = m.V * m.alpha + current
    V_ret = m(torch.zeros_like(current), current)
    assert torch.allclose(expected, V_ret)

def test_Membrane_resets_spikes(model1, batch_size, dt, spikes):
    m = ce.Membrane.configured(model1.areas.A1, batch_size, dt)
    X = spikes(batch_size, 5)
    expected = m.V * m.alpha
    expected[X>0] = 0
    V_ret = m(X, torch.zeros_like(X))
    assert torch.allclose(expected, V_ret)

def test_Membrane_adds_noise(model1_noisy, batch_size, dt):
    m = ce.Membrane.configured(model1_noisy.areas.A1, batch_size, dt)
    assert m.noisy
    noise = m.noise()
    m.noise.forward = lambda *args, **kwargs: noise
    current = torch.rand(batch_size, 5)
    m.V = torch.zeros_like(m.V)
    V = m(torch.zeros_like(current), current)
    assert torch.allclose(V, noise + current)

class TestRefractoryDynamics:
    @pytest.fixture(scope='class', autouse=True)
    def dynamics(self, model1, batch_size, dt, spikes):
        model1.areas.A1.populations.Exc.tau_ref = np.random.rand()
        model1.areas.A1.populations.Inh.tau_ref = np.random.rand()
        m = ce.Membrane.configured(model1.areas.A1, batch_size, dt)
        X = spikes(batch_size, 5)
        current = torch.rand_like(X)
        m.ref = (5*torch.rand(batch_size,5)).to(m.ref)
        m.V = torch.zeros_like(m.V)
        zeros = (m.ref==0) * (X==0)
        ones = (m.ref==1) * (X==0)
        twos = (m.ref==2) * (X==0)
        V = m(X, current)
        return m, X, current, V, zeros, ones, twos

    def test_Membrane_spikes_reset_V(self, dynamics, spikes):
        m, X, current, V, zeros, ones, twos = dynamics
        assert torch.all(V[X>0] == 0)

    def test_Membrane_spikes_incur_refractory(self, dynamics, model1, dt):
        m, X, current, V, zeros, ones, twos = dynamics
        taue = model1.areas.A1.populations.Exc.tau_ref
        taui = model1.areas.A1.populations.Inh.tau_ref
        assert torch.all(m.ref[:,:2][X[:,:2]>0] == int(np.round(taue/dt)) - 1)
        assert torch.all(m.ref[:,2:][X[:,2:]>0] == int(np.round(taui/dt)) - 1)

    def test_Membrane_current_is_added_to_nonrefractory(self, dynamics):
        m, X, current, V, zeros, ones, twos = dynamics
        assert torch.equal(V[zeros], current[zeros])
        assert torch.all(V[~zeros] != current[~zeros])

    def test_Membrane_no_negative_refractory_periods(self, dynamics):
        m, X, current, V, zeros, ones, twos = dynamics
        assert torch.all(m.ref >= 0)
        # in particular, zeros stay zero:
        assert torch.all(m.ref[zeros] == 0)

    def test_Membrane_refractory_period_decreases(self, dynamics):
        m, X, current, V, zeros, ones, twos = dynamics
        assert torch.all(m.ref[ones] == 0)
        assert torch.all(m.ref[twos] == 1)

    def test_Membrane_refractory_V_stays_zero(self, dynamics):
        m, X, current, V, zeros, ones, twos = dynamics
        assert torch.all(V[ones] == 0)
        assert torch.all(V[twos] == 0)
