from cantata import util
import cantata.elements as ce
import torch
import numpy as np


def test_ALIF_state(model1, batch_size, dt, module_tests):
    model1.areas.A1.populations.Exc.th_ampl = 0.5
    m = ce.ALIFSpikes(model1.areas.A1, batch_size, dt)
    module_tests.check_state(m, ['threshold'], [(batch_size, 5)])


def test_ALIF_reset_clears_threshold(model1, batch_size, dt, module_tests):
    model1.areas.A1.populations.Exc.th_ampl = 0.5
    m = ce.ALIFSpikes(model1.areas.A1, batch_size, dt)
    module_tests.check_reset_clears(m, 'threshold')


def test_ALIF_spikes_above_static_threshold(model1, batch_size, dt):
    m = ce.ALIFSpikes(model1.areas.A1, batch_size, dt)
    assert not m.adaptive
    V = torch.rand(batch_size, 5) * 2
    subthreshold = V < 1
    X, Xd = m(V)
    assert torch.all(X[subthreshold] == 0)
    assert torch.all(X[~subthreshold] == 1)


def test_ALIF_spikes_above_adaptive_threshold(model1, batch_size, dt):
    model1.areas.A1.populations.Exc.th_ampl = 0.5
    m = ce.ALIFSpikes(model1.areas.A1, batch_size, dt)
    assert m.adaptive
    m.threshold = torch.rand_like(m.threshold)
    V = torch.rand(batch_size, 5) * 2
    subthreshold = V < m.threshold + 1
    X, Xd = m(V)
    assert torch.all(X[subthreshold] == 0)
    assert torch.all(X[~subthreshold] == 1)


def test_ALIF_spikes_increase_adaptive_threshold(model1, batch_size, dt):
    amp = np.random.rand()
    model1.areas.A1.populations.Exc.th_ampl = amp
    m = ce.ALIFSpikes(model1.areas.A1, batch_size, dt)
    V = torch.rand(batch_size, 5) * 2
    spiking_Exc = V[:, :2] >= 1
    m(V)
    assert torch.all(m.threshold[:, :2][spiking_Exc] == amp)
    assert torch.all(m.threshold[:, :2][~spiking_Exc] == 0)
    assert torch.all(m.threshold[:, 2:] == 0)


def test_ALIF_adaptive_threshold_decays(model1, batch_size, dt):
    amp, tau = torch.rand(2)
    alpha_exc = util.decayconst(tau, dt)
    alpha_inh = util.decayconst(model1.areas.A1.populations.Inh.th_tau, dt)
    model1.areas.A1.populations.Exc.th_ampl = amp.item()
    model1.areas.A1.populations.Exc.th_tau = tau.item()
    m = ce.ALIFSpikes(model1.areas.A1, batch_size, dt)
    m.threshold = torch.rand_like(m.threshold)
    V = torch.zeros(batch_size, 5)
    expected = m.threshold.clone()
    expected[:, :2] *= alpha_exc
    expected[:, 2:] *= alpha_inh
    m(V)
    assert torch.allclose(m.threshold, expected)


def test_ALIF_Xd_returns_at_internal_delays(model1, batch_size, dt):
    dt_per_ms = int(np.round(1e-3/dt))
    m = ce.ALIFSpikes(model1.areas.A1, batch_size, dt)
    Xt = []
    for t in range(10*dt_per_ms):
        X, Xd = m(torch.rand(batch_size, 5) * 2)
        Xt.append(X)
    X, Xd = m(torch.zeros(batch_size, 5))
    assert torch.equal(Xd[0], Xt[-1])  # Min delay, Inh->Inh
    assert torch.equal(Xd[1], Xt[-5*dt_per_ms])  # 5 ms, Exc->*
    assert torch.equal(Xd[2], Xt[-10*dt_per_ms])  # 10 ms, Inh->Exc
    assert len(Xd) == 3


def test_ALIF_applies_surrogate_gradient(model1, batch_size, dt):
    m = ce.ALIFSpikes(model1.areas.A1, batch_size, dt)
    V = torch.rand(batch_size, 5).requires_grad_()
    X, Xd = m(V)
    assert type(X.grad_fn) == ce.alif.SurrGradSpike._backward_cls


def test_ALIF_can_change_device(model1, batch_size, dt):
    model1.areas.A1.populations.Exc.th_ampl = 0.5
    m = ce.ALIFSpikes(model1.areas.A1, batch_size, dt)
    for device in [torch.device('cuda'), torch.device('cpu')]:
        V = torch.rand(batch_size, 5, device=device)
        m.to(device)
        X, Xd = m(2*V)
        assert X.device == V.device
        for x in Xd:
            assert x.device == V.device


def test_ALIF_does_not_modify_children(module_tests, model1, batch_size, dt):
    m = ce.ALIFSpikes(model1.areas.A1, batch_size, dt)
    V = torch.rand(batch_size, 5)
    module_tests.check_no_child_modification(m, V)
