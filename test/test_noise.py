import cantata.elements as ce
import torch


def test_Noise_does_not_modify_children(module_tests, model1_noisy):
    batch_size, dt = 32, 1e-3
    m = ce.Noise(model1_noisy.areas.A1, batch_size, dt)
    module_tests.check_no_child_modification(m)


def test_Noise_inactive_without_noise(model1, batch_size, dt):
    m = ce.Noise(model1.areas.A1, batch_size, dt)
    assert not m.active
    assert torch.all(m() == 0)


def test_Noise_active_with_some_noise(model1_noisy, batch_size, dt):
    m = ce.Noise(model1_noisy.areas.A1, batch_size, dt)
    assert m.active
    assert torch.all(m() > 0)


def test_Noise_inactivates_with_N(model1_noisy, batch_size, dt):
    model1_noisy.areas.A1.populations.Exc.noise_N = 0
    model1_noisy.areas.A1.populations.Inh.noise_N = 0
    m = ce.Noise(model1_noisy.areas.A1, batch_size, dt)
    assert not m.active


def test_Noise_inactivates_with_rate(model1_noisy, batch_size, dt):
    model1_noisy.areas.A1.populations.Exc.noise_rate = 0
    model1_noisy.areas.A1.populations.Inh.noise_rate = 0
    m = ce.Noise(model1_noisy.areas.A1, batch_size, dt)
    assert not m.active


def test_Noise_inactivates_with_weight(model1_noisy, batch_size, dt):
    model1_noisy.areas.A1.populations.Exc.noise_weight = 0
    model1_noisy.areas.A1.populations.Inh.noise_weight = 0
    m = ce.Noise(model1_noisy.areas.A1, batch_size, dt)
    assert not m.active


def test_Noise_can_change_device(model1_noisy):
    batch_size, dt = 32, 1e-3
    m = ce.Noise(model1_noisy.areas.A1, batch_size, dt)
    for device in [torch.device('cuda'), torch.device('cpu')]:
        X = torch.zeros(batch_size, 5).to(device)
        m.to(device)
        out = m()
        assert out.device == X.device


def test_Noise_is_stateless(model1_noisy, batch_size, dt, module_tests):
    module_tests.check_state(
        ce.Noise(model1_noisy.areas.A1, batch_size, dt),
        [], []
    )
