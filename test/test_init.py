import pytest
from cantata import init, cfg
import torch
import numpy as np
from box import Box

def test_get_N_is_accurate(model_1):
    assert init.get_N() == 6

def test_get_N_caches(model_1):
    cached = init.get_N()
    cfg.model.populations.Exc1.n += 1
    assert init.get_N() == cached

def test_get_N_recalculates_when_forced(model_1):
    cached = init.get_N()
    cfg.model.populations.Exc1.n += 1
    assert init.get_N(force_calculate = True) == cached+1

def test_get_N_respects_invalid_preset(model_1):
    cached = cfg.model.N = 'some silly preset'
    assert init.get_N() == cached

def test_expand_to_neurons_linear(model_1):
    expected = torch.tensor([0,1,1,2,2,2], **cfg.tspec)
    assert torch.equal(init.expand_to_neurons('test_dummy'), expected)

def test_expand_to_neurons_diag(model_1):
    d = np.random.rand()
    expected = torch.tensor([d,1,1,2,2,2], **cfg.tspec).diag()
    assert torch.equal(init.expand_to_neurons('test_dummy', True, default=d), expected)

def test_expand_to_synapses_uses_default_value(model_1):
    d = np.random.rand()
    expected = torch.tensor([
        [d,d,d,d,d,d],
        [d,1,1,2,2,2],
        [d,1,1,2,2,2],
        [d,3,3,4,4,4],
        [d,3,3,4,4,4],
        [d,3,3,4,4,4]
    ], **cfg.tspec)
    proj = init.build_projections(*init.build_population_indices())
    assert torch.equal(init.expand_to_synapses('test_dummy', proj, default=d), expected)

def test_expand_to_synapses(model_1):
    expected = torch.tensor([
        [0,0,0,0,0,0],
        [0,1,1,2,2,2],
        [0,1,1,2,2,2],
        [0,3,3,4,4,4],
        [0,3,3,4,4,4],
        [0,3,3,4,4,4]
    ], **cfg.tspec)
    proj = init.build_projections(*init.build_population_indices())
    assert torch.equal(init.expand_to_synapses('test_dummy', proj), expected)

def test_build_population_indices_names(model_1):
    expected = ['Inp', 'Exc1', 'Inh1']
    names, _ = init.build_population_indices()
    assert names == expected

def test_build_population_indices_ranges(model_1):
    expected = [range(0,1), range(1,3), range(3,6)]
    _, ranges = init.build_population_indices()
    assert ranges == expected

def test_build_output_projections_indices(model_2):
    e1 = np.arange(150).reshape(1,-1).T
    i1 = np.arange(150,250).reshape(1,-1).T
    e2 = np.arange(250,300).reshape(1,-1).T
    expected = [
        (e1, np.array([1])),
        (i1, np.array([0]))
        # Omitted: e2, in* -> none
    ]
    received, _ = init.build_output_projections(*init.build_population_indices())
    assert len(received) == len(expected)
    assert np.all([np.all(a[i] == b[i])
        for a,b in zip(expected,received) for i in (0,1)])

def test_build_output_projections_density(model_2):
    expected = [Box({'density': 1.0,'spatial': False}),
                Box({'density': 1.0,'spatial': False})]
    _, received = init.build_output_projections(*init.build_population_indices())
    assert received == expected

def test_build_projections_params(model_1):
    expected = [cfg.model.populations.Inp.targets.Exc1,
                cfg.model.populations.Exc1.targets.Exc1,
                cfg.model.populations.Exc1.targets.Inh1,
                cfg.model.populations.Inh1.targets.Exc1,
                cfg.model.populations.Inh1.targets.Inh1]
    _, received = init.build_projections(*init.build_population_indices())
    assert received == expected

def test_build_projections_indices(model_1):
    inp = np.array([[0]])
    exc = np.array([[1,2]])
    inh = np.array([[3,4,5]])
    expected = [
        (inp.T, exc),
        (exc.T, exc),
        (exc.T, inh),
        (inh.T, exc),
        (inh.T, inh)
    ]
    received, _ = init.build_projections(*init.build_population_indices())
    assert len(expected) == len(received)
    assert np.all([np.all(a[i]==b[i])
        for a,b in zip(expected,received) for i in (0,1)])

def test_build_projections_indices_2(model_2):
    e1 = np.arange(150).reshape(1,-1)
    i1 = np.arange(150,250).reshape(1,-1)
    e2 = np.arange(250,300).reshape(1,-1)
    n0 = np.arange(300,340).reshape(1,-1)
    n1 = np.arange(340,370).reshape(1,-1)
    expected = [
        (e1.T, e1),
        (e1.T, i1),
        (e1.T, e2),
        (i1.T, e1),
        (i1.T, i1),
        (e2.T, e1),
        (n0.T, e1),
        (n0.T, i1),
        (n1.T, e2)
    ]
    received, _ = init.build_projections(*init.build_population_indices())
    assert len(expected) == len(received)
    assert np.all([np.all(a[i] == b[i])
        for a,b in zip(expected,received) for i in [0,1]])

def test_build_connectivity_densities(model_2):
    indices, params = init.build_projections(*init.build_population_indices())
    w = init.build_connectivity((indices, params))
    expected = np.array([
        0.1 * 150 * 150, # exc1->exc1
        0.3 * 150 * 100, # exc1->inh1
        0.8 * 150 * 50, # exc1->exc2
        0, # inh1->exc1
        0.5 * 100 * 100, # inh1->inh1
        1 * 50 * 150, # exc2->exc1
        0.1 * 40 * 150, # in0->exc1
        0.1 * 40 * 100, # in0->inh1
        0.1 * 30 * 50 # in1 -> exc2
    ])
    received = np.array([np.count_nonzero(w[idx].cpu()) for idx in indices])
    assert np.allclose(expected, received, atol=500)

def test_build_connectivity_no_spurious_connections(model_2):
    indices, params = init.build_projections(*init.build_population_indices())
    w = init.build_connectivity((indices, params))
    mask = np.ones((370,370), dtype=np.bool)
    for idx in indices:
        mask[idx] = False
    assert torch.count_nonzero(w[mask]) == 0

def test_build_connectivity_distribution(model_2):
    indices, params = init.build_projections(*init.build_population_indices())
    w = init.build_connectivity((indices, params))
    for idx in indices:
        ww = w[idx][w[idx]!=0]
        bounds = np.random.rand(5) * 0.9
        counts = np.array([torch.sum((lo < ww) * (ww < lo+.1)).item()
                           for lo in bounds])
        assert np.allclose(counts-counts.mean(), np.zeros_like(counts),
                           atol=max(30,.35*counts.mean()))

def test_build_delay_mapping_delays(model_1):
    projections = init.build_projections(*init.build_population_indices())
    _, delays = init.build_delay_mapping(projections)
    expected = [0, 5, 10]
    assert delays == expected

def test_build_delay_mapping_dmap(model_1):
    indices, params = init.build_projections(*init.build_population_indices())
    dmap, _ = init.build_delay_mapping((indices, params))
    inp = np.array([[0]])
    exc = np.array([[1,2]])
    inh = np.array([[3,4,5]])
    expected = [torch.zeros(6,6, dtype=torch.bool, device=cfg.tspec.device)
                for _ in range(3)]
    expected[0][inp.T, exc] = True
    expected[0][inh.T, inh] = True
    expected[1][exc.T, exc] = True
    expected[1][exc.T, inh] = True
    expected[2][inh.T, exc] = True
    assert len(dmap) == 3
    for i in range(3):
        assert torch.equal(dmap[i], expected[i]), i

def test_delays_are_truncated_to_runtime(model_1):
    cfg.n_steps = 7 + int(np.random.rand()*4) # [7,10]
    projections = init.build_projections(*init.build_population_indices())
    _, delays = init.build_delay_mapping(projections)
    expected = [0,5,cfg.n_steps-1]
    assert delays == expected

def test_get_input_spikes_density_mirrors_rate(model_1):
    seconds = 20
    cfg.n_steps = 1000 * seconds
    cfg.batch_size = 128
    cfg.n_inputs = 1
    rates = torch.rand(cfg.batch_size, 1, cfg.n_inputs, **cfg.tspec)*100 # Hz
    spikes = init.get_input_spikes(rates.expand(-1, cfg.n_steps, -1))
    received = spikes[:,:,0].sum(dim=1) / seconds # Hz
    assert torch.allclose(received, rates.squeeze(), atol=10)

def test_get_input_spikes_no_spurious_inputs(model_1):
    cfg.n_steps = 100
    cfg.batch_size = 128
    cfg.n_inputs = 1
    rates = torch.rand(cfg.batch_size, 1, cfg.n_inputs, **cfg.tspec)*100 # Hz
    spikes = init.get_input_spikes(rates.expand(-1, cfg.n_steps, -1))
    assert spikes[:,:,1:].detach().count_nonzero() == 0

def test_get_input_spikes_independent_of_dt(model_1):
    cfg.time_step = np.random.rand() * 5e-3 + .5e-3 # s
    cfg.n_steps = 20000
    cfg.batch_size = 128
    cfg.n_inputs = 1
    rates = torch.rand(cfg.batch_size, 1, cfg.n_inputs, **cfg.tspec)*100 # Hz
    spikes = init.get_input_spikes(rates.expand(-1, cfg.n_steps, -1))
    received = spikes[:,:,0].sum(dim=1) / (cfg.time_step * cfg.n_steps) # Hz
    assert torch.allclose(received, rates.squeeze(), atol=10)
