import pytest
from cantata import init, config
import torch
import numpy as np
from box import Box

def test_get_N(model1):
    assert init.get_N(model1.input) == 1
    assert init.get_N(model1.areas.A1) == 5
    assert init.get_N(model1.areas.A2) == 10

def test_expand_to_neurons_linear(model1):
    expected = torch.tensor([1,1,2,2,2])
    assert torch.equal(
        init.expand_to_neurons(model1.areas.A1, 'test_dummy'),
        expected)

def test_expand_to_neurons_uses_default(model1):
    del model1.areas.A1.populations.Exc.test_dummy
    d = np.random.rand()
    expected = torch.tensor([d,d,2,2,2])
    assert torch.equal(
        init.expand_to_neurons(model1.areas.A1, 'test_dummy', default=d),
        expected)

def test_expand_to_neurons_diagonal(model1):
    expected = torch.tensor([1,1,2,2,2]).diag()
    assert torch.equal(
        init.expand_to_neurons(model1.areas.A1, 'test_dummy', True),
        expected)

def test_expand_to_synapses_uses_default_value(model1):
    del model1.areas.A1.populations.Inh.targets.Exc.test_dummy
    d = np.random.rand()
    expected = torch.tensor([
        [1,1,2,2,2],
        [1,1,2,2,2],
        [d,d,4,4,4],
        [d,d,4,4,4],
        [d,d,4,4,4]
    ])
    proj = init.build_projections(model1.areas.A1)
    assert torch.equal(
        init.expand_to_synapses(proj, 5, 5, 'test_dummy', default=d),
        expected)

def test_expand_to_synapses(model1):
    expected = torch.tensor([
        [1,1,2,2,2],
        [1,1,2,2,2],
        [3,3,4,4,4],
        [3,3,4,4,4],
        [3,3,4,4,4]
    ], dtype=torch.float)
    proj = init.build_projections(model1.areas.A1)
    assert torch.equal(
        init.expand_to_synapses(proj, 5, 5, 'test_dummy'),
        expected)

def test_build_population_indices_names(model1):
    expected = ['Exc', 'Inh']
    names, _ = init.build_population_indices(model1.areas.A1)
    assert names == expected

def test_build_population_indices_ranges(model1):
    expected = [range(0,2), range(2,5)]
    _, ranges = init.build_population_indices(model1.areas.A1)
    assert ranges == expected

def test_build_projections_params(model1):
    expected = [model1.areas.A1.populations.Exc.targets.Exc,
                model1.areas.A1.populations.Exc.targets.Inh,
                model1.areas.A1.populations.Inh.targets.Exc,
                model1.areas.A1.populations.Inh.targets.Inh]
    _, received = init.build_projections(model1.areas.A1)
    assert received == expected

def test_build_projections_indices(model1):
    exc = np.array([[0,1]])
    inh = np.array([[2,3,4]])
    expected = [
        (exc.T, exc),
        (exc.T, inh),
        (inh.T, exc),
        (inh.T, inh)
    ]
    received, _ = init.build_projections(model1.areas.A1)
    assert len(expected) == len(received)
    assert np.all([np.all(a[i]==b[i])
        for a,b in zip(expected,received) for i in (0,1)])

def test_build_projections_indices_2(model2):
    e1 = np.arange(150).reshape(1,-1)
    i1 = np.arange(150,250).reshape(1,-1)
    e2 = np.arange(250,300).reshape(1,-1)
    expected = [
        (e1.T, e1),
        (e1.T, i1),
        (e1.T, e2),
        (i1.T, i1),
        (i1.T, e1),
        (e2.T, e1)
    ]
    received, _ = init.build_projections(model2.areas.A1)
    assert len(expected) == len(received)
    assert np.all([np.all(a[i] == b[i])
        for a,b in zip(expected,received) for i in [0,1]])

def test_build_projections_xarea_params(model2):
    expected = [model2.input.populations.In0.targets['A1:Exc1'],
                model2.input.populations.In0.targets['A1:Inh1'],
                model2.input.populations.In1.targets['A1:Exc2']]
    _, received = init.build_projections(
        model2.input, model2.areas.A1, 'A1')
    assert received == expected

def test_build_projections_xarea_indices(model2):
    e1 = np.arange(150).reshape(1,-1)
    i1 = np.arange(150,250).reshape(1,-1)
    e2 = np.arange(250,300).reshape(1,-1)
    in0 = np.arange(40).reshape(-1,1)
    in1 = np.arange(40,70).reshape(-1,1)
    expected = [
        (in0, e1),
        (in0, i1),
        (in1, e2)
    ]
    received, _ = init.build_projections(
        model2.input, model2.areas.A1, 'A1')
    assert len(expected) == len(received)
    assert np.all([np.all(a[i] == b[i])
        for a,b in zip(expected,received) for i in [0,1]])

@pytest.mark.parametrize('batch_size_', [0,1,32])
def test_build_connectivity_densities(model2, batch_size_):
    (indices, params) = init.build_projections(model2.areas.A1)
    w = init.build_connectivity((indices, params), 300, 300, batch_size_)
    expected = np.round([
        0.1 * 150 * 150, # exc1->exc1
        0.3 * 150 * 100, # exc1->inh1
        0.8 * 150 * 50, # exc1->exc2
        0.5 * 100 * 100, # inh1->inh1
        0, # inh1->exc1
        1 * 50 * 150, # exc2->exc1
    ])
    if batch_size_ == 0:
        received = np.array([
            np.count_nonzero(w[idx]) for idx in indices])
    else:
        expected = np.broadcast_to(expected, (batch_size_, len(expected)))
        received = np.zeros_like(expected)
        for i,idx in enumerate(indices):
            received[:,i] = torch.sum(w[:,idx[0],idx[1]] != 0, dim=(1,2))
    assert np.all(expected == received)

@pytest.mark.parametrize('batch_size_', [0,1,32])
def test_build_connectivity_respects_size(model1, batch_size_):
    (indices, params) = init.build_projections(
        model1.input, model1.areas.A1, 'A1')
    expected = torch.tensor([
        [1, 1, 0, 0, 0.]
    ])
    if batch_size_ > 0:
        expected = expected.unsqueeze(0).expand(batch_size_, 1, 5)
    w = init.build_connectivity((indices, params), 1, 5, batch_size_)
    received = torch.where(w>0, torch.ones(1), torch.zeros(1))
    assert torch.equal(expected, received)

@pytest.mark.parametrize('batch_size_', [0,1,32])
def test_build_connectivity_no_spurious_connections(model2, batch_size_):
    indices, params = init.build_projections(model2.areas.A1)
    w = init.build_connectivity((indices, params), 300, 300, batch_size_)
    mask = torch.ones((300,300), dtype=torch.bool)
    for idx in indices:
        mask[idx] = False
    if batch_size_ > 0:
        mask = mask.unsqueeze(0).expand(batch_size_, -1, -1)
    assert torch.count_nonzero(w[mask]) == 0

def test_build_connectivity_distribution(model2):
    indices, params = init.build_projections(model2.areas.A1)
    w = init.build_connectivity((indices, params), 300, 300)
    for idx in indices:
        ww = w[idx][w[idx]!=0]
        bounds = np.random.rand(5) * 0.9
        counts = np.array([torch.sum((lo < ww) * (ww < lo+.1)).item()
                           for lo in bounds])
        assert np.allclose(counts-counts.mean(), np.zeros_like(counts),
                           atol=max(30,.35*counts.mean()))

def test_get_connection_probabilities_NYI():
    assert False

def test_get_delays_internal(model1, dt):
    delays = init.get_delays(model1.areas.A1, dt, False)
    dt_per_ms = int(np.round(1e-3/dt))
    expected = [1, 5*dt_per_ms, 10*dt_per_ms]
    assert delays == expected

def test_build_delaymap_internal(model1):
    projections = init.build_projections(model1.areas.A1)
    dmap = init.get_delaymap(projections, 1e-3, model1.areas.A1)
    exc = np.array([[0,1]])
    inh = np.array([[2,3,4]])
    expected = torch.zeros(3,5,5)
    expected[0, inh.T, inh] = True
    expected[1, exc.T, exc] = True
    expected[1, exc.T, inh] = True
    expected[2, inh.T, exc] = True
    assert torch.equal(dmap, expected)

def test_delay_xarea_one_less_than_internal(model1, dt):
    delay = np.random.rand() + 5*dt
    internal = init.get_delay(delay, dt, False)
    external = init.get_delay(delay, dt, True)
    assert external == internal - 1

def test_get_delays_xarea_is_sorted_minimal_target_agnostic(model1, dt):
    dt_per_ms = int(np.round(1e-3/dt))
    expected = [1, 15*dt_per_ms - 1]
    delays = init.get_delays(model1.input, dt, True)
    assert delays == expected

def test_get_delaymap_xarea(model1):
    projections = init.build_projections(
        model1.areas.A1, model1.areas.A2, 'A2')
    dmap = init.get_delaymap(
        projections, 1e-3, model1.areas.A1, model1.areas.A2)
    exc = np.arange(2).reshape(-1,1)
    inh = np.arange(2,5).reshape(-1,1)
    deadend = np.arange(4).reshape(1,-1)
    silent = np.arange(4,10).reshape(1,-1)
    expected = torch.zeros(2,5,10)
    expected[0, exc, deadend] = True
    expected[1, inh, silent] = True
    assert torch.equal(dmap, expected)

def test_get_delaymap_xarea_leaves_unused_blank(model1):
    projections = init.build_projections(
        model1.input, model1.areas.A2, 'A2')
    dmap = init.get_delaymap(projections, 1e-3, model1.input, model1.areas.A2)
    expected = torch.zeros(2, 1, 10)
    expected[1, 0, :4] = True
    assert torch.equal(dmap, expected)
