import pytest
from cantata import init, cfg
import torch
import numpy as np

def test_get_N_is_accurate(model_1):
    assert init.get_N() == 5

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
    expected = torch.tensor([1,1,2,2,2], **cfg.tspec)
    assert torch.equal(init.expand_to_neurons('test_dummy'), expected)

def test_expand_to_neurons_diag(model_1):
    expected = torch.tensor([1,1,2,2,2], **cfg.tspec).diag()
    assert torch.equal(init.expand_to_neurons('test_dummy', True), expected)

def test_build_projections_params(model_1):
    expected = [cfg.model.populations.Exc1.targets.Exc1,
                cfg.model.populations.Exc1.targets.Inh1,
                cfg.model.populations.Inh1.targets.Exc1,
                cfg.model.populations.Inh1.targets.Inh1]
    _, received = init.build_projections()
    assert received == expected

def test_build_projections_indices(model_1):
    exc = np.array([[0,1]])
    inh = np.array([[2,3,4]])
    expected = [
        (exc.T, exc),
        (exc.T, inh),
        (inh.T, exc),
        (inh.T, inh)
    ]
    received, _ = init.build_projections()
    mat = np.arange(25).reshape((5,5))
    assert np.all([np.all(mat[a]==mat[b]) for a,b in zip(expected,received)])

def test_build_projections_indices_2(model_2):
    e1 = np.arange(150).reshape(1,-1)
    i1 = np.arange(150,250).reshape(1,-1)
    e2 = np.arange(250,300).reshape(1,-1)
    expected = [
        (e1.T, e1),
        (e1.T, i1),
        (e1.T, e2),
        (i1.T, e1),
        (i1.T, i1),
        (e2.T, e1)
    ]
    received, _ = init.build_projections()
    mat = np.empty((300,300))
    assert np.all([np.all(mat[a]==mat[b]) for a,b in zip(expected,received)])

def test_build_connectivity_densities(model_2):
    indices, params = init.build_projections()
    w = init.build_connectivity((indices, params))
    expected = np.array([
        0.1 * 150 * 150, # exc1->exc1
        0.3 * 150 * 100, # exc1->inh1
        0.8 * 150 * 50, # exc1->exc2
        0, # inh1->exc1
        0.5 * 100 * 100, # inh1->inh1
        1 * 50 * 150 # exc2->exc1
    ])
    received = np.array([np.count_nonzero(w[idx]) for idx in indices])
    assert np.allclose(expected, received, atol=500)

def test_build_connectivity_no_spurious_connections(model_2):
    indices, params = init.build_projections()
    w = init.build_connectivity((indices, params))
    mask = np.ones((300,300), dtype=np.bool)
    for idx in indices:
        mask[idx] = False
    assert np.count_nonzero(w[mask]) == 0

def test_build_connectivity_scaling(model_2):
    indices, params = init.build_projections()
    w = init.build_connectivity((indices, params))
    wscale = cfg.model.weight_scale*(1-np.exp(-cfg.time_step/cfg.model.tau_mem))
    density = lambda p: p.density if 'density' in p else 1
    expected = np.array([
        wscale / np.sqrt(len(idx[0]) * density(p)) if density(p)>0 else np.nan
        for idx,p in zip(indices,params)
    ])
    received = np.array([
        torch.std(w[idx] [w[idx] != 0]).item()
        for idx in indices
    ])
    assert np.allclose(received, expected, atol=.05, equal_nan=True)

def test_build_delay_mapping_delays(model_1):
    projections = init.build_projections()
    _, delays = init.build_delay_mapping(projections)
    expected = torch.tensor([0, 5, 10], dtype=delays.dtype)
    assert torch.equal(delays, expected)

def test_build_delay_mapping_dmap(model_1):
    indices, params = init.build_projections()
    dmap, _ = init.build_delay_mapping((indices, params))
    exc = np.array([[0,1]])
    inh = np.array([[2,3,4]])
    expected = torch.zeros(3,5,5, device=cfg.tspec.device, dtype=torch.bool)
    expected[0, inh.T, inh] = True
    expected[1, exc.T, exc] = True
    expected[1, exc.T, inh] = True
    expected[2, inh.T, exc] = True
    assert torch.equal(dmap, expected)
