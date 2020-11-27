import pytest
from cantata import util, cfg
import torch
import numpy as np
import sys

@pytest.fixture(params = ['torch', 'numpy'])
def framework(request):
    if request.param == 'torch':
        return dict(
            fw = torch,
            rand = torch.rand,
            randn = torch.randn,
            tensor = torch.tensor)
    else:
        return dict(
            fw = np,
            rand = np.random.rand,
            randn = np.random.randn,
            tensor = np.array)

def test_decayconst_tensor(framework):
    dt = np.random.rand()
    tau = framework['rand'](10)
    expected = framework['fw'].exp(-dt/tau)
    func = util.decayconst_tensor(dt, framework=framework['fw'])
    assert framework['fw'].allclose(func(tau), expected)

def test_decayconst():
    dt = cfg.time_step
    tau = np.random.rand()
    expected = np.exp(-dt/tau)
    assert np.allclose(util.decayconst(tau), expected)

def test_decayconst_relies_on_config():
    original_dt = cfg.time_step
    dt = cfg.time_step = np.random.rand()
    tau = np.random.rand()
    expected = np.exp(-dt/tau)
    assert np.allclose(util.decayconst(tau), expected)
    cfg.time_step = original_dt

def test_decayconst_is_float():
    assert type(util.decayconst(0.1)) == float

@pytest.mark.filterwarnings('ignore: overflow')
def test_sigmoid_project_fixed_points(framework):
    bounds = sorted([np.random.rand()*-10, np.random.rand()*10])
    points = framework['tensor']([-sys.float_info.max, 0, sys.float_info.max])
    expected = framework['tensor']([bounds[0], sum(bounds)/2, bounds[1]])
    received = util.sigmoid_project(points, bounds, framework=framework['fw'])
    assert framework['fw'].allclose(received, expected)

def test_sigmoid_project_preserves_order(framework):
    bounds = sorted([np.random.rand()*-10, np.random.rand()*10])
    points = framework['tensor'](sorted(np.random.randn(100)))*100
    projected = util.sigmoid_project(points, bounds, framework=framework['fw'])
    assert np.all(np.diff(projected) >= 0)
