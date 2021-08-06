import pytest
from cantata import util
import torch
import numpy as np
import sys


@pytest.fixture(params=['torch', 'numpy'])
def framework(request):
    if request.param == 'torch':
        return dict(
            fw=torch,
            rand=torch.rand,
            randn=torch.randn,
            tensor=torch.tensor)
    else:
        return dict(
            fw=np,
            rand=np.random.rand,
            randn=np.random.randn,
            tensor=np.array)


def test_decayconst():
    dt, tau = np.random.rand(2)
    expected = np.exp(-dt/tau)
    assert np.allclose(util.decayconst(tau, dt), expected)


def test_decayconst_is_float():
    assert type(util.decayconst(0.1, 0.2)) == float


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


def test_expfilt_to_zero():
    N = 100
    target = np.zeros(N)
    filtered = np.random.randn(100)
    alpha = 0.9
    expected = 0.9 * filtered
    received = util.expfilt(target, filtered, alpha)
    assert np.allclose(expected, received)


def test_expfilt_from_zero():
    N = 100
    filtered = np.zeros(N)
    target = np.random.randn(100)
    alpha = 0.7
    expected = 0.3 * target
    received = util.expfilt(target, filtered, alpha)
    assert np.allclose(expected, received)


def test_expfilt_is_decay_to_target():
    N = 100
    target = np.random.randn(N)
    filtered = np.random.randn(N)
    diff = target - filtered
    alpha = 0.8
    expected = filtered + 0.2*diff
    received = util.expfilt(target, filtered, alpha)
    assert np.allclose(expected, received)


def test_sunflower_NYI():
    assert False


def test_broadcast_outer_NYI():
    assert False


def test_polar_dist_NYI():
    assert False


def test_cartesian_NYI():
    assert False
