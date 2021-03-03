import pytest
from cantata import config, util, init
import cantata.elements as ce
import torch
import numpy as np

@pytest.fixture(params = [(5,), (5,6), (5,6,7)])
def shape(request):
    return request.param

def test_DelayBuffer_can_change_device(shape):
    delays = [2,3], [4]
    m = ce.DelayBuffer(shape, *delays)
    for device in [torch.device('cuda'), torch.device('cpu')]:
        X = torch.rand(shape).to(device)
        m.to(device)
        Xd1, Xd2 = m(X)
        for Xd in [Xd1, Xd2]:
            assert Xd.device == X.device

def test_DelayBuffer_maintains_shape(shape):
    delays = [2], [3,4]
    m = ce.DelayBuffer(shape, *delays)
    X1, X2 = m(torch.rand(shape))
    assert X1.shape == (1,) + shape
    assert X2.shape == (2,) + shape

def test_DelayBuffer_returns_None_for_empty_delay(shape):
    delays = [], [3,4]
    m = ce.DelayBuffer(shape, *delays)
    X1, X2 = m(torch.rand(shape))
    assert X1 is None

def test_DelayBuffer_accepts_completely_empty_delay(shape):
    delays = []
    m = ce.DelayBuffer(shape, delays)
    Xd, = m(torch.rand(shape))
    assert Xd is None

def test_DelayBuffer_rejects_invalid_delays(shape):
    delays = [-1, 4]
    with pytest.raises(ValueError):
        m = ce.DelayBuffer(shape, delays)

def test_DelayBuffer_returns_zero_delay_immediately(shape):
    delays = [0,3]
    m = ce.DelayBuffer(shape, delays)
    X = torch.rand(shape)
    Xd, = m(X)
    assert torch.equal(Xd[0], X)

def test_DelayBuffer_respects_order(shape):
    delays = [np.random.randint(10) for _ in range(5)]
    i = np.random.randint(6)
    delays.insert(i, 0)
    m = ce.DelayBuffer(shape, delays)
    X = torch.rand(shape)
    Xd, = m(X)
    assert torch.equal(Xd[i], X)

def test_DelayBuffer_delays_appropriately(shape):
    delays = [np.random.randint(10) for _ in range(5)]
    i = np.random.randint(6)
    D = np.random.randint(10)
    delays.insert(i, D)
    m = ce.DelayBuffer(shape, delays)
    X0 = torch.rand(shape)
    Xd, = m(X0)
    for j in range(D):
        Xd, = m(torch.rand(shape))
    assert torch.equal(Xd[i], X0)

def test_DelayBuffer_maintains_grad(shape):
    delays = [5,1]
    m = ce.DelayBuffer(shape, delays)
    X, Y = torch.rand(shape), torch.rand(shape)
    X1, X2 = X.clone().requires_grad_(), X.clone().requires_grad_()
    m(X1)
    Xd, = m(X1)
    Xd[1].sum().backward()
    X2.sum().backward()
    assert torch.equal(X1.grad, X2.grad)

def test_DelayBuffer_reset_clears_buffers(shape):
    t = np.random.randint(10) + 1
    m = ce.DelayBuffer(shape, [t])
    for i in range(t):
        m(torch.rand(shape))
    m.reset()
    for i in range(t):
        Xd, = m(torch.rand(shape))
        assert torch.all(Xd == 0), (t,i)
