import pytest
from cantata import config, util, init
import cantata.elements as ce
import torch
import numpy as np

class Host(torch.nn.Module):
    def __init__(self, delaymap, wmax):
        super(Host, self).__init__()
        self.register_buffer('delaymap', delaymap)
        self.register_buffer('wmax', wmax)

@pytest.fixture(params = [True, False], ids = ['xarea', 'internal'])
def constructor(model1, request, batch_size, dt):
    conf_pre = model1.areas.A1
    nPre = 5
    if request.param:
        conf_post = model1.areas.A2
        name_post = 'A2'
        nPost = 6
    else:
        conf_post = None
        name_post = None
        nPost = 5
    projections = init.build_projections(conf_pre, conf_post, name_post)
    delaymap = init.get_delaymap(projections, dt, conf_pre, conf_post)
    wmax = torch.rand(nPre, nPost)
    host = Host(delaymap, wmax)
    return projections, host, conf_pre, batch_size, nPre, nPost, dt

def spikes(*shape):
    X = torch.rand(shape) * 2
    X = torch.threshold(X, 1, 0)
    X = torch.clip(X, 0, 1)
    return X

def test_Abbott_can_change_device(constructor):
    m = ce.Abbott(*constructor)
    host, _, batch_size, nPre, nPost = constructor[1:6]
    d = host.delaymap.shape[0]
    for device in [torch.device('cuda'), torch.device('cpu')]:
        Xd = spikes(d, batch_size, nPre).to(device)
        Xpost = spikes(batch_size, nPost).to(device)
        m.to(device)
        host.to(device)
        W = m(Xd, Xpost)
        assert W.device == Xd.device

def test_Abbott_does_not_move_host(constructor):
    host = constructor[1]
    m = ce.Abbott(*constructor)
    dev = host.wmax.device
    for device in [torch.device('cuda'), torch.device('cpu')]:
        m.to(device)
        assert host.wmax.device == dev

def test_Abbott_state(constructor, module_tests):
    host, _, b,e,o = constructor[1:6]
    d = host.delaymap.shape[0]
    module_tests.check_state(
        ce.Abbott(*constructor),
        ['xbar_pre', 'xbar_post', 'W'],
        [(d,b,e),    (b,o),   (b,e,o)]
    )

def test_Abbott_reset_clears_xbar(constructor, module_tests):
    module_tests.check_reset_clears(
        ce.Abbott(*constructor),
        'xbar_pre', 'xbar_post',
        inputs = (constructor[1].wmax,)
    )

def test_Abbott_reset_resets_W_to_arg(constructor):
    m = ce.Abbott(*constructor)
    m.W = torch.rand_like(m.W)
    expected = torch.rand_like(constructor[1].wmax)
    m.reset(expected.clone())
    batch = np.random.randint(constructor[3])
    assert torch.equal(m.W[batch], expected)

def test_Abbott_deactivates_when_not_required(constructor):
    # Note that STDP_frac is not within the STDP model's responsibility.
    (_, params) = constructor[0]
    nPost = constructor[5]
    for p in params:
        p.A_p = p.A_d = 0.
    m = ce.Abbott(*constructor)
    assert not m.active

def test_Abbott_returns_premodification_W(constructor):
    host, _, b,e,o = constructor[1:6]
    d = host.delaymap.shape[0]
    m = ce.Abbott(*constructor)
    expected = torch.rand_like(m.W)
    m.W = expected.clone()
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.xbar_post = torch.rand_like(m.xbar_post)
    W = m(spikes(d,b,e), spikes(b,o))
    assert torch.equal(W, expected)
    assert not torch.equal(m.W, expected)

def test_Abbott_bounds_W_on_forward(constructor):
    host, _, b,e,o = constructor[1:6]
    d = host.delaymap.shape[0]
    m = ce.Abbott(*constructor)
    unbounded = torch.randn_like(m.W)
    m.W = unbounded.clone()
    m(spikes(d,b,e), spikes(b,o))
    batch = np.random.randint(b)
    surpass = unbounded[batch] > host.wmax
    assert torch.all(m.W[unbounded < 0] == 0)
    assert torch.all(m.W[batch][surpass] == host.wmax[surpass])

def test_Abbott_potentiates_on_post(constructor):
    host, _, b,e,o = constructor[1:6]
    d = host.delaymap.shape[0]
    m = ce.Abbott(*constructor)
    host.wmax = torch.ones_like(host.wmax)
    m.W = torch.zeros_like(m.W)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.xbar_post = torch.rand_like(m.xbar_post)
    Xpre, Xpost = torch.zeros(d,b,e), torch.zeros(b,o)
    if o == 5: # internal
        pre, post = np.ix_(range(2), range(2,5)) # Exc -> Inh
        delay, A_p = 1, 0.1
        npre, npost = 2, 3
    else: # xarea
        pre, post = np.ix_(range(2), range(4)) # Exc -> A2.deadend
        delay, A_p = 0, 0.3
        npre, npost = 2, 4
    Xpost[:, post] = 1
    expected = m.W.clone()
    dW = m.xbar_pre[delay, :, pre] * A_p
    expected[:, pre, post] += dW.expand(b,npre,npost)
    m(Xpre, Xpost)
    assert torch.equal(m.W, expected)

def test_Abbott_depresses_on_pre(constructor):
    host, _, b,e,o = constructor[1:6]
    d = host.delaymap.shape[0]
    m = ce.Abbott(*constructor)
    host.wmax = torch.ones_like(host.wmax)
    m.W = torch.ones_like(m.W)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.xbar_post = torch.rand_like(m.xbar_post)
    Xpre, Xpost = torch.zeros(d,b,e), torch.zeros(b,o)
    if o == 5: # internal
        pre, post = np.ix_(range(2), range(2,5)) # Exc -> Inh
        delay, A_d = 1, 0.2
        npre, npost = 2, 3
    else: # xarea
        pre, post = np.ix_(range(2,5), range(4,6)) # Inh -> A2.silent
        delay, A_d = 1, 0.4
        npre, npost = 3, 2
    Xpre[delay, :, pre] = 1
    expected = m.W.clone()
    dW = m.xbar_post[:, post] * A_d
    expected[:, pre, post] -= dW.expand(b,npre,npost)
    m(Xpre, Xpost)
    assert torch.equal(m.W, expected)

def test_Abbott_filters_Xpre_with_tau_p(constructor):
    _, host, conf, b,e,o, dt = constructor
    conf.tau_p = np.random.rand()
    d = host.delaymap.shape[0]
    m = ce.Abbott(*constructor)
    Xpre, Xpost = spikes(d,b,e), spikes(b,o)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    alpha = util.decayconst(conf.tau_p, dt)
    expected = util.expfilt(Xpre, m.xbar_pre, alpha)
    m(Xpre, Xpost)
    assert torch.allclose(m.xbar_pre, expected)

def test_Abbott_filters_Xpost_with_tau_d(constructor):
    _, host, conf, b,e,o, dt = constructor
    conf.tau_d = np.random.rand()
    d = host.delaymap.shape[0]
    m = ce.Abbott(*constructor)
    Xpre, Xpost = spikes(d,b,e), spikes(b,o)
    m.xbar_post = torch.rand_like(m.xbar_post)
    alpha = util.decayconst(conf.tau_d, dt)
    expected = util.expfilt(Xpost, m.xbar_post, alpha)
    m(Xpre, Xpost)
    assert torch.allclose(m.xbar_post, expected)