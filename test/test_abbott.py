import pytest
from cantata import config, util, init
import cantata.elements as ce
import torch
import numpy as np

class Host(torch.nn.Module):
    def __init__(self, delaymap, wmin, wmax):
        super(Host, self).__init__()
        self.register_buffer('delaymap', delaymap)
        self.register_buffer('wmin', wmin)
        self.register_buffer('wmax', wmax)
        W = torch.rand_like(wmin) * (wmax-wmin) + wmin
        self.register_buffer('W', W)

@pytest.fixture(params = [True, False], ids = ['xarea', 'internal'])
def construct(model1, request, batch_size, dt):
    conf_pre = model1.areas.A1
    nPre = 5
    if request.param:
        conf_post = model1.areas.A2
        name_post = 'A2'
        nPost = 10
    else:
        conf_post = None
        name_post = None
        nPost = 5
    projections = init.build_projections(conf_pre, conf_post, name_post)
    ctor = projections, conf_pre, batch_size, nPre, nPost, dt
    m = ce.Abbott(*ctor)
    delaymap = init.get_delaymap(projections, dt, conf_pre, conf_post)
    wmax = torch.rand(nPre, nPost)
    wmin = torch.rand(nPre, nPost) * wmax
    host = Host(delaymap, wmin, wmax)
    m.reset(host)
    return m, host, ctor

def test_Abbott_can_change_device(construct, spikes):
    m, host, constructor = construct
    b,e,o = constructor[2:5]
    d = host.delaymap.shape[0]
    for device in [torch.device('cuda'), torch.device('cpu')]:
        Xd = spikes(d,b,e).to(device)
        Xpost = spikes(b,o).to(device)
        m.to(device)
        host.to(device)
        W = m(Xd, Xpost)
        assert W.device == Xd.device

def test_Abbott_does_not_move_host(construct):
    m, host, constructor = construct
    dev = host.wmax.device
    for device in [torch.device('cuda'), torch.device('cpu')]:
        m.to(device)
        assert host.wmax.device == dev

def test_Abbott_state(construct, module_tests):
    m, host, constructor = construct
    b,e,o = constructor[2:5]
    d = host.delaymap.shape[0]
    module_tests.check_state(
        m,
        ['xbar_pre', 'xbar_post', 'W'],
        [(d,b,e),    (b,o),   (b,e,o)]
    )

def test_Abbott_reset_clears_xbar(construct, module_tests):
    m, host, constructor = construct
    module_tests.check_reset_clears(
        m,
        'xbar_pre', 'xbar_post',
        inputs = (host,)
    )

def test_Abbott_reset_resets_W_to_arg(construct):
    m, host, constructor = construct
    m.W = torch.rand_like(m.W)
    expected = host.W
    m.reset(host)
    batch = np.random.randint(len(m.W))
    assert torch.equal(m.W[batch], expected)

def test_Abbott_deactivates_when_not_required(construct):
    # Note that STDP_frac is not within the STDP model's responsibility.
    *_, constructor = construct
    b,e,o = constructor[2:5]
    (_, params) = constructor[0]
    for p in params:
        p.A_p = p.A_d = 0.
    m = ce.Abbott(*constructor)
    assert not m.active

def test_Abbott_returns_premodification_W(construct, spikes):
    m, host, constructor = construct
    b,e,o = constructor[2:5]
    d = host.delaymap.shape[0]
    expected = torch.rand_like(m.W)
    m.W = expected.clone()
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.xbar_post = torch.rand_like(m.xbar_post)
    W = m(spikes(d,b,e), spikes(b,o))
    assert torch.equal(W, expected)
    assert not torch.equal(m.W, expected)

def test_Abbott_ubounds_W_on_forward(construct, spikes):
    m, host, constructor = construct
    b,e,o = constructor[2:5]
    d = host.delaymap.shape[0]
    unbounded = torch.randn_like(m.W)
    m.W = unbounded.clone()
    m(spikes(d,b,e), spikes(b,o))
    batch = np.random.randint(b)
    surpass = unbounded[batch] > host.wmax
    assert torch.all(m.W[unbounded < 0] == 0)
    assert torch.all(m.W[batch][surpass] == host.wmax[surpass])

def test_Abbott_does_not_lbound_W_on_forward(construct, spikes):
    m, host, constructor = construct
    b,e,o = constructor[2:5]
    d = host.delaymap.shape[0]
    unbounded = torch.randn_like(m.W)
    m.W = unbounded.clone()
    m(spikes(d,b,e), spikes(b,o))
    batch = np.random.randint(b)
    underrun = unbounded[batch] < host.wmin
    assert torch.all(m.W[unbounded < 0] == 0)
    assert torch.all(m.W[batch][underrun] <= host.wmax[underrun])

def test_Abbott_potentiates_on_post(construct):
    m, host, constructor = construct
    b,e,o = constructor[2:5]
    d = host.delaymap.shape[0]
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
    expected = torch.clamp(expected, max=1)
    m(Xpre, Xpost)
    assert torch.equal(m.W, expected)

def test_Abbott_depresses_on_pre(construct):
    m, host, constructor = construct
    b,e,o = constructor[2:5]
    d = host.delaymap.shape[0]
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
        pre, post = np.ix_(range(2,5), range(4,10)) # Inh -> A2.silent
        delay, A_d = 1, 0.4
        npre, npost = 3, 6
    Xpre[delay, :, pre] = 1
    expected = m.W.clone()
    dW = m.xbar_post[:, post] * A_d
    expected[:, pre, post] -= dW.expand(b,npre,npost)
    expected = torch.clamp(expected, 0)
    m(Xpre, Xpost)
    assert torch.equal(m.W, expected)

def test_Abbott_filters_Xpre_with_tau_p(construct, spikes):
    m, host, constructor = construct
    proj, conf, b,e,o, dt = constructor
    conf.tau_p = np.random.rand()
    m = ce.Abbott(*constructor)
    m.reset(host)
    d = host.delaymap.shape[0]
    Xpre, Xpost = spikes(d,b,e), spikes(b,o)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    alpha = util.decayconst(conf.tau_p, dt)
    expected = util.expfilt(Xpre, m.xbar_pre, alpha)
    m(Xpre, Xpost)
    assert torch.allclose(m.xbar_pre, expected)

def test_Abbott_filters_Xpost_with_tau_d(construct, spikes):
    m, host, constructor = construct
    proj, conf, b,e,o, dt = constructor
    conf.tau_d = np.random.rand()
    m = ce.Abbott(*constructor)
    m.reset(host)
    d = host.delaymap.shape[0]
    Xpre, Xpost = spikes(d,b,e), spikes(b,o)
    m.xbar_post = torch.rand_like(m.xbar_post)
    alpha = util.decayconst(conf.tau_d, dt)
    expected = util.expfilt(Xpost, m.xbar_post, alpha)
    m(Xpre, Xpost)
    assert torch.allclose(m.xbar_post, expected)
