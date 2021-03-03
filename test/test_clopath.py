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

def test_Clopath_can_change_device(constructor):
    m = ce.Clopath(*constructor)
    host, _, batch_size, nPre, nPost = constructor[1:6]
    d = host.delaymap.shape[0]
    for device in [torch.device('cuda'), torch.device('cpu')]:
        Xd = spikes(d, batch_size, nPre).to(device)
        Xpost = spikes(batch_size, nPost).to(device)
        Vpost = torch.rand(batch_size, nPost).to(device)
        m.to(device)
        host.to(device)
        W = m(Xd, Xpost, Vpost)
        assert W.device == Xd.device

def test_Clopath_does_not_move_host(constructor):
    host = constructor[1]
    m = ce.Clopath(*constructor)
    dev = host.wmax.device
    for device in [torch.device('cuda'), torch.device('cpu')]:
        m.to(device)
        assert host.wmax.device == dev

def test_Clopath_state(constructor, module_tests):
    host, _, b,e,o = constructor[1:6]
    d = host.delaymap.shape[0]
    module_tests.check_state(
        ce.Clopath(*constructor),
        ['xbar_pre', 'u_dep', 'u_pot', 'W'],
        [(d,b,e),    (b,o), (b,o), (b,e,o)]
    )

def test_Clopath_reset_clears_filters(constructor, module_tests):
    module_tests.check_reset_clears(
        ce.Clopath(*constructor),
        'xbar_pre', 'u_dep', 'u_pot',
        inputs = (constructor[1].wmax,)
    )

def test_Clopath_reset_resets_W_to_arg(constructor):
    m = ce.Clopath(*constructor)
    m.W = torch.rand_like(m.W)
    expected = torch.rand_like(constructor[1].wmax)
    m.reset(expected.clone())
    batch = np.random.randint(constructor[3])
    assert torch.equal(m.W[batch], expected)

def test_Clopath_deactivates_when_not_required(constructor):
    # Note that STDP_frac is not within the STDP model's responsibility.
    (_, params) = constructor[0]
    nPost = constructor[5]
    for p in params:
        p.A_p = p.A_d = 0.
    m = ce.Clopath(*constructor)
    assert not m.active

def test_Clopath_returns_premodification_W(constructor):
    host, _, b,e,o = constructor[1:6]
    d = host.delaymap.shape[0]
    m = ce.Clopath(*constructor)
    expected = torch.rand_like(m.W)
    m.W = expected.clone()
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.u_dep = torch.rand_like(m.u_dep)
    m.u_pot = torch.rand_like(m.u_pot)
    W = m(spikes(d,b,e), spikes(b,o), torch.rand(b,o))
    assert torch.equal(W, expected)
    assert not torch.equal(m.W, expected)

def test_Clopath_bounds_W_on_forward(constructor):
    host, _, b,e,o = constructor[1:6]
    d = host.delaymap.shape[0]
    m = ce.Clopath(*constructor)
    unbounded = torch.randn_like(m.W)
    m.W = unbounded.clone()
    m(spikes(d,b,e), spikes(b,o), torch.rand(b,o))
    batch = np.random.randint(b)
    surpass = unbounded[batch] > host.wmax
    assert torch.all(m.W[unbounded < 0] == 0)
    assert torch.all(m.W[batch][surpass] == host.wmax[surpass])

def test_Clopath_potentiates_on_post(constructor):
    host, _, b,e,o = constructor[1:6]
    d = host.delaymap.shape[0]
    m = ce.Clopath(*constructor)
    host.wmax = torch.ones_like(host.wmax)
    m.W = torch.zeros_like(m.W)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.u_dep = torch.randn_like(m.u_dep)
    m.u_pot = torch.randn_like(m.u_pot)
    Xpre, Xpost, Vpost = torch.zeros(d,b,e), torch.zeros(b,o), torch.rand(b,o)
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
    xbar_pre = m.xbar_pre[delay, :, pre]
    upot_rect = torch.nn.functional.relu(m.u_pot[:, post])
    dW = xbar_pre * upot_rect * A_p
    expected[:, pre, post] += dW
    m(Xpre, Xpost, Vpost)
    assert torch.allclose(m.W, expected)

def test_Clopath_depresses_on_pre(constructor):
    host, _, b,e,o = constructor[1:6]
    d = host.delaymap.shape[0]
    m = ce.Clopath(*constructor)
    host.wmax = torch.ones_like(host.wmax)
    m.W = torch.ones_like(m.W)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.u_dep = torch.randn_like(m.u_dep)
    m.u_pot = torch.randn_like(m.u_pot)
    Xpre, Xpost, Vpost = torch.zeros(d,b,e), torch.zeros(b,o), torch.rand(b,o)
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
    dW = torch.nn.functional.relu(m.u_dep[:, post]) * A_d
    expected[:, pre, post] -= dW
    m(Xpre, Xpost, Vpost)
    assert torch.allclose(m.W, expected)

def test_Clopath_filters_Xpre_with_tau_x(constructor):
    _, host, conf, b,e,o, dt = constructor
    conf.tau_x = np.random.rand()
    d = host.delaymap.shape[0]
    m = ce.Clopath(*constructor)
    Xpre, Xpost, Vpost = spikes(d,b,e), spikes(b,o), torch.rand(b,o)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    alpha = util.decayconst(conf.tau_x, dt)
    expected = util.expfilt(Xpre, m.xbar_pre, alpha)
    m(Xpre, Xpost, Vpost)
    assert torch.allclose(m.xbar_pre, expected)

def test_Clopath_filters_udep_with_tau_d(constructor):
    _, host, conf, b,e,o, dt = constructor
    conf.tau_d = np.random.rand()
    d = host.delaymap.shape[0]
    m = ce.Clopath(*constructor)
    Xpre, Xpost, Vpost = spikes(d,b,e), spikes(b,o), torch.rand(b,o)
    m.u_dep = torch.rand_like(m.u_dep)
    alpha = util.decayconst(conf.tau_d, dt)
    expected = util.expfilt(Vpost, m.u_dep, alpha)
    m(Xpre, Xpost, Vpost)
    assert torch.allclose(m.u_dep, expected)

def test_Clopath_filters_upot_with_tau_p(constructor):
    _, host, conf, b,e,o, dt = constructor
    conf.tau_p = np.random.rand()
    d = host.delaymap.shape[0]
    m = ce.Clopath(*constructor)
    Xpre, Xpost, Vpost = spikes(d,b,e), spikes(b,o), torch.rand(b,o)
    m.u_pot = torch.rand_like(m.u_pot)
    alpha = util.decayconst(conf.tau_p, dt)
    expected = util.expfilt(Vpost, m.u_pot, alpha)
    m(Xpre, Xpost, Vpost)
    assert torch.allclose(m.u_pot, expected)
