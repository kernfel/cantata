import pytest
from cantata import util, init
import cantata.elements as ce
import torch
import numpy as np


class Host(torch.nn.Module):
    def __init__(self, delaymap, wmin, wmax):
        super(Host, self).__init__()
        self.register_buffer('delaymap', delaymap)
        self.register_buffer('wmin', wmin)
        self.register_buffer('wmax', wmax)
        W = torch.rand_like(wmin) * 2*wmax - wmax
        W[W.abs() < wmin] = 0
        self.register_buffer('W', W)


@pytest.fixture(params=[True, False], ids=['xarea', 'internal'])
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
    m = ce.Clopath(*ctor)
    delaymap = init.get_delaymap(projections, dt, conf_pre, conf_post)
    wmax = torch.rand(nPre, nPost)
    wmin = torch.rand(nPre, nPost) * wmax
    host = Host(delaymap, wmin, wmax)
    m.reset(host)
    return m, host, ctor


def test_Clopath_does_not_modify_children(module_tests, construct, spikes):
    m, host, constructor = construct
    b, e, o = constructor[2:5]
    d = host.delaymap.shape[0]
    Xd = spikes(d, b, e)
    Xpost = spikes(b, o)
    Vpost = torch.rand(b, o)
    module_tests.check_no_child_modification(m, Xd, Xpost, Vpost)


def test_Clopath_can_change_device(construct, spikes):
    m, host, constructor = construct
    b, e, o = constructor[2:5]
    d = host.delaymap.shape[0]
    for device in [torch.device('cuda'), torch.device('cpu')]:
        Xd = spikes(d, b, e).to(device)
        Xpost = spikes(b, o).to(device)
        Vpost = torch.rand(b, o).to(device)
        m.to(device)
        host.to(device)
        W = m(Xd, Xpost, Vpost)
        assert W.device == Xd.device


def test_Clopath_does_not_move_host(construct):
    m, host, constructor = construct
    dev = host.wmax.device
    for device in [torch.device('cuda'), torch.device('cpu')]:
        m.to(device)
        assert host.wmax.device == dev


def test_Clopath_state(construct, module_tests):
    m, host, constructor = construct
    b, e, o = constructor[2:5]
    d = host.delaymap.shape[0]
    module_tests.check_state(
        m,
        ['xbar_pre', 'u_dep', 'u_pot', 'W'],
        [(d, b, e),    (b, o), (b, o), (b, e, o)]
    )


def test_Clopath_reset_clears_filters(construct, module_tests):
    m, host, constructor = construct
    module_tests.check_reset_clears(
        m,
        'xbar_pre', 'u_dep', 'u_pot',
        inputs=(host,)
    )


def test_Clopath_reset_resets_W_to_arg(construct):
    m, host, constructor = construct
    m.W = torch.rand_like(m.W)
    expected = host.W
    m.reset(host)
    batch = np.random.randint(len(m.W))
    assert torch.equal(m.W[batch], expected)


def test_Clopath_deactivates_when_not_required(construct):
    # Note that STDP_frac is not within the STDP model's responsibility.
    *_, constructor = construct
    b, e, o = constructor[2:5]
    (_, params) = constructor[0]
    for p in params:
        p.A_p = p.A_d = 0.
    m = ce.Clopath(*constructor)
    assert not m.active


def test_Clopath_returns_premodification_W(construct, spikes):
    m, host, constructor = construct
    b, e, o = constructor[2:5]
    d = host.delaymap.shape[0]
    expected = torch.rand_like(m.W)
    m.W = expected.clone()
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.u_dep = torch.rand_like(m.u_dep)
    m.u_pot = torch.rand_like(m.u_pot)
    W = m(spikes(d, b, e), spikes(b, o), torch.rand(b, o))
    assert torch.equal(W, expected)
    assert not torch.equal(m.W, expected)


def test_Clopath_maintains_W_sign(construct, spikes):
    m, host, constructor = construct
    b, e, o = constructor[2:5]
    d = host.delaymap.shape[0]
    initial = torch.randn_like(host.W)
    initial[initial.abs() > .5] = 0
    host.W = initial.clone()
    m.W = torch.randn_like(m.W)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.u_dep = torch.rand_like(m.u_dep)
    m.u_pot = torch.rand_like(m.u_pot)
    m(spikes(d, b, e), spikes(b, o), torch.rand(b, o))
    # There may be some lower-bounded m.W==0, initial!=0, so:
    assert torch.all(m.W * initial >= 0)


def test_Clopath_does_not_resurrect_dead_host_weights(construct, spikes):
    m, host, constructor = construct
    b, e, o = constructor[2:5]
    d = host.delaymap.shape[0]
    initial = torch.randn_like(host.W)
    initial[initial.abs() > .5] = 0
    host.W = initial.clone()
    m.W = torch.randn_like(m.W)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.u_dep = torch.rand_like(m.u_dep)
    m.u_pot = torch.rand_like(m.u_pot)
    m(spikes(d, b, e), spikes(b, o), torch.rand(b, o))
    assert torch.all(m.W[:, initial == 0] == 0)


def test_Clopath_ubounds_W_on_forward(construct, spikes):
    m, host, constructor = construct
    b, e, o = constructor[2:5]
    d = host.delaymap.shape[0]
    m.W = torch.randn_like(m.W)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.u_dep = torch.rand_like(m.u_dep)
    m.u_pot = torch.rand_like(m.u_pot)
    m(spikes(d, b, e), spikes(b, o), torch.rand(b, o))
    assert torch.all(m.W.abs() <= host.wmax.expand_as(m.W))


def test_Clopath_potentiates_on_post(construct):
    m, host, constructor = construct
    b, e, o = constructor[2:5]
    d = host.delaymap.shape[0]
    host.wmax = torch.ones_like(host.wmax)
    m.W = torch.zeros_like(m.W)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.u_dep = torch.randn_like(m.u_dep)
    m.u_pot = torch.randn_like(m.u_pot)
    Xpre, Xpost, Vpost = torch.zeros(
        d, b, e), torch.zeros(b, o), torch.rand(b, o)
    if o == 5:  # internal
        pre, post = np.ix_(range(2), range(2, 5))  # Exc -> Inh
        delay, A_p = 1, 0.1
    else:  # xarea
        pre, post = np.ix_(range(2), range(4))  # Exc -> A2.deadend
        delay, A_p = 0, 0.3
    Xpost[:, post] = 1
    expected = m.W.clone()
    xbar_pre = m.xbar_pre[delay, :, pre]
    upot_rect = torch.nn.functional.relu(m.u_pot[:, post])
    dW = xbar_pre * upot_rect * A_p
    expected[:, pre, post] += dW
    expected = torch.clamp(expected, max=1) * host.W.sign()
    m(Xpre, Xpost, Vpost)
    assert torch.allclose(m.W, expected)


def test_Clopath_depresses_on_pre(construct):
    m, host, constructor = construct
    b, e, o = constructor[2:5]
    d = host.delaymap.shape[0]
    host.wmax = torch.ones_like(host.wmax)
    m.W = torch.ones_like(m.W)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    m.u_dep = torch.randn_like(m.u_dep)
    m.u_pot = torch.randn_like(m.u_pot)
    Xpre, Xpost, Vpost = torch.zeros(
        d, b, e), torch.zeros(b, o), torch.rand(b, o)
    if o == 5:  # internal
        pre, post = np.ix_(range(2), range(2, 5))  # Exc -> Inh
        delay, A_d = 1, 0.2
    else:  # xarea
        pre, post = np.ix_(range(2, 5), range(4, 10))  # Inh -> A2.silent
        delay, A_d = 1, 0.4
    Xpre[delay, :, pre] = 1
    expected = m.W.clone()
    dW = torch.nn.functional.relu(m.u_dep[:, post]) * A_d
    expected[:, pre, post] -= dW
    expected = torch.clamp(expected, 0) * host.W.sign()
    m(Xpre, Xpost, Vpost)
    assert torch.allclose(m.W, expected)


def test_Clopath_filters_Xpre_with_tau_x(construct, spikes):
    m, host, constructor = construct
    proj, conf, b, e, o, dt = constructor
    conf.tau_x = np.random.rand()
    m = ce.Clopath(*constructor)
    m.reset(host)
    d = host.delaymap.shape[0]
    Xpre, Xpost, Vpost = spikes(d, b, e), spikes(b, o), torch.rand(b, o)
    m.xbar_pre = torch.rand_like(m.xbar_pre)
    alpha = util.decayconst(conf.tau_x, dt)
    expected = util.expfilt(Xpre, m.xbar_pre, alpha)
    m(Xpre, Xpost, Vpost)
    assert torch.allclose(m.xbar_pre, expected)


def test_Clopath_filters_udep_with_tau_d(construct, spikes):
    m, host, constructor = construct
    proj, conf, b, e, o, dt = constructor
    conf.tau_d = np.random.rand()
    m = ce.Clopath(*constructor)
    m.reset(host)
    d = host.delaymap.shape[0]
    Xpre, Xpost, Vpost = spikes(d, b, e), spikes(b, o), torch.rand(b, o)
    m.u_dep = torch.rand_like(m.u_dep)
    alpha = util.decayconst(conf.tau_d, dt)
    expected = util.expfilt(Vpost, m.u_dep, alpha)
    m(Xpre, Xpost, Vpost)
    assert torch.allclose(m.u_dep, expected)


def test_Clopath_filters_upot_with_tau_p(construct, spikes):
    m, host, constructor = construct
    proj, conf, b, e, o, dt = constructor
    conf.tau_p = np.random.rand()
    m = ce.Clopath(*constructor)
    m.reset(host)
    d = host.delaymap.shape[0]
    Xpre, Xpost, Vpost = spikes(d, b, e), spikes(b, o), torch.rand(b, o)
    m.u_pot = torch.rand_like(m.u_pot)
    alpha = util.decayconst(conf.tau_p, dt)
    expected = util.expfilt(Vpost, m.u_pot, alpha)
    m(Xpre, Xpost, Vpost)
    assert torch.allclose(m.u_pot, expected)
