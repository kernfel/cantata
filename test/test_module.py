import pytest
import cantata
from cantata import cfg, Module
from cantata.module import SurrGradSpike
import torch
import numpy as np
from box import Box

def test_initialise_dynamic_state(model_1):
    # This is mostly a check that the shapes are as advertised.
    m = Module()
    N = cantata.init.get_N()
    bN0 = torch.zeros((cfg.batch_size, N), **cfg.tspec)
    bNN1 = torch.ones((cfg.batch_size, N, N), **cfg.tspec)
    state = m.initialise_dynamic_state()
    assert len(state) == 9
    assert 't' in state and state.t == 0
    assert 'mem' in state and torch.equal(state.mem, bN0)
    assert 'out' in state and torch.equal(state.out, bN0)
    assert 'w_p' in state and torch.equal(state.out, bN0)
    assert 'syn' in state and torch.equal(state.out, bN0)
    assert 'x_bar' in state and torch.equal(state.x_bar, bN0)
    assert 'u_pot' in state and torch.equal(state.u_pot, bN0)
    assert 'u_dep' in state and torch.equal(state.u_dep, bN0)
    assert 'w_stdp' in state and torch.equal(state.w_stdp, bNN1)

def test_initialise_epoch_state(model_1):
    m = Module()
    x = torch.rand(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    input = torch.einsum('bti,io->bto', x, m.w_in)
    W = m.dmap * torch.einsum('i,io->io', m.w_signs, torch.abs(m.w))
    epoch = m.initialise_epoch_state(x)
    assert len(epoch) == 3
    assert 'input' in epoch and torch.allclose(epoch.input, input)
    assert 'W' in epoch and torch.allclose(epoch.W, W)
    assert 'p_depr_mask' in epoch and torch.equal(epoch.p_depr_mask, m.p<0)

def test_initialise_recordings_adds_minimal_set(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, [])
    assert 'out' in record
    assert torch.equal(record.out,
        torch.zeros((cfg.n_steps,) + state.out.shape, **cfg.tspec))
    assert 'w_p' in record
    assert torch.equal(record.w_p,
        torch.zeros((cfg.n_steps,) + state.w_p.shape, **cfg.tspec))
    assert 'x_bar' in record
    assert torch.equal(record.x_bar,
        torch.zeros((cfg.n_steps,) + state.x_bar.shape, **cfg.tspec))
    assert '_state_records' in record
    assert '_epoch_records' in record
    assert len(record) == 5

def test_initialise_recordings_is_minimal_by_default(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    expected = m.initialise_recordings(state, epoch, [])
    received = m.initialise_recordings(state, epoch)
    assert expected.keys() == received.keys()

def test_initialise_recordings_adds_requested_dynamic_vars(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, ['mem'])
    assert 'mem' in record
    assert torch.equal(record.mem,
        torch.zeros((cfg.n_steps,) + state.mem.shape, **cfg.tspec))

def test_initialise_recordings_sets_list_correctly(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, ['mem', 'shouldbeignored'])
    for var in ('out', 'w_p', 'x_bar', 'mem'):
        assert var in record._state_records
    assert 'shouldbeignored' not in record._state_records
    assert len(record._state_records) == 4

def test_initialise_recordings_adds_copy_of_epoch_members(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    x = torch.rand(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(x)
    record = m.initialise_recordings(state, epoch, ['W'])
    assert 'W' in record
    assert 'W' in record._epoch_records
    assert torch.equal(record.W, epoch.W)
    assert id(record.W) != id(epoch.W)

def test_mark_spikes_triggers_all_above_threshold(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    torch.nn.init.uniform_(state.mem, -2.0, 4.0)
    state.mem[0,0] = 1.0 # exactly
    expected = (state.mem > 1.0) * 1.0 # float-ify
    m.mark_spikes(state)
    assert torch.equal(state.out, expected)
    assert expected[0,0] == False # at threshold should not fire

def test_mark_spikes_returns_detached_replica(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    torch.nn.init.uniform_(state.mem, -2.0, 4.0)
    detached = m.mark_spikes(state)
    assert torch.equal(state.out, detached)
    assert detached.requires_grad == False

def test_mark_spikes_applies_surrogate_gradient(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    torch.nn.init.uniform_(state.mem, -2.0, 4.0)
    state.mem.requires_grad_()
    m.mark_spikes(state)
    assert type(state.out.grad_fn) == SurrGradSpike._backward_cls

def test_record_state_inserts_at_state_time(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, ['mem'])
    t = state.t = 4
    torch.nn.init.uniform_(state.out)
    torch.nn.init.uniform_(state.w_p)
    torch.nn.init.uniform_(state.x_bar)
    torch.nn.init.uniform_(state.mem)
    m.record_state(state, record)
    assert torch.equal(state.out, record.out[t])
    assert torch.equal(state.w_p, record.w_p[t])
    assert torch.equal(state.x_bar, record.x_bar[t])
    assert torch.equal(state.mem, record.mem[t])

def test_record_state_leaves_nonpresent_data_intact(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, ['mem'])
    copy = Box()
    for var in ('out', 'w_p', 'x_bar', 'mem'):
        torch.nn.init.uniform_(record[var])
        copy[var] = torch.clone(record[var])
    t = state.t = 4
    m.record_state(state, record)
    for var in ('out', 'w_p', 'x_bar', 'mem'):
        assert torch.equal(record[var][:t-1], copy[var][:t-1])
        assert torch.equal(record[var][t+1:], copy[var][t+1:])

def test_record_state_does_not_affect_epoch_recordings(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    x = torch.rand(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(x)
    record = m.initialise_recordings(state, epoch, ['W'])
    state.t = int(np.random.rand() * 10)
    m.record_state(state, record)
    assert torch.equal(epoch.W, record.W)

def test_compute_STDP_nospikes_nochange(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, [])
    torch.nn.init.uniform_(state.w_stdp, cfg.model.stdp_wmin, cfg.model.stdp_wmax)
    expected = torch.clone(state.w_stdp)
    m.compute_STDP(state, record)
    assert torch.allclose(state.w_stdp, expected)

def test_compute_STDP_depression(model_1):
    # Depression triggers on presynaptic spikes as
    # X[pre] * A_d * relu(u_dep[post])
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, [])
    u = np.random.rand() + 1
    uneg = np.random.rand() - 2
    record.out[0,0,0] = 1 # presynaptic spike in Exc1[0] at t=0
    # Postsynaptic activity traces in Exc1[0,1] and Inh1[0,1]
    state.u_dep[0,:4] = torch.tensor([u, uneg, u, uneg])
    # Expect no depression in e->e or negative u_dep
    expected = torch.tensor([1, 1, 1 - u*0.2, 1], **cfg.tspec)
    state.t = 5 # because e->e and e->i are both delayed by 5 ms
    m.compute_STDP(state, record)
    assert torch.allclose(state.w_stdp[0,0,:4], expected)

def test_compute_STDP_potentiation(model_1):
    # Potentiation triggers on postsynaptic spikes as
    # x_bar[pre] * A_p * X[post] * relu(u_pot[post])
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, [])
    u = np.random.rand() + 1
    uneg = np.random.rand() - 2
    x_bar = np.random.rand() + 1
    record.x_bar[0,0,0] = x_bar # Presynaptic activity trace in Exc1[0]
    # Postsynaptic activity traces in Exc1[0,1] and Inh1[0,1]
    state.u_pot[0,:4] = torch.tensor([u, uneg, u, uneg])
    state.out[0,:4] = 1 # Postsynaptic spikes
    # Expect no potentiation in e->e or negative u_pot
    expected = torch.tensor([1, 1, 1 + u*x_bar*0.1, 1], **cfg.tspec)
    state.t = 5
    m.compute_STDP(state, record)
    assert torch.allclose(state.w_stdp[0,0,:4], expected)

def test_compute_STDP_combined(model_1):
    # Combine the two above...
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, [])
    u_dep = np.random.rand() + 1
    u_pot = np.random.rand() + 1
    x_bar = np.random.rand() + 1
    record.x_bar[0,0,0] = x_bar # Presynaptic activity trace in Exc1[0]
    record.out[0,0,0] = 1 # presynaptic spike in Exc1[0] at t=0
    # Postsynaptic activity traces in Inh1[0,1]
    state.u_pot[0,2:4] = torch.tensor([u_pot*2, u_pot])
    state.u_dep[0,2:4] = torch.tensor([u_dep, u_dep*2])
    state.out[0,2:4] = 1 # Postsynaptic spikes
    expected = torch.tensor([1 + u_pot*x_bar*0.1*2 - u_dep*0.2,
                             1 + u_pot*x_bar*0.1 - 2*u_dep*0.2], **cfg.tspec)
    state.t = 5
    m.compute_STDP(state, record)
    assert torch.allclose(state.w_stdp[0,0,2:4], expected)

def test_STDP_presynaptic_timeconstant():
    cfg.model.tau_x = np.random.rand() * 0.1
    expected = cantata.util.decayconst(cfg.model.tau_x)
    m = Module()
    assert np.allclose(m.alpha_x, expected)

def test_STDP_potentiation_timeconstant():
    cfg.model.tau_p = np.random.rand() * 0.1
    expected = cantata.util.decayconst(cfg.model.tau_p)
    m = Module()
    assert np.allclose(m.alpha_p, expected)

def test_STDP_depression_timeconstant():
    cfg.model.tau_d = np.random.rand() * 0.1
    expected = cantata.util.decayconst(cfg.model.tau_d)
    m = Module()
    assert np.allclose(m.alpha_d, expected)

def test_STDP_xbar_filtered(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, [])
    n_iter = int(np.random.rand()*10) + 1
    torch.nn.init.uniform_(state.x_bar, 0., 5.)
    expected = torch.clone(state.x_bar)
    for _ in range(n_iter):
        state.out[torch.rand_like(state.out) > 0.5] = 1
        m.compute_STDP(state, record)
        expected = expected*m.alpha_x + state.out*(1-m.alpha_x)
    assert torch.allclose(state.x_bar, expected)

def test_STDP_udep_filtered(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, [])
    n_iter = int(np.random.rand()*10) + 1
    torch.nn.init.uniform_(state.u_dep, 0., 5.)
    expected = torch.clone(state.u_dep)
    for _ in range(n_iter):
        torch.nn.init.uniform_(state.mem)
        m.compute_STDP(state, record)
        expected = expected*m.alpha_d + state.mem*(1-m.alpha_d)
    assert torch.allclose(state.u_dep, expected)

def test_STDP_upot_filtered(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, [])
    n_iter = int(np.random.rand()*10) + 1
    torch.nn.init.uniform_(state.u_pot, 0., 5.)
    expected = torch.clone(state.u_pot)
    for _ in range(n_iter):
        torch.nn.init.uniform_(state.mem)
        m.compute_STDP(state, record)
        expected = expected*m.alpha_p + state.mem*(1-m.alpha_p)
    assert torch.allclose(state.u_pot, expected)

def test_STDP_weight_bounds(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, [])
    span = cfg.model.stdp_wmax - cfg.model.stdp_wmin
    lo = cfg.model.stdp_wmin - span
    hi = cfg.model.stdp_wmax + span
    torch.nn.init.uniform_(state.w_stdp, lo, hi)
    m.compute_STDP(state, record)
    assert torch.all(state.w_stdp >= cfg.model.stdp_wmin)
    assert torch.all(state.w_stdp <= cfg.model.stdp_wmax)

def test_compute_STP_nospikes_nochange(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    m.compute_STP(state, epoch)
    assert torch.allclose(state.w_p, torch.zeros_like(state.out, **cfg.tspec))

def test_compute_STP_depression_initial(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    state.out[0,0] = 1
    m.compute_STP(state, epoch)
    expected = torch.zeros_like(state.out, **cfg.tspec)
    expected[0,0] = -0.1
    assert torch.allclose(state.w_p, expected)

def test_compute_STP_depression_is_multiplicative(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    state.out[0,0:2] = 1
    state.w_p[0,0] = -0.5
    state.w_p[0,1] = -1
    m.compute_STP(state, epoch)
    expected = torch.zeros_like(state.out, **cfg.tspec)
    expected[0,0] = -0.5*m.alpha_r - 0.05
    expected[0,1] = -1.0*m.alpha_r # maximal depression already reached
    assert torch.allclose(state.w_p, expected)

def test_compute_STP_nospikes_nochange_facilitation(model_1):
    cfg.model.populations.Exc1.p = 0.1
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    m.compute_STP(state, epoch)
    assert torch.allclose(state.w_p, torch.zeros_like(state.out, **cfg.tspec))

def test_compute_STP_facilitation_initial(model_1):
    cfg.model.populations.Exc1.p = 0.1
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    state.out[0,0] = 1
    m.compute_STP(state, epoch)
    expected = torch.zeros_like(state.out, **cfg.tspec)
    expected[0,0] = 0.1
    assert torch.allclose(state.w_p, expected)

def test_compute_STP_facilitation_is_additive(model_1):
    cfg.model.populations.Exc1.p = 0.1
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    state.out[0,0:2] = 1
    state.w_p[0,0] = 0.5
    state.w_p[0,1] = 2.0
    m.compute_STP(state, epoch)
    expected = torch.zeros_like(state.out, **cfg.tspec)
    expected[0,0] = 0.5*m.alpha_r + 0.1
    expected[0,1] = 2.0*m.alpha_r + 0.1
    assert torch.allclose(state.w_p, expected)

def test_STP_recovery_timeconstant():
    cfg.model.tau_r = np.random.rand() * 0.5
    expected = cantata.util.decayconst(cfg.model.tau_r)
    m = Module()
    assert np.allclose(m.alpha_r, expected)

def test_STP_recovery(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    n_iter = int(np.random.rand()*10) + 1
    torch.nn.init.normal_(state.w_p)
    expected = state.w_p * m.alpha_r ** n_iter
    for _ in range(n_iter):
        m.compute_STP(state, epoch)
    assert torch.allclose(state.w_p, expected)
