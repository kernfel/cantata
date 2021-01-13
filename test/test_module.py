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
    assert len(state) == 10
    assert 't' in state and state.t == 0
    assert 'mem' in state and torch.equal(state.mem, bN0)
    assert 'out' in state and torch.equal(state.out, bN0)
    assert 'w_p' in state and torch.equal(state.out, bN0)
    assert 'x_bar' in state and torch.equal(state.x_bar, bN0)
    assert 'u_pot' in state and torch.equal(state.u_pot, bN0)
    assert 'u_dep' in state and torch.equal(state.u_dep, bN0)
    assert 'w_stdp' in state and torch.equal(state.w_stdp, bNN1)

    assert 'syn' in state and torch.equal(state.out, bN0)
    assert 'poisson' in state and torch.equal(state.out, bN0)

def test_initialise_epoch_state(model_1):
    m = Module()
    x = torch.rand(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    input = torch.einsum('bti,io->bto', x, m.w_in)
    W = torch.einsum('i,io->io', m.w_signs, torch.abs(m.w))
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
    for var in ['out', 'w_p', 'x_bar']:
        assert var in record, var
        assert len(record[var]) == cfg.n_steps, var
        i = np.random.randint(cfg.n_steps)
        assert torch.equal(record[var][i], torch.zeros_like(state[var])), var
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
    assert len(record.mem) == cfg.n_steps
    i = np.random.randint(cfg.n_steps)
    assert torch.equal(record.mem[i], torch.zeros_like(state.mem))

def test_initialise_recordings_ignores_duplicates(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, ['mem', 'out', 'mem'])
    for var in ['mem', 'out']:
        assert var in record
        assert len(record[var]) == cfg.n_steps

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

def test_record_state_leaves_past_data_intact(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    epoch = Box()
    record = m.initialise_recordings(state, epoch, ['mem'])
    copy = Box()
    t = state.t = 4 + np.random.randint(cfg.n_steps-6)
    past = np.random.randint(t)
    for var in ('out', 'w_p', 'x_bar', 'mem'):
        torch.nn.init.uniform_(state[var])
        torch.nn.init.uniform_(record[var][past])
        copy[var] = torch.clone(record[var][past])
    m.record_state(state, record)
    for var in ('out', 'w_p', 'x_bar', 'mem'):
        assert torch.equal(record[var][past], copy[var])
        assert torch.count_nonzero(record[var][-1]) == 0

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
    record.out[0][0,0] = 1 # presynaptic spike in Exc1[0] at t=0
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
    record.x_bar[0][0,0] = x_bar # Presynaptic activity trace in Exc1[0]
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
    record.x_bar[0][0,0] = x_bar # Presynaptic activity trace in Exc1[0]
    record.out[0][0,0] = 1 # presynaptic spike in Exc1[0] at t=0
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

def test_get_synaptic_current_nospikes_nocurrent(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    record = m.initialise_recordings(state, epoch)
    state.t = 10
    for i in range(state.t):
        torch.nn.init.ones_(record.out[i])
    record.out[0][:,2:5] = 0
    record.out[5][:,0:2] = 0
    record.out[10][:,2:5] = 0
    currents = m.get_synaptic_current(state, epoch, record)
    assert torch.allclose(currents, torch.zeros_like(currents))

def test_get_synaptic_current_uses_correct_delays(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    record = m.initialise_recordings(state, epoch)
    b = int(np.random.rand() * cfg.batch_size)
    # record.out: (t,batch,pre)
    record.out[0][b,2] = 1 # i->e, delay 10
    record.out[5][b,0] = 1 # e->e and e->i, delay 5
    record.out[10][b,3] = 1 # i->i, delay 0
    state.t = 10
    currents = m.get_synaptic_current(state, epoch, record)
    expected = torch.zeros_like(currents)
    for exc in range(2):
        # epoch.W: (pre,post)
        expected[b,exc] = epoch.W[2,exc] + epoch.W[0,exc]
    for inh in range(2,5):
        expected[b,inh] = epoch.W[0,inh] + epoch.W[3,inh]
    assert torch.allclose(currents, expected)

def test_get_synaptic_current_applies_STP(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    record = m.initialise_recordings(state, epoch)
    b = int(np.random.rand() * cfg.batch_size)
    # record.out: (t;batch,pre)
    record.out[0][b,2] = 1 # i->e, delay 10
    record.out[5][b,0] = 1 # e->e and e->i, delay 5
    record.out[10][b,3] = 1 # i->i, delay 0
    state.t = 10
    # record.w_p: (t;batch,pre)
    p0 = record.w_p[0][b,2] = 2*np.random.rand()-1
    p5 = record.w_p[5][b,0] = 2*np.random.rand()-1
    p10 = record.w_p[10][b,3] = 2*np.random.rand()-1
    currents = m.get_synaptic_current(state, epoch, record)
    expected = torch.zeros_like(currents)
    for exc in range(2):
        # epoch.W: (pre,post)
        expected[b,exc] = epoch.W[2,exc]*(p0+1) + epoch.W[0,exc]*(p5+1)
    for inh in range(2,5):
        expected[b,inh] = epoch.W[0,inh]*(p5+1) + epoch.W[3,inh]*(p10+1)
    assert torch.allclose(currents, expected)

def test_get_synaptic_current_applies_STDP(model_1):
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    record = m.initialise_recordings(state, epoch)
    b = int(np.random.rand() * cfg.batch_size)
    # record.out: (t;batch,pre)
    record.out[0][b,2] = 1 # i->e, delay 10
    record.out[5][b,0] = 1 # e->e and e->i, delay 5
    record.out[10][b,3] = 1 # i->i, delay 0
    state.t = 10
    # state.w_stdp: (batch,pre,post)
    state.w_stdp[b,2,0:2] = torch.rand(2)*2
    state.w_stdp[b,0,0:5] = torch.rand(5)*2
    state.w_stdp[b,3,2:5] = torch.rand(3)*2
    currents = m.get_synaptic_current(state, epoch, record)
    expected = torch.zeros_like(currents)
    for exc in range(2):
        # epoch.W: (pre,post)
        expected[b,exc] = epoch.W[2,exc]*state.w_stdp[b,2,exc]\
                        + epoch.W[0,exc]*state.w_stdp[b,0,exc]
    for inh in range(2,5):
        expected[b,inh] = epoch.W[0,inh]*state.w_stdp[b,0,inh]\
                        + epoch.W[3,inh]*state.w_stdp[b,3,inh]
    assert torch.allclose(currents, expected)

def integrate_vm_decay(n_iter):
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    record = m.initialise_recordings(state, epoch)
    torch.nn.init.normal_(state.mem)
    expected = torch.clone(state.mem)
    for _ in range(n_iter):
        m.integrate(state, epoch, record)
        expected *= m.alpha_mem
    return state, expected
def test_integrate_vm_decay(model_1):
    state, expected = integrate_vm_decay(np.random.randint(1,11))
    assert torch.allclose(state.mem, expected)
def test_integrate_vm_decay_noisy(model_1_noisy):
    state, expected = integrate_vm_decay(1)
    assert torch.allclose(state.mem, expected + state.poisson)

def integrate_resets_spikes():
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    record = m.initialise_recordings(state, epoch)
    state.out[torch.rand_like(state.out) > 0.5] = 1
    m.integrate(state, epoch, record)
    return state
def test_integrate_resets_spikes(model_1):
    state = integrate_resets_spikes()
    assert torch.allclose(state.mem, -state.out)
def test_integrate_resets_spikes_noisy(model_1_noisy):
    state = integrate_resets_spikes()
    assert torch.allclose(state.mem, state.poisson - state.out)

def integrate_adds_synaptic_current():
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    record = m.initialise_recordings(state, epoch)
    record.out[0][torch.rand_like(state.out) > 0.5] = 1
    expected = m.get_synaptic_current(state, epoch, record)
    m.integrate(state, epoch, record)
    return state, expected
def test_integrate_adds_synaptic_current(model_1):
    state, expected = integrate_adds_synaptic_current()
    assert torch.allclose(state.mem, expected)
def test_integrate_adds_synaptic_current_noisy(model_1_noisy):
    state, expected = integrate_adds_synaptic_current()
    assert torch.allclose(state.mem, expected + state.poisson)

def integrate_adds_input():
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.randn(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    record = m.initialise_recordings(state, epoch)
    m.integrate(state, epoch, record)
    return state, epoch
def test_integrate_adds_input(model_1):
    state, epoch = integrate_adds_input()
    assert torch.allclose(state.mem, epoch.input[:,0])
def test_integrate_adds_input_noisy(model_1_noisy):
    state, epoch = integrate_adds_input()
    assert torch.allclose(state.mem, epoch.input[:,0] + state.poisson)

def test_integrate_only_touches_state(model_1_noisy):
    # This also tests integrate's subroutines for the same purpose.
    m = Module()

    state = m.initialise_dynamic_state()
    state.t = np.random.randint(0,cfg.n_steps-1)
    state_clone = Box()
    state.out[torch.rand_like(state.out) > 0.5] = 1
    state_clone.out = torch.clone(state.out)
    torch.nn.init.uniform_(state.w_stdp, 0, 2)
    state_clone.w_stdp = torch.clone(state.w_stdp)
    for key in ['w_p', 'x_bar', 'u_pot', 'u_dep', 'syn', 'mem']:
        torch.nn.init.normal_(state[key])
        state_clone[key] = torch.clone(state[key])

    inputs = torch.randn(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    epoch_clone = Box()
    for key in epoch:
        epoch_clone[key] = torch.clone(epoch[key])

    record = m.initialise_recordings(state, epoch, ['mem'])
    record_clone = record.copy()
    for t in range(cfg.n_steps):
        record.out[t][torch.rand_like(record.out[t]) > 0.5] = 1
        torch.nn.init.uniform_(record.w_p[t], -1,1)
        torch.nn.init.uniform_(record.x_bar[t])
        torch.nn.init.uniform_(record.mem[t], -1,1)
        for key in ['out', 'w_p', 'x_bar', 'mem']:
            record_clone[key][t] = torch.clone(record[key][t])

    m.integrate(state, epoch, record)

    # These should change:
    for key in state_clone:
        if key != 'out':
            assert not torch.allclose(state_clone[key], state[key]), key

    # These should not:
    assert torch.equal(state_clone.out, state.out)
    for key in epoch_clone:
        assert torch.equal(epoch_clone[key], epoch[key]), key
    for key in record._state_records:
        for t in range(cfg.n_steps):
            assert torch.equal(record_clone[key][t], record[key][t]), key

def test_trained_parameter_setup(model_1):
    m = Module()
    params = {k:v for k,v in m.named_parameters()}
    for name in ['w_in', 'w', 'w_out']:
        assert name in params
        assert params[name].requires_grad

def test_forward_prepares_backward(model_1):
    for pop in cfg.model.populations.values():
        for target in pop.targets.values():
            target.delay = 0
    m = Module()
    inputs = torch.randn(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    record = m.forward(inputs)
    record.readout.sum().backward()
    with torch.no_grad(): # circumvent a bug in torch
        for name, p in m.named_parameters():
            z = torch.zeros_like(p)
            assert not torch.equal(p.grad, z), name

def test_poisson_input_off_by_default(model_1):
    m = Module()
    assert not m.has_poisson

def test_poisson_input_off_by_missing_values(model_1):
    params = ['poisson_N', 'poisson_rate', 'poisson_weight']
    values = [np.random.randint(1000,size=2),
        np.random.rand(2)*50, np.random.rand(2)]
    # Set one of the three relevant parameters to zero in each population:
    zero = np.random.randint(3,size=2)
    for i,pop in enumerate(cfg.model.populations.values()):
        for j,par in enumerate(params):
            pop[par] = 0 if zero[i]==j else values[j][i]
    m = Module()
    assert not m.has_poisson

def test_poisson_input_scales_with_weight(model_1):
    cfg.model.populations.Exc1.poisson_N = 1
    cfg.model.populations.Exc1.poisson_rate = 1/cfg.time_step
    cfg.model.populations.Exc1.poisson_weight = w = np.random.rand() * 10
    m = Module()
    p = m.get_poisson_background()
    expected = torch.tensor([w,w,0,0,0], **cfg.tspec)
    assert torch.all(p == expected)

def test_poisson_input_is_binomial(model_2):
    r = torch.rand(3)
    n = torch.rand(3)*1000 + 500
    cfg.model.populations.Exc2.n = 200
    cfg.model.populations.Exc1.poisson_N = n[0]
    cfg.model.populations.Inh1.poisson_N = n[1]
    cfg.model.populations.Exc2.poisson_N = n[2]
    cfg.model.populations.Exc1.poisson_rate = r[0]/cfg.time_step
    cfg.model.populations.Inh1.poisson_rate = r[1]/cfg.time_step
    cfg.model.populations.Exc2.poisson_rate = r[2]/cfg.time_step
    cfg.model.populations.Exc1.poisson_weight = 1
    cfg.model.populations.Inh1.poisson_weight = 1
    cfg.model.populations.Exc2.poisson_weight = 1
    m = Module()
    p = m.get_poisson_background()

    means = torch.tensor([
        torch.mean(p[:,:150]),
        torch.mean(p[:,150:250]),
        torch.mean(p[:,250:])
    ]).squeeze()
    expected = r*n
    assert torch.all(expected * 0.95 < means)
    assert torch.all(expected * 1.05 > means)

    variances = torch.tensor([
        torch.var(p[:,:150]),
        torch.var(p[:,150:250]),
        torch.var(p[:,250:])
    ])
    expected = n*r*(1-r)
    # Not the most stringent test, because variance varies a lot:
    assert torch.mean(torch.abs(expected-variances) / (expected+variances)) < 0.1

def test_integrate_adds_poisson_input(model_1_noisy):
    m = Module()
    state = m.initialise_dynamic_state()
    inputs = torch.zeros(cfg.batch_size, cfg.n_steps, cfg.n_inputs, **cfg.tspec)
    epoch = m.initialise_epoch_state(inputs)
    record = m.initialise_recordings(state, epoch)
    m.integrate(state, epoch, record)
    assert m.has_poisson
    assert torch.count_nonzero(state.poisson) > 0, 'No noise (could be spurious)'
    assert torch.allclose(state.mem, state.poisson)
