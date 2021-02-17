import torch
from torch.nn.functional import relu
import numpy as np
from box import Box
from cantata import util, init, cfg

class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.N = N = init.get_N(True)

        # Network structure
        self.w_signs = init.expand_to_neurons('sign')

        # Weights
        projections = init.build_projections()
        w = init.build_connectivity(projections)
        dmap, delays = init.build_delay_mapping(projections)
        self.w = torch.nn.Parameter(w) # LEARN
        self.dmap = dmap
        self.delays = delays

        w_out = init.build_connectivity(
            init.build_output_projections(),
            (N, cfg.n_outputs)
        )
        self.w_out = torch.nn.Parameter(w_out) # LEARN

        # Short-term plasticity
        self.p = init.expand_to_neurons('p')
        self.alpha_r = util.decayconst(cfg.model.tau_r)

        # STDP
        self.A_p = init.expand_to_synapses('A_p', projections)
        self.A_d = init.expand_to_synapses('A_d', projections)
        self.alpha_x = util.decayconst(cfg.model.tau_x)
        self.alpha_p = util.decayconst(cfg.model.tau_p)
        self.alpha_d = util.decayconst(cfg.model.tau_d)

        # Membrane time constants
        if cfg.model.tau_mem_gamma > 0:
            alpha, beta = torch.tensor([
                cfg.model.tau_mem_gamma,
                cfg.model.tau_mem_gamma/cfg.model.tau_mem
            ], device = cfg.tspec.device)
            G = torch.distributions.gamma.Gamma(alpha, beta)
            self.alpha_mem = util.decayconst(G.sample((N,)))
        else:
            alpha = util.decayconst(cfg.model.tau_mem)
            self.alpha_mem = torch.ones(N, **cfg.tspec) * alpha

        self.alpha_mem_out = util.decayconst(cfg.model.tau_mem_out)
        self.t_refractory = int(max(1,np.round(cfg.model.tau_ref / cfg.time_step)))

        # Noise
        noise_N = init.expand_to_neurons('noise_N').expand(cfg.batch_size, N)
        noise_p = init.expand_to_neurons('noise_rate') * cfg.time_step
        noise_p.clamp_(0,1)
        self.noise_binom = torch.distributions.Binomial(noise_N, noise_p)
        self.noise_weight = init.expand_to_neurons('noise_weight')
        self.has_noise = torch.any(
            (self.noise_weight > 0) * (noise_N[0,:] > 0) * (noise_p > 0))

    def forward(self, inputs, record_vars = []):
        state, epoch, record = self.forward_init(inputs, record_vars)
        self.forward_run(state, epoch, record)
        return self.forward_close(record)

    def forward_init(self, inputs, record_vars):
        state = self.initialise_dynamic_state()
        epoch = self.initialise_epoch_state(inputs)
        record = self.initialise_recordings(state, epoch, record_vars)
        return state, epoch, record

    def forward_run(self, state, epoch, record):
        for state.t in range(cfg.n_steps):
            state.mem += epoch.input[:, state.t]
            self.mark_spikes(state)
            self.record_state(state, record)
            self.integrate(state, epoch, record)

    def forward_close(self, record):
        self.finalise_recordings(record)
        record.readout = self.compute_readout(record)
        return record

    def initialise_dynamic_state(self):
        '''
        Sets up the dynamic state dictionary, that is, the set of dynamic
        variables that affect and are affected by integration.
        @return Box
        '''
        bN = torch.zeros((cfg.batch_size,self.N), **cfg.tspec)
        return Box(dict(
            t = 0,
            # mem (b,N): Membrane potential
            mem = torch.nn.init.uniform_(bN.clone()),
            # out (b,N): Spike raster
            out = bN.clone(),
            # refractory (b,N): Remaining refractory period
            refractory = bN.clone(),
            # w_p (b,N): Short-term plastic weight component, presynaptic
            w_p = bN.clone(),
            # x_bar (b,N): Filtered version of out for STDP, presyn component
            x_bar = bN.clone(),
            # u_pot (b,N): Filtered version of mem for STDP, postsyn LTP
            u_pot = bN.clone(),
            # u_dep (b,N): Filtered version of mem for STDP, postsyn LTD
            u_dep = bN.clone(),
            # w_stdp (b,N,N): STDP-dependent weight factor
            w_stdp = torch.ones((cfg.batch_size,self.N,self.N), **cfg.tspec),

            # syn (b,N): Postsynaptic current caused by model-internal
            #            presynaptic partners; kept for recording purposes
            syn = bN.clone(),
            # noise (b,N): Background noise input, kept for recordings
            noise = bN.clone()
        ))

    def initialise_epoch_state(self, inputs):
        '''
        Sets up the static state dictionary, that is, the set of parameters
        that are unchanged during integration, but may be altered from one
        epoch to the next.
        Note that this does not include parameters like readout weights
        that require no epochal setup.
        @return Box
        '''
        W = torch.einsum('e,eo->eo', self.w_signs, torch.abs(self.w))
        return Box(dict(
            # input (b,t,N): Spikes in poisson populations
            input = init.get_input_spikes(inputs),
            # W (N,N): Weights, pre;post
            W = W,
            # wmax_stdp (N,N): maximum STDP factor
            wmax_stdp = torch.where(W==0, W, cfg.model.stdp_wmax_total/W.abs()),
            # p_depr_mask (N): Mask marking short-term depressing synapses
            p_depr_mask = self.p < 0,
        ))

    def initialise_recordings(self, state, epoch, record_vars = []):
        '''
        Sets up the recordings dictionary, including both mandatory recordings
        required by delayed propagation, and recordings requested by the user.
        Names directly reflect the recorded state variables.
        @return Box
        '''
        records = Box(dict(
            _state_records = [],
            _epoch_records = []
        ))
        for varname in ['out', 'w_p', 'x_bar'] + record_vars:
            if varname in records:
                continue
            elif varname in state:
                records[varname] = [torch.zeros_like(state[varname])
                                    for _ in range(cfg.n_steps)]
                records._state_records.append(varname)
            elif varname in epoch:
                records[varname] = torch.clone(epoch[varname])
                records._epoch_records.append(varname)
        return records

    def mark_spikes(self, state):
        '''
        Marks spiking neurons (state.mem > 1) in state.out
        @read state.mem
        @write state.out
        '''
        mthr = state.mem-1.0
        state.out = SurrGradSpike.apply(mthr)

    def record_state(self, state, record):
        '''
        Records state values for delays and analysis
        @read state
        @write record
        '''
        for varname in record._state_records:
            record[varname][state.t] = state[varname]

    def compute_STP(self, state, epoch):
        '''
        Perform short-term plasticity weight update
        @read state [out, w_p]
        @write state.w_p
        '''
        dw_p = state.out*self.p*(1 + epoch.p_depr_mask*state.w_p)
        state.w_p = state.w_p*self.alpha_r + dw_p

    def compute_STDP(self, state, epoch, record):
        '''
        Perform spike-timing dependent plasticity weight update
        @read state [t, u_dep, u_pot, out, w_stdp]
        @read epoch [wmax_stdp]
        @read record [out, x_bar]
        @write state [x_bar, u_dep, u_pot, w_stdp]
        '''
        X = torch.zeros_like(state.w_stdp)
        x_bar_delayed = torch.zeros_like(state.w_stdp)
        for i, d in enumerate(self.delays):
            X += torch.einsum('be,eo->beo',
                    record.out[state.t-d], self.dmap[i])
            x_bar_delayed += torch.einsum('be,eo->beo',
                    record.x_bar[state.t-d], self.dmap[i])

        dW_dep = torch.einsum('beo,eo,bo->beo',
            X, self.A_d, relu(state.u_dep))
        dW_pot = torch.einsum('beo,eo,bo->beo',
            x_bar_delayed, self.A_p, state.out.detach()*relu(state.u_pot))

        state.x_bar = util.expfilt(state.out.detach(), state.x_bar, self.alpha_x)
        state.u_pot = util.expfilt(state.mem, state.u_pot, self.alpha_p)
        state.u_dep = util.expfilt(state.mem, state.u_dep, self.alpha_d)
        state.w_stdp = torch.clamp(torch.min(state.w_stdp + dW_pot - dW_dep,
            epoch.wmax_stdp), 0)

    def get_synaptic_current(self, state, epoch, record):
        '''
        Computes the total synaptic current, including all plasticity effects.
        @read state
        @read epoch
        @read record
        @return (batch, post) tensor of synaptic currents
        '''
        syn = torch.zeros_like(state.w_stdp)
        for i,d in enumerate(self.delays):
            syn += torch.einsum('be,be,eo,eo->beo',
                    record.out[state.t-d], (1 + record.w_p[state.t-d]),
                    epoch.W, self.dmap[i])
        return torch.einsum('beo,beo->bo', syn, state.w_stdp)

    def get_noise_background(self):
        '''
        Computes a noisy background input corresponding to jointly weighted
        independent noise spike sources.
        '''
        return self.noise_binom.sample() * self.noise_weight


    def integrate(self, state, epoch, record):
        '''
        Perform an integration time step, advancing the state.
        @read state
        @read epoch
        @read record
        @write state
        '''
        # Synaptic currents
        state.syn = self.get_synaptic_current(state, epoch, record)

        # Plasticity weight and state updates
        self.compute_STP(state, epoch)
        self.compute_STDP(state, epoch, record)

        # Integrate
        state.mem = self.alpha_mem*state.mem \
            + state.syn

        if self.has_noise:
            state.noise = self.get_noise_background()
            state.mem += state.noise

        # Refractory period -- no gradients here, because
        # indexing by variable is not gradient-accessible
        with torch.no_grad():
            state.refractory[state.out > 0] = self.t_refractory
            state.mem[state.refractory > 0] = 0
            state.refractory[state.refractory > 0] -= 1

    def finalise_recordings(self, record):
        # Swap axes from (t,b,*) to (b,t,*)
        for varname in record._state_records:
            record[varname] = torch.stack(record[varname], dim=1)

    def compute_readout(self, record):
        h2 = torch.einsum("abc,cd->abd", (record.out, self.w_out))
        out = torch.zeros((cfg.batch_size, cfg.n_outputs), **cfg.tspec)
        out_rec = []
        for t in range(cfg.n_steps):
            out = self.alpha_mem_out*out +h2[:,t]
            out_rec.append(out)

        out_rec = torch.stack(out_rec,dim=1)
        return out_rec


class SurrGradSpike(torch.autograd.Function):
    scale = 100.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input >= 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
