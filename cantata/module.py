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

        w_in = torch.empty((cfg.n_inputs, N),  **cfg.tspec)
        wscale_in = cfg.model.weight_scale * (1.0-util.decayconst(cfg.model.tau_mem))
        torch.nn.init.normal_(w_in, mean=0.0, std=wscale_in/np.sqrt(cfg.n_inputs))
        self.w_in = torch.nn.Parameter(w_in) # LEARN

        w_out = torch.empty((N, cfg.n_outputs), **cfg.tspec)
        wscale_out = cfg.model.weight_scale * (1.0-util.decayconst(cfg.model.tau_mem_out))
        torch.nn.init.normal_(w_out, mean=0.0, std=wscale_out/np.sqrt(N))
        self.w_out = torch.nn.Parameter(w_out) # LEARN

        # Short-term plasticity
        self.p = init.expand_to_neurons('p')
        self.alpha_r = util.decayconst(cfg.model.tau_r)

        # STDP
        self.A_p = init.expand_to_synapses('A_p', projections)
        self.A_d = self.A_p * init.expand_to_synapses('A_d_ratio', projections)
        self.alpha_x = util.decayconst(cfg.model.tau_x)
        self.alpha_p = util.decayconst(cfg.model.tau_p)
        self.alpha_d = util.decayconst(cfg.model.tau_d)

        # Membrane time constants
        self.alpha_mem = util.decayconst(cfg.model.tau_mem)
        self.alpha_mem_out = util.decayconst(cfg.model.tau_mem_out)

    def forward(self, inputs, record_vars = []):
        state = self.initialise_dynamic_state()
        epoch = self.initialise_epoch_state(inputs)
        record = self.initialise_recordings(state, epoch, record_vars)

        for state.t in range(cfg.n_steps):
            self.mark_spikes(state)
            self.record_state(state, record)
            self.integrate(state, epoch, record)

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
            mem = bN.clone(),
            # out (b,N): Spike raster
            out = bN.clone(),
            # w_p (b,N): Short-term plastic weight component, presynaptic
            w_p = bN.clone(),
            # syn (b,N): Postsynaptic current caused by model-internal
            #            presynaptic partners; kept for recording purposes
            syn = bN.clone(),
            # x_bar (b,N): Filtered version of out for STDP, presyn component
            x_bar = bN.clone(),
            # u_pot (b,N): Filtered version of mem for STDP, postsyn LTP
            u_pot = bN.clone(),
            # u_dep (b,N): Filtered version of mem for STDP, postsyn LTD
            u_dep = bN.clone(),
            # w_stdp (b,N,N): STDP-dependent weight factor
            w_stdp = torch.ones((cfg.batch_size,self.N,self.N), **cfg.tspec)
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
        return Box(dict(
            # input (b,t,N): Input currents
            input = torch.einsum("abc,cd->abd", (inputs, self.w_in)),
            # W (d,N,N): Weights, pre;post
            W = self.dmap * torch.einsum('e,eo->eo', self.w_signs, torch.abs(self.w)),
            # p_depr_mask (N): Mask marking short-term depressing synapses
            p_depr_mask = self.p < 0,
        ))

    def initialise_recordings(self, state, epoch, record_vars):
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
                records[varname] = torch.zeros(
                    (cfg.n_steps,) + state[varname].shape, **cfg.tspec)
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
        return state.out.detach()

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
        @return weight update values (batch,pre)
        '''
        dw_p = state.out*self.p*(1 + epoch.p_depr_mask*state.w_p)
        state.w_p = state.w_p*self.alpha_r + dw_p

    def compute_STDP(self, state, record):
        '''
        Perform spike-timing dependent plasticity weight update
        @read state [t, u_dep, u_pot, out, w_stdp]
        @read record [out, x_bar]
        @write state [x_bar, u_dep, u_pot, w_stdp]
        '''
        X = torch.einsum('dbe,deo->beo',
            record.out[state.t - self.delays], self.dmap)
        dW_dep = torch.einsum('beo,eo,bo->beo',
            X, self.A_d, relu(state.u_dep))

        x_bar_delayed = torch.einsum('dbe,deo->beo',
            record.x_bar[state.t - self.delays], self.dmap)
        dW_pot = torch.einsum('beo,eo,bo->beo',
            x_bar_delayed, self.A_p, state.out.detach()*relu(state.u_pot))

        state.x_bar = self.alpha_x*state.x_bar + (1 - self.alpha_x)*state.out.detach()
        state.u_pot = self.alpha_p*state.u_pot + (1 - self.alpha_p)*state.mem
        state.u_dep = self.alpha_d*state.u_dep + (1 - self.alpha_d)*state.mem
        state.w_stdp = torch.clamp(state.w_stdp + dW_pot - dW_dep, \
            cfg.model.stdp_wmin, cfg.model.stdp_wmax)

    def integrate(self, state, epoch, record):
        '''
        Perform an integration time step, advancing the state.
        @read state
        @read epoch
        @read record
        @write state
        '''
        t = state.t
        # Synaptic currents
        syn_p = record.out[t - self.delays] * (1 + record.w_p[t - self.delays])
        syn = torch.einsum('dbe,deo,beo->bo', syn_p, epoch.W, state.w_stdp)

        # Plasticity weight and state updates
        self.compute_STP(state, epoch)
        self.compute_STDP(state, record)

        # Integrate
        state.mem = self.alpha_mem*state.mem \
            + epoch.input[:,t] \
            + syn \
            - state.out.detach()
        state.syn = syn

    def finalise_recordings(self, record):
        # Swap axes from (t,b,*) to (b,t,*)
        for varname in record._state_records:
            record[varname].transpose_(0, 1)

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
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
