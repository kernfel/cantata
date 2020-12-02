import torch
import numpy as np
from box import Box
from cantata import util, init, cfg

class Module(torch.nn.Module):
    def __init__(self, record_hidden = False):
        super(Module, self).__init__()
        self.record_hidden = record_hidden
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
        self.alpha_p = util.decayconst(cfg.model.tau_p)


        # Membrane time constants
        self.alpha_mem = util.decayconst(cfg.model.tau_mem)
        self.alpha_mem_out = util.decayconst(cfg.model.tau_mem_out)

    def forward(self, inputs):
        state = self.initialise_dynamic_state()
        epoch = self.initialise_epoch_state(inputs)
        record = self.initialise_recordings()

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
        return Box(dict(
            mem = torch.zeros((cfg.batch_size,self.N), **cfg.tspec),
            # mem (b,N): Membrane potential
            out = torch.zeros((cfg.batch_size,self.N), **cfg.tspec),
            # out (b,N): Spike raster
            w_p = torch.zeros((cfg.batch_size,self.N), **cfg.tspec),
            # w_p (b,N): Short-term plastic weight component, presynaptic
            syn = torch.zeros((cfg.batch_size,self.N), **cfg.tspec)
            # syn (b,N): Postsynaptic current caused by model-internal
            #            presynaptic partners
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
            input = torch.einsum("abc,cd->abd", (inputs, self.w_in)),
            # input (b,t,N): Input currents
            W = self.dmap * torch.einsum('e,eo->eo', self.w_signs, torch.abs(self.w)),
            # W (d,N,N): Weights, pre;post
            p_depr_mask = self.p < 0,
            # p_depr_mask (N): Mask marking short-term depressing synapses
        ))

    def initialise_recordings(self):
        '''
        Sets up the recordings dictionary, including both mandatory recordings
        required by delayed propagation, and recordings requested by the user.
        Names directly reflect the recorded state variables.
        @return Box
        '''
        records = Box(dict(
            out = torch.zeros((cfg.n_steps, cfg.batch_size, self.N), **cfg.tspec),
            w_p = torch.zeros((cfg.n_steps, cfg.batch_size, self.N), **cfg.tspec)
        ))
        if self.record_hidden:
            records.mem = []
            records.syn = []
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
        record.out[state.t] = state.out
        record.w_p[state.t] = state.w_p
        if self.record_hidden:
            record.mem.append(state.mem)
            record.syn.append(state.syn)

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
        state.syn = torch.einsum('dbe,deo->bo', syn_p, epoch.W)
        state.w_p = state.w_p*self.alpha_p \
                + state.out*self.p*(1 + epoch.p_depr_mask*state.w_p)

        # Integrate
        state.mem = self.alpha_mem*state.mem \
            + epoch.input[:,t] \
            + state.syn \
            - state.out.detach()

    def finalise_recordings(self, record):
        # Swap axes from (t,b,*) to (b,t,*) for mandatory recordings
        record.out.transpose_(0, 1)
        record.w_p.transpose_(0, 1)
        # Stack user-requested recordings into (b,t,*) tensors
        if self.record_hidden:
            record.mem = torch.stack(record.mem, dim=1)
            record.syn = torch.stack(record.syn, dim=1)

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
