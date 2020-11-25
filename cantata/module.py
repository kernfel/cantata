import torch
import numpy as np
from cantata import util, init, cfg

class Module(torch.nn.Module):
    def __init__(self, record_hidden = False):
        super(Module, self).__init__()
        self.record_hidden = record_hidden
        self.N = N = sum([p.n for p in cfg.model.populations.values()])

        # Network structure
        self.w_signs = init.expand_to_neurons('sign')


        # Weights
        w, self.projection_indices, self.projection_params = init.build_connectivity()
        self.w = torch.nn.Parameter(w) # LEARN

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
        # Compute the input currents
        h1 = torch.einsum("abc,cd->abd", (inputs, self.w_in))

        # Zero-initialise state variables
        mem = torch.zeros((cfg.batch_size,self.N), **cfg.tspec)
        out = torch.zeros((cfg.batch_size,self.N), **cfg.tspec)
        w_p = torch.zeros((cfg.batch_size,self.N), **cfg.tspec)

        # Add Dale's Law signs to recurrent weights
        rec_weights = torch.einsum('e,eo->eo', self.w_signs, torch.abs(self.w))

        # Prepare short-term plasticity
        p_depr_mask = self.p < 0

        # Lists to collect states
        spk_rec = []
        if self.record_hidden:
            mem_rec = []
            p_rec = []
            syn_rec = []

        # Compute hidden layer activity
        for t in range(cfg.n_steps):
            # Mark spikes
            mthr = mem-1.0
            out = SurrGradSpike.apply(mthr)
            rst = torch.zeros_like(mem)
            rst[mthr > 0] = 1.0

            # Synaptic currents
            syn_p = torch.einsum('be,eo->bo', out*(1+w_p), rec_weights)
            w_p = w_p*self.alpha_p + out*self.p*(1 + p_depr_mask*w_p)

            # Record
            spk_rec.append(out)
            if self.record_hidden:
                mem_rec.append(mem)
                p_rec.append(w_p)
                syn_rec.append(syn_p)

            # Integrate
            mem = self.alpha_mem*mem +h1[:,t] +syn_p -rst

        # Recordings
        self.spk_rec = torch.stack(spk_rec,dim=1)
        if self.record_hidden:
            self.mem_rec = torch.stack(mem_rec,dim=1)
            self.p_rec = torch.stack(p_rec,dim=1)
            self.syn_rec = torch.stack(syn_rec,dim=1)

        # Readout layer
        h2 = torch.einsum("abc,cd->abd", (self.spk_rec, self.w_out))
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
