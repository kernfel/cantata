import torch
import numpy as np
from cantata import cfg, util

def expand_to_neurons(varname, diagonal = False):
    '''
    Expands the population-level variable `varname` from `params` into a neuron-level tensor.
    * If diagonal is False (default), the returned tensor is a size (N) vector.
    * If diagonal is True, the returned tensor is a diagonal matrix of size (N,N).
    '''
    nested = [[p[varname]] * p['n'] for p in cfg.model.populations.values()]
    flat = [i for part in nested for i in part]
    t = torch.tensor(flat, **cfg.tspec)
    return t.diag() if diagonal else t


def build_connectivity():
    '''
    Builds the flat initial weight matrix based on cfg.model.
    @arg wscale: Adjusted weight scale
    @return w: N*N weight matrix as a torch tensor
    @return projection_indices: A list of (pre,post) indices into w,
        corresponding to population-level projection pathways
    @return projection_params: A list of references to the corresponding projection
        parameter sets in cfg, i.e. the dicts under cfg.model.populations.*.targets
    '''
    # Build population indices:
    names, ranges = [], []
    N = 0
    for name,pop in cfg.model.populations.items():
        names.append(name)
        ranges.append(range(N, N+pop.n))
        N += pop.n

    # Build projection indices:
    projection_indices, projection_params = [], []
    for sname,pop in cfg.model.populations.items():
        source = ranges[names.index(sname)]
        for tname, params in pop.targets.items():
            target = ranges[names.index(tname)]
            projection_indices.append(np.ix_(source, target))
            projection_params.append(params)

    # Initialise matrices
    w = torch.empty((N,N), **cfg.tspec)
    mask = torch.empty((N,N), **cfg.tspec)
    zero = torch.zeros(1, **cfg.tspec)
    wscale = cfg.model.weight_scale * (1.0-util.decayconst(cfg.model.tau_mem))
    torch.nn.init.normal_(w, mean=0.0, std=wscale)
    torch.nn.init.uniform_(mask, -1, 0)

    # Build connectivity:
    for idx, p in zip(projection_indices, projection_params):
        indeg = len(idx[0]) * p['density']
        w[idx] /= np.sqrt(indeg)
        mask[idx] += p['density']
    w = torch.where(mask>0, w, zero)

    return w, projection_indices, projection_params
