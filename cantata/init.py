import torch
import numpy as np
from cantata import cfg, util
from box import Box

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

def expand_to_synapses(varname, projections):
    '''
    Expands the projection-level variable `varname` into a
    synapse-level N*N matrix.
    @arg projections is a tuple of indices and param references as supplied by
    `build_projections`.
    '''
    ret = torch.empty((get_N(), get_N()), **cfg.tspec)
    for idx, p in zip(*projections):
        ret[idx] = p[varname]
    return ret

def get_N(force_calculate = False):
    '''
    Returns the total number of neurons as specified in cfg.model.N
    If cfg.model.N is not specified, calculates it from cfg.populations and
    deposits the result in cfg.model.N.
    @alters cfg
    '''
    if force_calculate or ('N' not in cfg.model):
        cfg.model.N = sum([p.n for p in cfg.model.populations.values()])
    return cfg.model.N

def build_population_indices():
    '''
    Builds the list of population names, as well as the list of index ranges
    occupied by each population.
    @return names [list(str)]
    @return ranges [list(range)]
    '''
    names, ranges = [], []
    N = 0
    for name,pop in cfg.model.populations.items():
        names.append(name)
        ranges.append(range(N, N+pop.n))
        N += pop.n
    return names, ranges

def build_input_projections():
    '''
    Builds the input projection indices and their corresponding density values.
    @see_also `build_projections()`
    @return projection_indices: A list of (input,post) indices into the I*N
    input connectivity matrix
    @return projection_density: A list of density values corresponding to each
    projection
    '''
    names, ranges = build_population_indices()

    projection_indices, projection_density = [], []
    for sname,pop in cfg.model.populations.items():
        target = ranges[names.index(sname)]
        for index, density in pop.inputs.items():
            projection_indices.append(np.ix_([index], target))
            projection_density.append(density)
    return projection_indices, projection_density

def build_projections():
    '''
    Builds the projection indices and corresponding parameter set references
    @return projection_indices: A list of (pre,post) indices into the N*N connectivity matrix,
        corresponding to population-level projection pathways
    @return projection_params: A list of references to the corresponding projection
        parameter sets in cfg, i.e. the dicts under cfg.model.populations.*.targets
    '''
    # Build population indices:
    names, ranges = build_population_indices()

    # Build projection indices:
    projection_indices, projection_params = [], []
    for sname,pop in cfg.model.populations.items():
        source = ranges[names.index(sname)]
        for tname, params in pop.targets.items():
            target = ranges[names.index(tname)]
            projection_indices.append(np.ix_(source, target))
            projection_params.append(params)

    return projection_indices, projection_params

def build_connectivity(projections):
    '''
    Builds the flat initial weight matrix based on cfg.model.
    @arg projections: (projection_indices, projection_params) tuple as produced
        by build_projections().
    @return w: N*N weight matrix as a torch tensor
    '''
    N = get_N()

    # Initialise matrices
    w = torch.empty((N,N), **cfg.tspec)
    mask = torch.empty((N,N), **cfg.tspec)
    zero = torch.zeros(1, **cfg.tspec)
    wscale = cfg.model.weight_scale * (1.0-util.decayconst(cfg.model.tau_mem))
    torch.nn.init.normal_(w, mean=0.0, std=wscale)
    torch.nn.init.uniform_(mask, -1, 0)

    # Build connectivity:
    for idx, p in zip(*projections):
        indeg = len(idx[0]) * p.density
        w[idx] /= np.sqrt(indeg)
        mask[idx] += p.density
    w = torch.where(mask>0, w, zero)

    return w

def build_delay_mapping(projections):
    '''
    Builds a sorted stack of binary N*N matrices corresponding to the projection
    delays specified in the model, such that for a given delay, the corresponding
    N*N matrix can be multiplied element-wise with the weight matrix to yield the
    true weights at that delay.
    Delays are sorted in ascending order.
    @arg projection: (projection_indices, projection_params) tuple as produced
        by build_projections().
    @return dmap: A list of d N*N boolean tensors, where d is the number of delays
    @return delays: The corresponding list of d integer delays in units of cfg.time_step
    '''
    delays_dict = {}
    for idx, p in zip(*projections):
        # Maximum delay is T-1, such that state.t-delay points at t+1
        # This wraps around to 0 at t=T, but that integration step is discarded.
        d = min(cfg.n_steps-1, int(p.delay / cfg.time_step))
        if d not in delays_dict.keys():
            delays_dict[d] = []
        delays_dict[d].append(idx)

    N = get_N()
    delays = sorted(delays_dict.keys())
    dmap = [torch.zeros(N,N, dtype=torch.bool, device=cfg.tspec.device)
            for _ in delays]

    for i, d in enumerate(delays):
        for pre, post in delays_dict[d]:
            dmap[i][pre, post] = True

    return dmap, delays
