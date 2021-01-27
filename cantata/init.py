import torch
import numpy as np
from cantata import cfg, util
from box import Box

def expand_to_neurons(varname, diagonal = False, default = 0.):
    '''
    Expands the population-level variable `varname` from `params` into a neuron-level tensor.
    * If diagonal is False (default), the returned tensor is a size (N) vector.
    * If diagonal is True, the returned tensor is a diagonal matrix of size (N,N).
    '''
    nested = [[p[varname] if varname in p else default] * p['n']
        for p in cfg.model.populations.values()]
    flat = [i for part in nested for i in part]
    t = torch.tensor(flat, **cfg.tspec)
    return t.diag() if diagonal else t

def expand_to_synapses(varname, projections, default = 0.):
    '''
    Expands the projection-level variable `varname` into a
    synapse-level N*N matrix.
    @arg projections is a tuple of indices and param references as supplied by
    `build_projections`.
    '''
    ret = torch.ones((get_N(), get_N()), **cfg.tspec) * default
    for idx, p in zip(*projections):
        if varname in p:
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

def build_output_projections():
    '''
    Builds the output projection indices and their corresponding density values.
    @see_also `build_projections()`
    @return projection_indices: A list of (pre,output) indices into the N*O
    output connectivity matrix
    @return projection_params: A list of {density:1.0} Box dicts required for
    `build_connectivity()`.
    '''
    names, ranges = build_population_indices()
    projection_indices, projection_params = [], []
    for sname, pop in cfg.model.populations.items():
        source = ranges[names.index(sname)]
        if 0 <= pop.output < cfg.n_outputs:
            projection_indices.append(np.ix_(source, [pop.output]))
            projection_params.append(Box({'density': 1.0}))
    return projection_indices, projection_params

def build_projections():
    '''
    Builds the projection indices and corresponding parameter set references
    @return projection_indices: A list of (pre,post) indices into the N*N connectivity matrix,
        corresponding to population-level projection pathways
    @return projection_params: A list of references to the corresponding projection
        parameter sets in cfg, i.e. the dicts under cfg.model.populations.*.targets
    '''
    names, ranges = build_population_indices()
    projection_indices, projection_params = [], []
    for sname, source in zip(names, ranges):
        spop = cfg.model.populations[sname]
        for tname, target in zip(names, ranges):
            if tname in spop.targets:
                projection_indices.append(np.ix_(source, target))
                projection_params.append(spop.targets[tname])

    return projection_indices, projection_params

def build_connectivity(projections, shape = None, wscale = 0):
    '''
    Builds the flat initial weight matrix based on cfg.model.
    @arg projections: (projection_indices, projection_params) tuple as produced
        by build_projections().
    @arg shape: (N_pre, N_post) tuple. If None (default), (N,N) is assumed.
    @arg wscale: Weight scale. Weights are drawn from a normal distribution
        N(0, wscale/sqrt(d)), where d is the indegree (= mean number of incoming
        synapses, accounting for connection density) of the projection.
        If wscale=0 (default), wscale is computed as util.wscale(cfg.model.tau_mem)
    @return w: Weight matrix as a torch tensor
    '''
    if shape is None:
        shape = (get_N(), get_N())
    if wscale == 0:
        wscale = util.wscale(cfg.model.tau_mem)

    # Initialise matrices
    w = torch.empty(shape, **cfg.tspec)
    mask = torch.empty(shape, **cfg.tspec)
    zero = torch.zeros(1, **cfg.tspec)
    torch.nn.init.normal_(w, mean=0.0, std=wscale)
    torch.nn.init.uniform_(mask, -1, 0)

    # Build connectivity:
    for idx, p in zip(*projections):
        indeg = len(idx[0]) * p.density
        w[idx] /= np.sqrt(indeg)
        mask[idx] += p.density # Not a bug, provided indices are not overlapping.
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

def get_input_spikes(rates):
    spikes = torch.zeros(cfg.batch_size, cfg.n_steps, get_N(), **cfg.tspec)
    rmap = expand_to_neurons('rate')
    norm_rates = torch.clip(rates * cfg.time_step, 0, 1)
    for i in range(cfg.n_inputs):
        mask = (rmap == i)
        spikes[:,:,mask] = torch.bernoulli(norm_rates[:,:,(i,)].expand(
            (cfg.batch_size, cfg.n_steps, mask.count_nonzero())
        ))
    return spikes
