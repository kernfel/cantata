import torch
import numpy as np
from cantata import util

def expand_to_neurons(conf, varname, diagonal = False, default = 0.):
    '''
    Expands the population-level variable `varname` from `params` into a neuron-level tensor.
    * If diagonal is False (default), the returned tensor is a size (N) vector.
    * If diagonal is True, the returned tensor is a diagonal matrix of size (N,N).
    '''
    nested = [[p[varname] if varname in p else default] * p['n']
        for p in conf.populations.values()]
    flat = [i for part in nested for i in part]
    t = torch.tensor(flat)
    return t.diag() if diagonal else t

def expand_to_synapses(conf, varname, projections, default = 0.):
    '''
    Expands the projection-level variable `varname` into a
    synapse-level N*N matrix.
    @arg projections is a tuple of indices and param references as supplied by
    `build_projections`.
    '''
    N = get_N(conf)
    ret = torch.ones(N,N) * default
    for idx, p in zip(*projections):
        if varname in p:
            ret[idx] = p[varname]
    return ret

def build_population_indices(conf):
    '''
    Builds the list of population names, as well as the list of index ranges
    occupied by each population.
    @return names [list(str)]
    @return ranges [list(range)]
    '''
    names, ranges = [], []
    N = 0
    for name,pop in conf.populations.items():
        names.append(name)
        ranges.append(range(N, N+pop.n))
        N += pop.n
    return names, ranges

def build_projections(conf, names, ranges):
    '''
    Builds the projection indices and corresponding parameter set references
    @return projection_indices: A list of (pre,post) indices into the N*N
        connectivity matrix, corresponding to population-level projection
        pathways
    @return projection_params: A list of references to the corresponding
        projection parameter sets in conf, i.e. the dicts nested within
        conf.populations.*.targets
    '''
    projection_indices, projection_params = [], []
    for sname, source in zip(names, ranges):
        spop = conf.populations[sname]
        for tname, target in zip(names, ranges):
            if tname in spop.targets:
                projection_indices.append(np.ix_(source, target))
                projection_params.append(spop.targets[tname])

    return projection_indices, projection_params

def build_connectivity(conf, projections, nPre, nPost):
    '''
    Builds the flat initial weight matrix based on conf.
    @arg projections: (projection_indices, projection_params) tuple as produced
        by build_projections().
    @arg nPre, nPost: Total size of the pre- and postsynaptic populations
    @return w: Weight matrix as a torch tensor
    '''
    mask = torch.rand(nPre, nPost) - 1
    for idx, p in zip(*projections):
        # Assumes that indices are not overlapping.
        if p.spatial:
            mask[idx] += spatial_p_connect(
                len(idx[0]), len(idx[1]), p.density, p.sigma)
        else:
            mask[idx] += p.density
    return torch.where(mask>0, torch.rand(nPre,nPost), torch.zeros(1))

def spatial_p_connect(n_pre, n_post, p0, sigma):
    '''
    Connects two populations in a distance-dependent manner.
    Both populations are laid out in a unit circle sunflower seed pattern.
    Connection probability follows a Gaussian profile, that is
    p(connect at distance d) = p0 * p(X>d), X ~ N(0,sigma).
    Note that boundary effects are not corrected.
    '''
    pre, post = util.sunflower(n_pre), util.sunflower(n_post)
    d = util.polar_dist(*pre, *post)
    probability = (1-torch.erf(d/sigma/np.sqrt(2))) * p0
    return probability

def build_delay_mapping(projections, nPre, nPost, dt):
    '''
    Builds a sorted stack of binary N*N matrices corresponding to the projection
    delays specified in the model, such that for a given delay, the corresponding
    N*N matrix can be multiplied element-wise with the weight matrix to yield the
    true weights at that delay.
    Delays are sorted in ascending order.
    @arg projection: (projection_indices, projection_params) tuple as produced
        by build_projections().
    @return dmap: A list of d N*N boolean tensors, where d is the number of delays
    @return delays: The corresponding list of d integer delays in units of dt
    '''
    delays_dict = {}
    for idx, p in zip(*projections):
        d = int(p.delay / dt)
        if d not in delays_dict.keys():
            delays_dict[d] = []
        delays_dict[d].append(idx)

    delays = sorted(delays_dict.keys())
    dmap = torch.zeros(len(delays), nPre, nPost, dtype=torch.bool)

    for i, d in enumerate(delays):
        for pre, post in delays_dict[d]:
            dmap[i, pre, post] = True

    return dmap, delays
