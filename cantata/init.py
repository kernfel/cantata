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

def expand_to_synapses(projections, nPre, nPost, varname, default = 0.):
    '''
    Expands the projection-level variable `varname` into a
    synapse-level N*N matrix.
    @arg projections is a tuple of indices and param references as supplied by
    `build_projections`.
    '''
    ret = torch.ones(nPre, nPost) * default
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

def build_projections(conf):
    '''
    Builds projection indices and corresponding parameter set references of
    area-internal projections.
    @arg conf: Area configuration
    @return projection_indices: A list of (pre,post) indices into the N*N
        connectivity matrix, corresponding to population-level projection
        pathways
    @return projection_params: A list of references to the corresponding
        projection parameter sets in conf, i.e. the dicts nested within
        conf.populations.*.targets
    '''
    names, ranges = build_population_indices(conf)
    projection_indices, projection_params = [], []
    for sname, source in zip(names, ranges):
        spop = conf.populations[sname]
        for tname, target in zip(names, ranges):
            if tname in spop.targets:
                projection_indices.append(np.ix_(source, target))
                projection_params.append(spop.targets[tname])
    return projection_indices, projection_params

def build_projections_xarea(conf_pre, conf_post, areaname_post):
    '''
    Builds projection indices and corresponding parameter set references of
    cross-area projections.
    @arg conf_pre, conf_post: Area configurations
    @arg areaname_post (str): Target area name; projections refer to cross-area
        targets as `areaname_post`.`population_name`
    @return: see build_projections()
    '''
    names_pre, idx_pre = build_population_indices(conf_pre)
    names_post, idx_post = build_population_indices(conf_post)
    qualified_names = [f'{areaname_post}.{name}' for name in names_post]
    projection_indices, projection_params = [], []
    for sname, source in zip(names_pre, idx_pre):
        for qual_tname, tparams in conf_pre.populations[sname].targets.items():
            try:
                target = idx_post[qualified_names.index(qual_tname)]
            except ValueError:
                continue
            projection_indices.append(np.ix_(source, target))
            projection_params.append(tparams)
    return projection_indices, projection_params

def build_connectivity(projections, nPre, nPost):
    '''
    Builds the flat initial weight matrix.
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
    @return dmap: A d,nPre,nPost float tensor of 0s and 1s
    @return delays: The corresponding list of d integer delays in units of dt
    '''
    delays_dict = {}
    for idx, p in zip(*projections):
        d = delay_internal(p.delay, dt)
        if d not in delays_dict.keys():
            delays_dict[d] = []
        delays_dict[d].append(idx)

    delays = sorted(delays_dict.keys())
    dmap = torch.zeros(len(delays), nPre, nPost)

    for i, d in enumerate(delays):
        for pre, post in delays_dict[d]:
            dmap[i, pre, post] = True

    return dmap, delays

def delay_internal(delay_seconds, dt):
    ''' Consistency helper for internal delay functions. '''
    return max(1, int(np.round(delay_seconds / dt)))
def delay_xarea(delay_seconds, dt):
    '''
    Consistency helper for cross-area delay functions.
    Note that processing of cross-area connections is one time step behind;
    therefore, the delay must be one less.
    As a consequence, the minimum effective delay is 2 time steps.
    '''
    return max(1, int(np.round(delay_seconds / dt)) - 1)

def get_delays_xarea(conf, dt):
    '''
    Finds all cross-area projections (i.e., all projections whose target name
    contains a '.') and returns a list of the delays of those projections,
    sorted ascending.
    @arg conf: Area configuration of the source area
    @arg dt: Time step in seconds
    @return list(int): Cross-area delays in units of timesteps
    '''
    delays_set = set()
    for pop in conf.populations.values():
        for tname, p in pop.targets.items():
            if '.' in tname:
                d = delay_xarea(p.delay, dt)
                delays_set.add(d)
    return sorted(list(delays_set))

def get_delaymap_xarea(proj_xarea, delays_xarea, nPre, nPost, dt):
    '''
    Builds the delaymap corresponding to a set of cross-area projections.
    @arg proj_xarea: (indices, params) as returned by build_projections_xarea()
    @arg delays_xarea: List of delays of the source area as returned by
        get_delays_xarea()
    @arg nPre, nPost: Total number of neurons in the source and target area,
        respectively
    @arg dt: Timestep in seconds
    @return tensor(len(delays_xarea), nPre, nPost) marking [in/]active
        projections with blocks of [0/]1, respectively.
        Note that some delays may not be associated with any active projections,
        since delays_xarea covers delays to all target areas, whereas
        proj_xarea only considers a single target area.
    '''
    dmap = torch.zeros(len(delays_xarea), nPre, nPost)
    for idx, p in zip(*proj_xarea):
        d = delay_xarea(p.delay, dt)
        i = delays_xarea.index(d), *idx
        dmap[i] = True
    return dmap
