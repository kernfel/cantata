import torch
import numpy as np
from cantata import util


def get_N(conf):
    return sum([p.n for p in conf.populations.values()])


def expand_to_neurons(conf, varname, diagonal=False, default=0.):
    '''
    Expands the population-level variable `varname` from `params` into a
        neuron-level tensor.
    * If diagonal is False (default), the returned tensor is a size (N) vector.
    * If diagonal is True, the returned tensor is a diagonal matrix of size
        (N,N).
    '''
    nested = [[p[varname] if varname in p else default] * p['n']
              for p in conf.populations.values()]
    flat = [i for part in nested for i in part]
    t = torch.tensor(flat)
    return t.diag() if diagonal else t


def expand_to_synapses(projections, nPre, nPost, varname, default=0.):
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
    for name, pop in conf.populations.items():
        names.append(name)
        ranges.append(range(N, N+pop.n))
        N += pop.n
    return names, ranges


def build_projections(conf_pre, conf_post=None, areaname_post=None):
    '''
    Builds projection indices and corresponding parameter set references of
    projections from a source to a target area.
    @arg conf_pre: Source area configuration.
    @arg conf_post: Target area configuration, or None (default) to imply
        area-internal connections.
    @arg areaname_post (str): Target area name; projections refer to cross-area
        targets as `areaname_post`.`population_name`
    @return projection_indices: A list of (pre,post) indices into the
        connectivity matrix, corresponding to population-level projection
        pathways
    @return projection_params: A list of references to the corresponding
        projection parameter sets in conf_pre, i.e. the dicts nested within
        conf_pre.populations.*.targets
    '''
    names_pre, idx_pre = build_population_indices(conf_pre)
    xarea = conf_post is not None and conf_post is not conf_pre
    if xarea:
        names_post, idx_post = build_population_indices(conf_post)
        qualified_names = [f'{areaname_post}:{name}' for name in names_post]
    else:
        idx_post = idx_pre
        qualified_names = names_pre
    projection_indices, projection_params = [], []
    for sname, source in zip(names_pre, idx_pre):
        for qual_tname, tparams in conf_pre.populations[sname].targets.items():
            try:
                target = idx_post[qualified_names.index(qual_tname)]
            except ValueError:
                continue
            projection_indices.append(np.ix_(source, target))
            tparams = tparams.copy()
            tparams.autapses = tparams.autapses or xarea
            projection_params.append(tparams)
    return projection_indices, projection_params


def build_connectivity(projections, nPre, nPost, batch_size=0):
    '''
    Builds the flat initial weight matrix.
    @arg projections: (projection_indices, projection_params) tuple as produced
        by build_projections().
    @arg nPre, nPost: Total size of the pre- and postsynaptic populations
    @arg batch_size: Builds weights as (batch,pre,post) if batch_size > 0
    @return w: Weight matrix as a torch tensor
    '''
    has_batch = batch_size > 0
    batch_size = max(batch_size, 1)
    mask = torch.zeros(batch_size, nPre, nPost, dtype=torch.bool)
    w = torch.empty(batch_size, nPre, nPost)
    for idx, p in zip(*projections):
        # Assumes that indices are not overlapping.
        # Find probability
        e, o = idx[0].size, idx[1].size
        prob = get_connection_probabilities(p, e, o)

        # Ensure fixed number of connections per projection
        N_target = int(prob.sum().round())
        if N_target == 0:
            continue

        flat_conn_indices = prob.flatten().multinomial(N_target)
        submask = torch.zeros(e, o, dtype=torch.bool)
        submask.flatten()[flat_conn_indices] = True
        mask[:, idx[0], idx[1]] = submask

        # Weight values
        if p.uniform:
            w[:, idx[0], idx[1]] = torch.rand(batch_size, e, o)
        else:
            w[:, idx[0], idx[1]] = torch.abs(
                torch.randn(batch_size, e, o) / np.sqrt(e))
    w = torch.where(mask, w, torch.zeros(1))
    return w if has_batch else w[0]


def get_connection_probabilities(syn, n_pre, n_post):
    '''
    Produces a (n_pre, n_post) connection probability matrix as specified in
        syn. Relevant parameters in syn include:
     * syn.connectivity, string ('random', 'spatial', 'one-to-one')
        - If 'random', connectivity is uniformly drawn with
            p(connect) = density.
        - If 'spatial', populations are laid out in a unit circle sunflower
            seed pattern, and connectivity follows a Gaussian profile, s.t.
            p(connect at distance d) = density * exp(-(d/sigma)**2/2) <= 1.
            Note that boundary effects are not corrected.
        - If 'one-to-one', n_pre==n_post is required, and pre_i is connected
            to post_i for i in [0,n_pre].
     * syn.density, float
        - Note that density>1 may be a reasonable choice for spatial
            connectivity.
     * syn.sigma, float
        - Width of the spatial Gaussian profile (see above).
    '''
    pre, post = util.sunflower(n_pre), util.sunflower(n_post)
    d = util.polar_dist(*pre, *post)
    return get_connection_probability(syn, d)


def get_connection_probability(syn, distance):
    '''
    Maps distances to connection probability as configured in @arg syn,
    cf. get_connection_probabilities().
    '''
    if syn.connectivity == 'spatial':
        p = torch.exp(-(distance/syn.sigma)**2/2) * syn.density
        # p = (1-torch.erf(distance/syn.sigma/np.sqrt(2))) * syn.density
    elif syn.connectivity == 'random':
        p = torch.ones_like(distance) * syn.density
    elif syn.connectivity == 'one-to-one':
        assert distance.shape[0] == distance.shape[1]
        p = torch.eye(distance.shape[0]) * syn.density
    if (not syn.autapses
            and len(distance.shape) == 2
            and distance.shape[0] == distance.shape[1]):
        p.fill_diagonal_(0)
    return p.clamp(0, 1)


def get_delay(delay_seconds, dt, xarea):
    '''
    Consistency helper for delay functions.
    Note that processing of cross-area connections is one time step behind;
    therefore, the delay must be one less.
    As a consequence, the minimum effective xarea delay is 2 time steps.
    '''
    return max(1, int(np.round(delay_seconds / dt)) - (1 if xarea else 0))


def get_delays(conf, dt, xarea):
    '''
    Returns a list of delays of projections originating from the given area,
    sorted ascending.
    @arg conf: Area configuration of the source area
    @arg dt: Time step in seconds
    @arg xarea: Whether to consider cross-area projections.
        If False (default), consider internal projections only.
        If True, consider cross-area projections only.
    @return list(int): Delays in units of timesteps
    '''
    delays_set = set()
    for pop in conf.populations.values():
        for tname, p in pop.targets.items():
            if (xarea and ':' in tname) or (not xarea and ':' not in tname):
                d = get_delay(p.delay, dt, xarea)
                delays_set.add(d)
    return sorted(list(delays_set))


def get_delaymap(projections, dt, conf_pre, conf_post=None):
    '''
    Builds the delaymap corresponding to a set of projections.
    @arg projections: (indices, params) as returned by build_projections()
    @arg dt: Timestep in seconds
    @arg conf_pre: Source area configuration
    @arg conf_post: Target area configuration, or None (default) to imply
        area-internal connections.
    @return tensor(len(delays), nPre, nPost) marking [in/]active
        projections with blocks of [0/]1, respectively.
        Note that some delays may not be associated with any active projections
        under cross-area conditions.
    '''
    xarea = conf_post is not None and conf_post is not conf_pre
    if not xarea:
        conf_post = conf_pre
    delays = get_delays(conf_pre, dt, xarea)
    dmap = torch.zeros(len(delays), get_N(conf_pre), get_N(conf_post))
    for idx, p in zip(*projections):
        d = get_delay(p.delay, dt, xarea)
        i = delays.index(d), *idx
        dmap[i] = True
    return dmap
