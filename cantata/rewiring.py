import numpy as np
import torch
import warnings


def rewire(syn, eta, alpha, T, hard=False, K=None):
    '''
    Implements deep rewiring, cf. Bellec, Kappel, Maass, Legenstein 2018.

    Weights are active/on or inactive/off depending on the value of syn.W.
    Invoking rewire() has the following effects on active weights:
    - L1 regularisation: W -= eta*alpha
    - Random walk: W += sqrt(2*eta*T) * v, v ~ N(0,1)
    - Deactivation: If W_i crosses zero as a result of the above, set it to 0.
    - Hard rewiring (hard==True): Maintain fraction of active weights:
        Compensate active weights shortfall exactly by resurrecting a random
        set of inactive weights, activating them with w~=0.
    - Soft rewiring (hard==False): A random subset of inactive weights is
        activated, as if the random walk operation were applied also to
        inactive weights.
        The probability of reactivation here follows the asymptotic
        distribution of level crossings in a random walk derived in
        Burridge & Guerre 1996.
    '''
    if hard and K is None and not hasattr(syn, 'projections'):
        warnings.warn('Requested hard rewiring without providing K on '
                      'a synapse without projections. Falling back to soft '
                      'rewiring.')

        hard = False

    with torch.no_grad():

        # Apply decay and random walk
        active_mask = syn.W_signed != 0
        Wa = syn.W_signed[active_mask].abs()
        Wa = Wa - Wa*eta*alpha + np.sqrt(2*eta*T) * torch.randn_like(Wa)

        # Deactivate on zero crossing
        Wa[Wa < 0] = 0

        # Apply to weight origin syn.W
        # Note: Sign must be maintained for synapses with sign constraint 0
        W_new = torch.zeros_like(syn.W)
        W_new[active_mask] = Wa * syn.W_signed[active_mask].sign()
        syn.W[:] = W_new

        if not hard:
            # Soft rewiring
            # sig: std of the random walk
            # MAD = sqrt(2/pi)*sig: MAD of the random walk increment
            # n(1) = MAD/sig * |N(0,1)| = sqrt(2/pi) * |N(0,1)| :
            #    Number of level crossings (1 particle)
            # n(k) = k*n(1) = k * sqrt(2/pi) * |N(0,1)|:
            #    Number of level crossings (k particles)
            # n(k) should be integer, so let's floor that.
            inactive_mask = syn.W == 0
            n_inactive = inactive_mask.sum().item()
            n_new = min(
                int(n_inactive * np.sqrt(2/np.pi) * torch.randn(1).abs()),
                n_inactive)
            if n_new > 0:
                new_indices = torch.randperm(
                    n_inactive, device=syn.W.device)[:n_new]
                tiny = torch.finfo(syn.W.dtype).tiny
                syn.W[inactive_mask][new_indices] = tiny
        else:
            raise NotImplementedError
            # Need to break out into projections for this.
