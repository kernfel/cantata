import torch
import numpy as np
from cantata.config import cfg

def decayconst_tensor(dt, framework = torch):
    return lambda tau: framework.exp(-dt/tau)

def decayconst(tau):
    if type(tau) == torch.Tensor:
        return torch.exp(-cfg.time_step/tau)
    return float(np.exp(-cfg.time_step/tau))

def sigmoid_project(value, bounds, framework = torch):
    return bounds[0] + (bounds[1]-bounds[0])/(1 + framework.exp(-value))

def expfilt(target, filtered, alpha):
    return alpha*filtered + (1 - alpha)*target

# Sunflower seed pattern:
# https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle
phi = (np.sqrt(5)+1)/2 # The golden ratio
def sunflower(n, alpha=1):
    '''
    Returns a sunflower seed pattern of n points. alpha>1 causes truncation of
    radii at the outer edge, making a smoother boundary.
    '''
    b = torch.tensor(np.round(alpha * np.sqrt(n)))
    k = torch.arange(1,n+1)
    r = torch.sqrt(k-1/2) / torch.sqrt(n - (b+1)/2)
    r[r>1] = 1
    theta = 2*np.pi*k / phi**2
    return r, theta

def broadcast_outer(a,b):
    assert a.dim() == b.dim() == 1
    a = a.reshape(-1,1).expand(-1,len(b))
    b = b.reshape(1,-1)
    return a,b

def polar_dist(r1, th1, r2, th2):
    r1,r2 = broadcast_outer(r1,r2)
    th1,th2 = broadcast_outer(th1,th2)
    return torch.sqrt(r1**2 + r2**2 - 2*r1*r2*torch.cos(th1-th2))

def cartesian(r, theta):
    return r*theta.cos(), r*theta.sin()
