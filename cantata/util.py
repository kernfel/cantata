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
