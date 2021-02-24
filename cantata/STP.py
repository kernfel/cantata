import torch
from torch.nn.functional import relu
import numpy as np
from box import Box
from cantata import util, init, cfg

class STP(torch.nn.Module):
    def __init__(self, population, batch_size):
        super(STP, self).__init__()
        self.register_buffer('w_stp', torch.zeros(batch_size, population.n))


    def
