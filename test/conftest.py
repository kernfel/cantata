import pytest
from box import Box
from pathlib import Path
import os
import torch
import numpy as np
from cantata import config

def pytest_addoption(parser):
    parser.addoption(
        "--batch_size",
        action="store",
        help="Sets batch_size to 2^[arg].")


def pytest_generate_tests(metafunc):
    if "batch_size" in metafunc.fixturenames:
        bs = metafunc.config.getoption("batch_size", None)
        if bs is None:
            bsize = [1, 32]
        else:
            bsize = [2**int(bs)]
        metafunc.parametrize("batch_size", bsize, scope='session')

@pytest.fixture(scope='session')
def conf_components():
    '''
    Loads all fixture configs for modular use in test-scope fixtures,
    and ensures cfg resets after testing this module.
    '''
    path = Path(__file__).parent / 'fixtures' / 'config'
    filenames = os.listdir(path)
    bits = {}
    for f in filenames:
        base, suffix = f.split('.')
        bits[base] = config.read_file(path / f)
    return Box(bits)

@pytest.fixture(scope='class')
def model1(conf_components):
    return config.read(conf_components.model1)

@pytest.fixture(scope='class')
def model2(conf_components):
    return config.read(conf_components.model2)

@pytest.fixture(scope='session', params = [1e-3, 1e-4])
def dt(request):
    return request.param

@pytest.fixture(scope='class')
def model1_noisy(conf_components, dt):
    conf = conf_components.model1.copy()
    conf.areas.A1.populations.Exc.noise_N = np.random.randint(500) + 500
    conf.areas.A1.populations.Exc.noise_rate = np.random.rand()/dt
    conf.areas.A1.populations.Exc.noise_weight = np.random.rand()/500
    conf.areas.A1.populations.Inh.noise_N = np.random.randint(500) + 500
    conf.areas.A1.populations.Inh.noise_rate = np.random.rand()/dt
    conf.areas.A1.populations.Inh.noise_weight = np.random.rand()/500
    return config.read(conf)

class GenericModuleTests:
    @staticmethod
    def check_state(model, keys, shapes):
        state = model.state_dict()
        own_keys = []
        for key in state.keys():
            if '.' not in key:
                own_keys.append(key)
        assert len(own_keys) == len(keys)
        for key, shape in zip(keys, shapes):
            assert key in keys
            assert state[key].shape == shape, key

    @staticmethod
    def check_reset_clears(model, *keys, inputs = None):
        for key in keys:
            buffer = getattr(model, key)
            setattr(model, key, (2*torch.rand(buffer.shape)).to(buffer))
        if inputs is None:
            model.reset()
        else:
            model.reset(*inputs)
        for key in keys:
            buffer = getattr(model, key)
            assert torch.all(buffer == 0)

@pytest.fixture(scope='session')
def module_tests():
    return GenericModuleTests

@pytest.fixture(scope='session')
def spikes():
    def fn(*shape):
        X = torch.rand(shape) * 2
        X = torch.threshold(X, 1, 0)
        X = torch.clip(X, 0, 1)
        return X
    return fn
