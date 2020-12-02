import pytest
import cantata
from cantata import config
from box import Box
import torch
import os

@pytest.fixture()
def defaults():
    conf = Box({
        'time_step': 1e-3,
        'n_steps': 100,
        'batch_size': 32,
        'n_inputs': 20,
        'n_outputs': 1,
        'tspec': {'device': torch.device('cuda:0'), 'dtype':torch.float},
        'model': {
            'weight_scale': 10.0,
            'tau_mem': 20e-3,
            'tau_mem_out': 5e-3,
            'tau_r': 100e-3,
            'tau_x': 12e-3,
            'tau_p': 30e-3,
            'tau_d': 10e-3,
            'stdp_wmin': 0.0,
            'stdp_wmax': 2.0,
            'populations': {
                'name': {
                    'n': 1,
                    'sign': 1,
                    'p': 0.0,
                    'targets': {
                        'name': {
                            'density': 1.0,
                            'delay': 0.0,
                            'A_p': 0.0,
                            'A_d_ratio': 1.5
                        }
                    }
                }
            }
        },
        'train': {}
    })
    return conf

@pytest.fixture(scope='module')
def dummy(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('config.dummy')
    main = Box({
        'model_config':'model.yaml',
        'train_config':'train.yaml'
    })
    model = Box({'data': 0})
    train = Box({'data': 1})

    main_path = tmp_path / 'main.yaml'
    model_path = tmp_path / main.model_config
    main.to_yaml(main_path)
    model.to_yaml(model_path)
    train.to_yaml(tmp_path / main.train_config)

    unsanitised = main + {
        'model': model,
        'train': train
    }
    expected = unsanitised.copy()
    config.sanitise(expected)
    return Box(dict(main=main, unsanitised=unsanitised, expected=expected,
                    main_path=main_path, model_path=model_path))

@pytest.fixture(scope='module')
def scientific(tmp_path_factory):
    strings = Box({
        1: '1e2',
        2: '2e-3',
        3: '-3e4',
        4: '-4e-5',
        5: '5.6e6',
        6: '7.8e-7',
        7: '-9.1e8',
        8: '-2.3e-9'
    })
    floats = Box(dict([(k,float(v)) for k,v in strings.items()]))
    path = tmp_path_factory.mktemp('config.scientific') / 'scientific.yaml'
    with open(path, 'w') as file:
        yaml = strings.to_yaml()
        unquoted = ''.join(yaml.split("'"))
        file.write(unquoted)
    return Box(dict(path=path, floats=floats))

@pytest.fixture(scope='module')
def reset():
    original_master = config._latest_master
    yield None
    config.load(original_master)


def test_default_config_has_model():
    assert len(config.cfg.model) > 0, 'Must load sensible defaults'
def test_default_config_has_train():
    assert len(config.cfg.train) > 0, 'Must load sensible defaults'

def test_read_file(dummy):
    assert config.read_file(dummy.model_path) == dummy.unsanitised.model

def test_read_file_scientific_notation(scientific):
    assert config.read_file(scientific.path) == scientific.floats, \
        'YML-compliant scientific notation is not read correctly'

@pytest.fixture
def template():
    return Box(dict(
        an_int = 1,
        a_float = 1.2,
        a_list = [],
        a_string = 'asdf',
        a_dict = {},
        a_nested_dict = {'name':{}}
    ))

def test_sanitise_section_replicates_template(template):
    conf = Box()
    config.sanitise_section(conf, template)
    assert conf == template

def test_sanitise_section_retains_data(template):
    filled = template.copy()
    filled.arbitrary_data = True
    filled.an_int = -8
    filled.a_float = 1.4e-9
    filled.a_list = [1,2,'no list item type checks']
    filled.a_string = 'some other string'
    filled.a_dict = dict(a=2, b=1)
    filled.a_nested_dict = {'foo': {'tired': 1}, 'spam': {'wired': True}}
    copy = filled.copy()
    config.sanitise_section(filled, template)
    assert filled == copy

def test_sanitise_section_without_typechecks(template):
    perversion = template.copy()
    perversion.an_int = 'not an int'
    perversion.a_float = 3
    perversion.a_list = 4.5
    perversion.a_string = {}
    perversion.a_dict = []
    perversion.a_nested_dict = 'absolutely not'
    copy = perversion.copy()
    config.sanitise_section(perversion, template, False)
    assert perversion == copy

def test_sanitise_empty_nested_dicts_remain_empty(defaults):
    conf = defaults.copy()
    conf.model.populations.clear()
    copy = conf.copy()
    config.sanitise(conf)
    assert conf == copy

def test_sanitise_fills_nested_entries_from_None(defaults):
    conf = defaults.copy()
    conf.model.populations.other_name = None
    defaults.model.populations.other_name = defaults.model.populations.name
    config.sanitise(conf)
    assert conf == defaults

def test_sanitise_empty_input_yields_defaults(defaults):
    conf = Box()
    config.sanitise(conf)
    assert conf == defaults

def test_sanitise_accepts_defaults_unchanged(defaults):
    conf = defaults.copy()
    config.sanitise(conf)
    assert conf == defaults

def test_read_config_path(dummy):
    assert config.read_config(dummy.main_path) == dummy.expected
def test_read_config_master(dummy):
    os.chdir(dummy.main_path.parent)
    assert config.read_config(dummy.main) == dummy.expected

def test_load_path(dummy, reset):
    config.load(dummy.main_path)
    assert config.cfg == dummy.expected
def test_load_master(dummy, reset):
    os.chdir(dummy.main_path.parent)
    config.load(dummy.main)
    assert config.cfg == dummy.expected

def test_load_affects_references(dummy, reset):
    config.load(dummy.main_path)
    assert cantata.cfg == dummy.expected

def test_reload(dummy, reset):
    config.load(dummy.main_path)
    dummy.expected.model.data = 'A new value'
    dummy.expected.model.to_yaml(dummy.model_path)
    config.reload()
    assert config.cfg == dummy.expected

def test_read_config_overwrites_subordinates(dummy):
    main = dummy.main.copy()
    main.update(model='something', train='something else')
    assert config.read_config(dummy.main) == dummy.expected
