import pytest
import cantata
from cantata import config
from box import Box
import torch
import os

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

def test_read_file(dummy):
    assert config.read_file(dummy.model_path) == dummy.unsanitised.model

def test_read_file_scientific_notation(scientific):
    assert config.read_file(scientific.path) == scientific.floats, \
        'YML-compliant scientific notation is not read correctly'

def test_sanitise_recursive_yields_populated_unkeyed_from_None():
    value = None
    default = Box({'asdf':'jkl'})
    returned = config.sanitise_recursive(value, default)
    assert returned == default

def test_sanitise_recursive_yields_empty_named_from_None():
    value = None
    default = Box({'NAME': 'foo'})
    returned = config.sanitise_recursive(value, default)
    assert returned == Box()

def test_sanitise_recursive_yields_empty_indexed_from_None():
    value = None
    default = Box({'INDEX': 'bar'})
    returned = config.sanitise_recursive(value, default)
    assert returned == Box()

def test_sanitise_recursive_yields_default_from_None():
    value = None
    default = 'a default value'
    returned = config.sanitise_recursive(value, default)
    assert returned == default

def test_sanitise_recursive_casts_float_to_int():
    value = torch.rand(1).item()
    default = 1
    returned = config.sanitise_recursive(value, default)
    assert type(returned) == int

def test_sanitise_recursive_casts_int_to_float():
    value = torch.randint(10, (1,)).item()
    assert type(value) == int
    default = 1.0
    returned = config.sanitise_recursive(value, default)
    assert type(returned) == float

@pytest.fixture
def template():
    return Box(dict(
        an_int = 1,
        a_float = 0.0,
        a_dict = dict(
            named_dicts = {'NAME': {'named_dict_entry': 5}},
            named_value = {'NAME': 5},
            indexed_dicts = {'INDEX': {'indexed_dict_entry': 1.2}},
            indexed_value = {'INDEX': 0.4}
        )
    ))

def test_sanitise_recursive_inserts_unkeyed_defaults(template):
    conf = Box()
    expected = Box(dict(
        an_int = 1,
        a_float = 0.0,
        a_dict = dict(
            named_dicts = {},
            named_value = {},
            indexed_dicts = {},
            indexed_value = {}
        )
    ))
    config.sanitise_recursive(conf, template)
    assert conf == expected

def test_sanitise_recursive_inserts_unkeyed_from_None(template):
    conf = Box({'an_int': None, 'a_float': None, 'a_dict': None})
    expected = Box(dict(
        an_int = 1,
        a_float = 0.0,
        a_dict = dict(
            named_dicts = {},
            named_value = {},
            indexed_dicts = {},
            indexed_value = {}
        )
    ))
    config.sanitise_recursive(conf, template)
    assert conf == expected

def test_sanitise_recursive_typechecks_unkeyed_types(template):
    conf = template.copy()
    conf.an_int = 15.2
    conf.a_float = -3
    conf.a_dict = Box() # ignore for this test
    config.sanitise_recursive(conf, template)
    assert conf.an_int == 15
    assert type(conf.an_int) == int
    assert conf.a_float == -3.0
    assert type(conf.a_float) == float

def test_sanitise_recursive_typechecks_unkeyed_nested(template):
    defaults = Box({'super': template})
    conf = defaults.copy()
    conf.super.an_int = 6.45
    conf.super.a_float = 21
    conf.super.a_dict = Box() # ignore for this test
    config.sanitise_recursive(conf, defaults)
    assert conf.super.an_int == 6
    assert type(conf.super.an_int) == int
    assert conf.super.a_float == 21.0
    assert type(conf.super.a_float) == float

def test_sanitise_recursive_makes_integer_index_keys(template):
    conf = Box({'a_dict': {'indexed_value': {'1': 1., 2: 2.},
                            'indexed_dicts': {'3': {}, 4: {}}}})
    config.sanitise_recursive(conf, template)
    expected_value = {1: 1., 2: 2.}
    expected_dicts = {3: template.a_dict.indexed_dicts.INDEX,
                      4: template.a_dict.indexed_dicts.INDEX}
    assert conf.a_dict.indexed_value == expected_value
    assert conf.a_dict.indexed_dicts == expected_dicts

def test_sanitise_recursive_throws_for_invalid_index_keys(template):
    conf = Box({'a_dict': {'indexed_value': {'not an int': 3}}})
    with pytest.raises(TypeError):
        config.sanitise_recursive(conf, template)

def test_sanitise_recursive_inserts_keyed_defaults(template):
    conf = Box({'a_dict': {'named_dicts': {'name1': None, 'name2': {}},
                            'named_value': {'name3': None},
                            'indexed_dicts': {4: {}, 5: None},
                            'indexed_value': {6: None}}})
    named = template.a_dict.named_dicts.NAME
    indexed = template.a_dict.indexed_dicts.INDEX
    expected = Box({'named_dicts': {'name1': named, 'name2': named},
                    'named_value': {'name3': template.a_dict.named_value.NAME},
                    'indexed_dicts': {4: indexed, 5: indexed},
                    'indexed_value': {6: template.a_dict.indexed_value.INDEX}})
    config.sanitise_recursive(conf, template)
    assert conf.a_dict == expected

def test_sanitise_recursive_does_not_insert_unmentioned_keys(template):
    conf = Box({'a_dict': {'named_dicts': {},
                            'named_value': None,
                            'indexed_dicts': None,
                            'indexed_value': {}}})
    expected = Box({'named_dicts': {},
                    'named_value': {},
                    'indexed_dicts': {},
                    'indexed_value': {}})
    config.sanitise_recursive(conf, template)
    assert conf.a_dict == expected

def test_sanitise_recursive_typechecks_keyed_values(template):
    conf = Box({'a_dict': {'named_dicts': {'name1': {'named_dict_entry': 0.5}},
                            'named_value': {'name2': -5.2},
                            'indexed_dicts': {'3': {'indexed_dict_entry': 4}},
                            'indexed_value': {'4': 8}}})
    expected = Box({'named_dicts': {'name1': {'named_dict_entry': 0}},
                    'named_value': {'name2': -5},
                    'indexed_dicts': {3: {'indexed_dict_entry': 4.0}},
                    'indexed_value': {4: 8.0}})
    config.sanitise_recursive(conf, template)
    assert conf.a_dict == expected

def test_sanitise_recursive_retains_optional_entries(template):
    conf = Box({'top': 1, 'a_dict': {'nested': 2,
        'named_dicts': {'name': {'deep': 3}}}})
    config.sanitise_recursive(conf, template)
    assert 'top' in conf and conf.top == 1
    assert 'nested' in conf.a_dict and conf.a_dict.nested == 2
    assert 'deep' in conf.a_dict.named_dicts.name
    assert conf.a_dict.named_dicts.name.deep == 3

def test_sanitise_recursive_raises_for_inconvertible(template):
    conf = Box({'an_int': 'absolutely not an int'})
    with pytest.raises(TypeError):
        config.sanitise_recursive(conf, template)

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
