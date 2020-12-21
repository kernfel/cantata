import yaml
import re
import box
from box import Box
from pathlib import Path
import torch

# Parse scientific notation correctly
# cf. https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
# cf. https://github.com/yaml/pyyaml/issues/173
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def read_file(path):
    '''
    Reads the YAML file specified by @arg path and returns its contents
    in a Box.
    '''
    with open(path, 'r') as ymlfile:
        return Box(yaml.load(ymlfile, Loader=loader))

def read_config(master):
    '''
    Loads a complete configuration from an orchestrating master.
    @arg master: path_like | dict | Box
        Top-level orchestration data. Model and training configurations can be
        supplied directly in 'model' and 'train' entries, or by referring to
        subordinate files in the 'model_config' and 'train_config' entries, respectively.
        In that case, existing 'model' or 'train' entries will be overwritten.
        * If @arg master is path_like, data is read from the indicated YAML file,
        and subordinate file paths are resolved from its parent directory.
        * Otherwise, subordinate file paths are resolved from the working directory.
    @return A Box containing the loaded configuration.
    '''
    if type(master) == dict or type(master) == Box:
        conf = Box(master)
        dir = Path()
    else:
        conf = read_file(master)
        dir = Path(master).parent
    if 'model_config' in conf:
        conf.model = read_file(dir / conf.model_config)
    if 'train_config' in conf:
        conf.train = read_file(dir / conf.train_config)
    return sanitise(conf)

def load(master):
    '''
    Loads a master as specified in `config.read_config` and sets the global
    `config.cfg` to the result.
    '''
    global _latest_master
    cfg.clear()
    cfg.update(read_config(master))
    _latest_master = master

def reload():
    '''
    Reloads the last master passed to `load` to e.g. reflect changes in
    configuration files.
    '''
    load(_latest_master)

def sanitise(conf):
    '''
    Validates a full configuration against cfg_defaults, filling in any missing
    values and aligning existing entries' types with the corresponding default's.
    '''
    sanitise_section(conf, cfg_defaults.base)
    sanitise_section(conf.model, cfg_defaults.model)
    for pop in conf.model.populations.values():
        sanitise_section(pop, cfg_defaults.population)
        for target in pop.targets.values():
            sanitise_section(target, cfg_defaults.target)

    sanitise_section(conf.tspec, cfg_defaults.tspec, False)
    if type(conf.tspec.device) != torch.device:
        conf.tspec.device = torch.device(conf.tspec.device)
    if type(conf.tspec.dtype) != torch.dtype:
        conf.tspec.dtype = torch_types[conf.tspec.dtype]

    return conf

def sanitise_section(section, section_defaults, typecheck = True):
    '''
    Validates a config section, filling in any missing values and aligning
    existing entries' types with the corresponding default's.
    '''
    for key, value in section_defaults.items():
        if key not in section:
            section[key] = value
        elif typecheck and type(section[key]) != type(value):
            if section[key] == None:
                section[key] = value
            else:
                try:
                    section[key] = type(value)(section[key])
                except ValueError:
                    raise TypeError('Failed to cast {}:{} from {} to {}'.format(
                    key, section[key], type(section[key]), type(value)
                    )) from None

        if typecheck and type(value) == Box and 'name' in value:
            for nested_key, nested_value in section[key].items():
                if type(nested_value) != type(value.name):
                    if nested_value == None:
                        section[key][nested_key] = value.name
                    else:
                        try:
                            section[key][nested_key] = type(value.name)(nested_value)
                        except ValueError:
                            raise TypeError('Failed to cast {}:{}:{} from {} to {}'.format(
                            key, nested_key, nested_value,
                            type(nested_value), type(value.name)
                            )) from None

torch_types = {
    'float': torch.float,
    'float32': torch.float32,
    'double': torch.double,
    'float64': torch.float64,
    'cfloat': torch.cfloat,
    'complex64': torch.complex64,
    'cdouble': torch.cdouble,
    'complex128': torch.complex128,
    'half': torch.half,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'short': torch.short,
    'int16': torch.int16,
    'int': torch.int,
    'int32': torch.int32,
    'long': torch.long,
    'int64': torch.int64,
    'bool': torch.bool
}

cfg = Box()
cfg_defaults = read_file(Path(__file__).parent / 'configs' / 'defaults.yaml')
load(Path(__file__).parent / 'configs' / 'base.yaml')
