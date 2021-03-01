import oyaml as yaml
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

defaults = read_file(Path(__file__).parent / 'configs' / 'defaults.yaml')

def read(master):
    '''
    Loads a complete configuration.
    @arg master: path_like | dict | Box:
        Top-level configuration data. If path_like, expects a yaml file.
    @return A Box containing the sanitised configuration.
    '''
    if type(master) == dict or type(master) == Box:
        conf = Box(master)
    else:
        conf = read_file(master)
    return sanitise(conf)

def sanitise(section, defaults = defaults, path = 'config'):
    '''
    Validates a config section or single entry, filling in any missing values
    and aligning existing entries' types with the corresponding default's.
    Default entries keyed with 'NAME' and 'INDEX' are treated as special cases,
    allowing an arbitrary number of entries aligned with the default value,
    and keyed with arbitrary string ('NAME') or integer ('INDEX') indices,
    respectively.
    '''
    if section == None:
        if type(default) != Box:
            return default
        elif 'NAME' in default or 'INDEX' in default:
            return Box()
        else:
            section = Box()
            # Fall through to final case to populate
    elif type(section) != type(default):
        try:
            return type(default)(section)
        except ValueError:
            raise TypeError('Failed to cast {}:{} from {} to {}'.format(
            path, section, type(section), type(default)
            )) from None
    elif type(section) != Box:
        return section
    elif 'NAME' in default:
        for name, value in section.items():
            section[name] = sanitise(
                value, default.NAME, f'{path}.{name}')
        return section
    elif 'INDEX' in default:
        indexed = Box()
        for index, value in section.items():
            int_index = sanitise(index, 0, f'{path}.$index')
            indexed[int_index] = sanitise(
                value, default.INDEX, f'{path}.{index}')
        return indexed
    # else: default is a default-keyed dict
    for key, default_value in default.items():
        section[key] = sanitise(
            section[key] if key in section else None,
            default_value, f'{path}.{key}')
    return section

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
