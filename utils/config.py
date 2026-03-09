import json
import os
from collections.abc import Mapping
from functools import reduce

import yaml


class Config(Mapping):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    # Support item assignment
    def __setitem__(self, key, value):
        setattr(self, key, value)

    # Support ** unpacking
    def __iter__(self):
        return iter(self.__dict__)

    # Support in operator
    def __contains__(self, key):
        return hasattr(self, key)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return self.to_json()
    
    def save(self, path):
        with open(path, "w") as f:
            f.write(self.to_json())
    
    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        return Config.from_dict(data)

    def to_dict(self):
        final_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                final_dict[key] = value.to_dict()
            elif isinstance(value, list):
                final_dict[key] = [
                    item.to_dict() if isinstance(item, Config) else item
                    for item in value
                ]
            else:
                final_dict[key] = value
        return final_dict

    def to_json(self, ident=4, sort_keys=True):
        return yaml.dump(self.to_dict(), indent=ident, sort_keys=sort_keys)

    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, _dict):
        if not isinstance(_dict, dict):
            raise ValueError("Input data must be a dictionary")
        return setup_config(_dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as f:
            return cls.from_dict(yaml.load(f, Loader=yaml.FullLoader))

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_file(cls, path):
        if path.endswith(".yaml"):
            return cls.from_yaml(path)
        elif path.endswith(".json"):
            return cls.from_json(path)
        else:
            raise ValueError("Only YAML and JSON files are supported")

    def iterate_items(self, root=""):
        """
        Iterate over the items of the config
        """
        for key, value in self.__dict__.items():
            path = f"{root}.{key}" if root else key
            if isinstance(value, Config):
                yield from value.iterate_items(root=path)
            else:
                yield (path, value)


def setup_config(cfg):
    """
    Recursively turns the config dictionary into a dataclass
    """
    if not any(isinstance(cfg, t) for t in [dict, list, tuple, Config]):
        return cfg
    for key, value in cfg.items():
        if any(isinstance(value, t) for t in [dict, Config]):
            cfg[key] = setup_config(cfg=value)
        elif isinstance(value, list):
            cfg[key] = [setup_config(item) for item in value]
        elif isinstance(value, tuple):
            cfg[key] = tuple(setup_config(item) for item in value)
    return Config(**cfg)


def _load_config(config_path):
    """
    Recursively load the config file and its dependencies
    """
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    with open(config_path, "r") as f:
        lines = f.readlines()
    config = yaml.load("".join(lines), Loader=yaml.FullLoader)
    final_config = {}
    for path in config.pop("LOAD", []):
        # if LOAD : _path_ -> load the file (recursively)
        loaded_config = _load_config(path)
        final_config = recursive_update(final_config, loaded_config)
    # Current config has precedence over loaded elements:
    final_config = recursive_update(
        final_config, config
    )
    return final_config

def replace_value(config: Config, value: str, CONSTANTS: dict):
    """
    Replace the value with the config value
    """
    while "${" in str(value):
        # get the key
        start = value.find("${")
        end = value.find("}")
        key = value[start + 2 : end]
        try:
            _config = config
            _config = reduce(
                lambda cfg, key: cfg.__getitem__(key), key.split("."), _config
            )
        except AttributeError:
            if key in CONSTANTS:
                _config = CONSTANTS.get(key, None)
            else:
                raise ValueError(f"Key not found: {key}")
            
        # replace the value
        # value = value[:start] + str(_config) + value[end + 1 :]
        # value = dtype(value[:start]) + _config + dtype(value[end + 1 :])
        if len(value[:start]) > 0:
            _config = value[:start] + type(value[:start])(_config)
        if len(value[end + 1 :]) > 0:
            _config = type(value[end + 1 :])(_config) + value[end + 1 :]
        value = _config

    return value


def interpolate(config, _config, CONSTANTS):
    """
    Interpolate the config values with the root config
    
    Args:
    - config: the config to interpolate
    - _config: the root config
    - CONSTANTS: the constants to interpolate

    For example:
    ${root.key} -> config.root.key
    """
    if not isinstance(config, Config):
        return config
    for key, value in config.items():
        if isinstance(value, Config):
            config[key] = interpolate(value, _config, CONSTANTS)
        elif isinstance(value, list):
            config[key] = [interpolate(item, _config, CONSTANTS) for item in value]
        elif isinstance(value, str) and "${" in value:
            value = replace_value(_config, value, CONSTANTS)
            config[key] = value

    return config


def load_config(config_path, CONSTANTS={}):
    """
    Load the config file
    """
    # !!load should be a custom tag that nests the loaded config
    def anon(loader, node):
        return _load_config(node.value)
    yaml.add_constructor(u'!load_yaml', anon)
    config = _load_config(config_path)
    config = Config.from_dict(config)
    config = interpolate(config, config, CONSTANTS)
    return config


def recursive_update(state, u):
    """
    Args:
    - state: the dictionary to update
    - u: the dictionary to update from
    Precedence: u over d
    """
    assert isinstance(state, dict) and isinstance(u, dict), "Both d and u must be dictionaries"
    for k, v in u.items():
        value = state.get(k, None)
        if k == "reset" and v is True:
            state = {}
        elif isinstance(v, dict):
            state[k] = recursive_update(value or {}, v)
        elif isinstance(v, list):
            state[k] = (value or []) + v if v[0] is not None else v[1:]
        else:
            state[k] = v
    return state
