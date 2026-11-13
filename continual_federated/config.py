"""Configuration loading and validation for continual federated runners."""

from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    return validate_config(config)


def validate_config(config):
    method = config.get('method', 'legacy')
    if method not in ('legacy', 'proposed'):
        raise ValueError('method must be legacy or proposed')
    memory = config.setdefault('distribution_memory', {})
    if method == 'proposed':
        if config.get('replay', False):
            raise ValueError('proposed mode must disable replay')
        if not memory.get('enabled', True):
            raise ValueError('proposed mode requires distribution memory')
        for key in ('max_entries', 'top_k'):
            if int(memory.get(key, 1)) < 1:
                raise ValueError('{} must be positive'.format(key))
        if float(memory.get('temperature', 1.0)) <= 0:
            raise ValueError('distribution temperature must be positive')
        if not 0 <= float(memory.get('min_similarity', 0.1)) <= 1:
            raise ValueError('min_similarity must be in [0, 1]')
    return config
