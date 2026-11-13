"""Configuration loading and validation for continual federated runners."""

from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    return validate_config(config)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('configuration must be a mapping')
    method = config.get('method', 'legacy')
    if method not in ('legacy', 'proposed'):
        raise ValueError('method must be legacy or proposed')
    strategy = config.setdefault('strategy', 'fedmixer')
    if strategy not in ('fedavg', 'fedmixer'):
        raise ValueError('strategy must be fedavg or fedmixer')
    memory = config.setdefault('distribution_memory', {})
    distillation = config.setdefault('distillation', {})
    for key in ('history', 'representation', 'mutual'):
        distillation.setdefault(key, True)
    for key in ('history_weight', 'representation_weight'):
        value = float(distillation.get(key, 0.0))
        if value < 0:
            raise ValueError('{} must be non-negative'.format(key))
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
        if float(memory.get('variance_weight', 0.5)) < 0:
            raise ValueError('variance_weight must be non-negative')
    return config
