"""Adapters for the cloned data_fcl seasonal NPZ pipeline."""

import json
from pathlib import Path

import numpy as np
import torch


def _batches(path, batch_size):
    with np.load(path, allow_pickle=False) as task:
        inputs = task['train_x'].astype(np.float32, copy=True)
        targets = task['train_y'].astype(np.float32, copy=True)
        test_inputs = task['test_x'].astype(np.float32, copy=True)
        test_targets = task['test_y'].astype(np.float32, copy=True)
        metadata = {
            'task_id': str(task['season_start']),
            'season_id': str(task['season_id']) if 'season_id' in task else path.stem,
        }

    def make_batches(values, labels):
        return [
            (torch.from_numpy(values[start:start + batch_size]),
             torch.from_numpy(labels[start:start + batch_size]))
            for start in range(0, len(values), batch_size)
        ]

    return make_batches(inputs, targets), make_batches(test_inputs, test_targets), metadata


def load_season_rounds(processed_dir, batch_size=32, stations=None):
    """Return chronological active-client training/evaluation rounds.

    Missing station-season tasks are represented by ``None``. They are never
    replaced with synthetic or zero-valued data.
    """
    root = Path(processed_dir)
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('missing data_fcl manifest: {}'.format(manifest_path))
    with manifest_path.open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    selected = None if stations is None else {str(value) for value in stations}
    included = [
        item for item in manifest
        if item.get('status') == 'included'
        and (selected is None or str(item.get('station_id')) in selected)
    ]
    if not included:
        raise ValueError('manifest contains no selected included season tasks')

    station_ids = sorted({str(item['station_id']) for item in included})
    tasks = {}
    for item in included:
        station_id = str(item['station_id'])
        season_id = str(item['season_id'])
        path = root / 'windows' / station_id / (season_id + '.npz')
        if not path.is_file():
            raise FileNotFoundError('manifest task is missing: {}'.format(path))
        train, test, metadata = _batches(path, batch_size)
        tasks[(station_id, str(item['season_start']))] = (train, test, metadata)

    starts = sorted({str(item['season_start']) for item in included})
    train_rounds = []
    test_rounds = []
    round_metadata = []
    for start in starts:
        train_round = []
        test_round = []
        for station_id in station_ids:
            task = tasks.get((station_id, start))
            train_round.append(None if task is None else task[0])
            test_round.append(None if task is None else task[1])
        train_rounds.append(train_round)
        test_rounds.append(test_round)
        round_metadata.append({
            'task_position': len(round_metadata),
            'season_start': start,
            'active_clients': [index for index, value in enumerate(train_round)
                               if value is not None],
        })
    return train_rounds, test_rounds, station_ids, round_metadata
