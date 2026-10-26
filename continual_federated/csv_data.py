"""Small, dataset-agnostic helpers for numeric CSV time series."""

import csv

import torch


def load_csv_series(path, target=None, limit=None):
    with open(path, newline='', encoding='utf-8') as handle:
        rows = csv.DictReader(handle)
        fieldnames = rows.fieldnames or []
        numeric_fields = [field for field in fieldnames if field != 'date']
        target = target or numeric_fields[-1]
        if target not in fieldnames:
            raise ValueError('target column not found: {}'.format(target))
        values = []
        for row in rows:
            try:
                values.append(float(row[target]))
            except (TypeError, ValueError):
                continue
            if limit and len(values) >= limit:
                break
    if len(values) < 2:
        raise ValueError('CSV must contain at least two numeric target values')
    return torch.tensor(values, dtype=torch.float32)


def make_rounds(values, client_count=2, rounds=2, seq_len=4, pred_len=1,
                batch_size=32, return_evaluation=False, standardize=False):
    if len(values) < client_count * (seq_len + pred_len):
        raise ValueError('dataset is too small for the requested clients and windows')
    streams = list(torch.tensor_split(values, client_count))
    statistics = []
    for stream in streams:
        reference_end = max(seq_len + pred_len,
                            len(stream) // rounds * 4 // 5)
        reference = stream[:reference_end]
        statistics.append((reference.mean(), reference.std().clamp_min(1e-6)))
    result = []
    evaluation = []
    for round_index in range(rounds):
        client_round = []
        client_test_round = []
        for stream in streams:
            start = 0 if round_index == 0 else round_index * len(stream) // rounds
            end = len(stream) if round_index == rounds - 1 else (round_index + 1) * len(stream) // rounds
            chunk = stream[start:end]
            if return_evaluation:
                required_length = max(
                    int((seq_len + pred_len) / 0.8 + 1),
                    int(pred_len / 0.2 + 1))
                if len(chunk) < required_length:
                    raise ValueError(
                        'each round chunk needs at least {} values for train/test windows; got {}'.format(
                            required_length, len(chunk)))
            if standardize:
                mean, std = statistics[len(client_round)]
                chunk = (chunk - mean) / std
            split = max(seq_len + pred_len, int(len(chunk) * 0.8))
            split = min(split, len(chunk))

            def windows(chunk_values, first_target, last_target):
                batches = []
                target_count = max(0, last_target - first_target)
                for batch_start in range(0, target_count, batch_size):
                    target_indices = range(
                        first_target + batch_start,
                        min(first_target + batch_start + batch_size, last_target))
                    inputs = torch.stack([
                        chunk_values[index - seq_len:index] for index in target_indices
                    ]).unsqueeze(-1)
                    targets = torch.stack([
                        chunk_values[index:index + pred_len] for index in target_indices
                    ]).unsqueeze(-1)
                    batches.append((inputs, targets))
                return batches

            client_round.append(windows(chunk, seq_len, split - pred_len + 1))
            test_start = max(seq_len, split)
            client_test_round.append(windows(chunk, test_start, len(chunk) - pred_len + 1))
        result.append(client_round)
        evaluation.append(client_test_round)
    if return_evaluation:
        return result, evaluation
    return result