"""Train FedMixer on chronological station-season NPZ tasks."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continual_federated import (ContinualClient, ContinualFedMixer,
                                 load_config, load_season_rounds)
from models.TimeMixer import Model


def make_model(seq_len, pred_len, channels):
    configs = SimpleNamespace(
        task_name='long_term_forecast', seq_len=seq_len, label_len=1,
        pred_len=pred_len, down_sampling_window=2, down_sampling_layers=1,
        channel_independence=1, e_layers=1, moving_avg=3, enc_in=channels,
        c_out=channels, use_future_temporal_feature=0, d_model=16,
        embed='timeF', freq='d', dropout=0.0, use_norm=1,
        decomp_method='moving_avg', d_ff=32, top_k=3, factor=1,
        activation='gelu', down_sampling_method='avg')
    return Model(configs).float()


def load_public_anchor(path):
    if path is None:
        return None
    with __import__('numpy').load(path, allow_pickle=False) as data:
        if 'anchor_x' not in data:
            raise ValueError(
                'public anchor NPZ must contain independently sourced anchor_x; '
                'train_x is not accepted as a public anchor')
        return torch.from_numpy(data['anchor_x'].astype('float32', copy=True))


def evaluate(model, test_rounds, seen_rounds, client_index, forward):
    errors = []
    with torch.no_grad():
        for round_index in range(seen_rounds + 1):
            batches = test_rounds[round_index]
            client_batches = batches[client_index]
            if client_batches is None:
                continue
            for inputs, targets in client_batches:
                predictions = forward(model, inputs)
                errors.append((
                    F.l1_loss(predictions, targets).item(),
                    F.mse_loss(predictions, targets).item()))
    if not errors:
        return {'mae': float('nan'), 'rmse': float('nan')}
    return {
        'mae': sum(value[0] for value in errors) / len(errors),
        'rmse': (sum(value[1] for value in errors) / len(errors)) ** 0.5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', default='./data/global_surface/processed')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--stations', nargs='*')
    parser.add_argument('--method', choices=['legacy', 'proposed'], default='proposed')
    parser.add_argument('--public-anchor-npz')
    parser.add_argument('--checkpoint')
    parser.add_argument('--config', default='./configs/proposed.yaml')
    parser.add_argument('--metrics-output',
                        default='./output/gsod_metrics.json')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.method == 'proposed' and config['method'] != 'proposed':
        raise ValueError('the selected config is not a proposed configuration')

    train_rounds, test_rounds, station_ids, metadata = load_season_rounds(
        args.processed_dir, args.batch_size, args.stations)
    first_batch = next(
        batch for round_batches in train_rounds
        for client_batches in round_batches if client_batches
        for batch in client_batches)
    seq_len = first_batch[0].shape[1]
    pred_len = first_batch[1].shape[1]
    channels = first_batch[0].shape[2]
    clients = []
    for _ in station_ids:
        model = make_model(seq_len, pred_len, channels)
        clients.append(ContinualClient(
            model, torch.optim.Adam(model.parameters(), lr=0.001),
            torch.device('cpu'), replay_enabled=args.method == 'legacy',
            memory_size=200, memory_batch_size=16))

    def forward(model, inputs):
        return model(inputs, None, None, None)

    public_inputs = load_public_anchor(args.public_anchor_npz)
    memory_config = dict(config['distribution_memory'])
    memory_config.pop('enabled', None)
    memory_config.pop('reference_encoder', None)
    trainer = ContinualFedMixer(
        clients, forward, public_inputs=public_inputs, method=args.method,
        generate_public_anchors=False,
        history_weight=config['distillation'].get('history_weight', 0.3),
        representation_weight=config['distillation'].get(
            'representation_weight', 0.1),
        history_enabled=config['distillation'].get('history', True),
        representation_enabled=config['distillation'].get('representation', True),
        mutual_enabled=config['distillation'].get('mutual', True),
        retention_enabled=config['distillation'].get('retention_weighting', True),
        strategy=config.get('strategy', 'fedmixer'),
        distribution_memory_kwargs=memory_config)
    results = []
    for round_index, round_batches in enumerate(train_rounds):
        history = trainer.fit([round_batches], start_round=round_index + 1)
        round_log = history[0]
        metrics = []
        for client_index, client in enumerate(clients):
            metrics.append(evaluate(
                client.model, test_rounds, round_index, client_index, forward))
        result = {
            'round': round_index + 1,
            'season_start': metadata[round_index]['season_start'],
            'active_clients': metadata[round_index]['active_clients'],
            'metrics': metrics,
            'distribution_bytes': [
                memory.byte_count for memory in trainer.distribution_memories],
        }
        results.append(result)
        print('round={} season_start={} active={} metrics={} seasonal_drift={} '
              'trend_drift={} reclustered={} retrievals={} loss_components={} '
              'memory_bytes={}'.format(
            round_index + 1, metadata[round_index]['season_start'],
            metadata[round_index]['active_clients'], metrics,
            round_log['seasonal_drift'], round_log['trend_drift'],
            round_log['reclustered'], round_log['retrievals'],
            round_log['loss_components'], round_log['memory_bytes']))

    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.metrics_output).open('w', encoding='utf-8') as handle:
        json.dump({
            'method': args.method,
            'processed_dir': str(Path(args.processed_dir)),
            'station_ids': station_ids,
            'rounds': results,
        }, handle, indent=2)
    print('metrics saved to {}'.format(args.metrics_output))

    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'schema_version': 1,
            'station_ids': station_ids,
            'metadata': metadata,
            'trainer': trainer.state_dict(),
            'clients': [client.model.state_dict() for client in clients],
        }
        torch.save(checkpoint, args.checkpoint)
        print('checkpoint saved to {}'.format(args.checkpoint))


if __name__ == '__main__':
    main()
