"""Compare FedMixer with and without continual replay/drift on Electricity clients."""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continual_federated import ContinualClient, ContinualFedMixer, FedMixerServer
from continual_federated.csv_data import load_csv_series, make_rounds
from models.TimeMixer import Model


def make_model(device):
    configs = SimpleNamespace(
        task_name='long_term_forecast', seq_len=96, label_len=48, pred_len=96,
        down_sampling_window=2, down_sampling_layers=1, channel_independence=1,
        e_layers=2, moving_avg=25, enc_in=1, c_out=1, use_future_temporal_feature=0,
        d_model=16, embed='timeF', freq='h', dropout=0.1, use_norm=1,
        decomp_method='moving_avg', d_ff=32, top_k=5, factor=1,
        activation='gelu', down_sampling_method='avg',
    )
    return Model(configs).float().to(device)


def build_rounds(data_dir, client_count, rounds, limit):
    client_batches = []
    for client_index in range(client_count):
        path = Path(data_dir) / 'clients' / 'client_{:03d}.csv'.format(client_index)
        full_values = load_csv_series(path, target='load')
        if full_values.std() <= 1e-6:
            raise ValueError(
                '{} is constant (std={:.3g}); regenerate the dataset with --max-rows 0'.format(
                    path, full_values.std().item()))
        values = full_values[:limit] if limit else full_values
        if values.std() <= 1e-6:
            raise ValueError(
                '--limit {} selects a constant prefix of {}; use --limit 0 for the full timeline'.format(
                    limit, path))
        train_rounds, test_rounds = make_rounds(
            values, 1, rounds, 96, 96, batch_size=32,
            return_evaluation=True, standardize=True)
        client_batches.append((train_rounds, test_rounds))
    train_batches = [[client_batches[index][0][round_index][0]
                      for index in range(client_count)] for round_index in range(rounds)]
    test_batches = [[client_batches[index][1][round_index][0]
                     for index in range(client_count)] for round_index in range(rounds)]
    return train_batches, test_batches


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def comparison_variants(mode, include_ablations=False):
    variants = {
        'baseline': [('FedAvg', False, 'legacy', 'fedavg', {})],
        'continual': [('FedMixer-FCL', True, 'legacy', 'fedmixer', {})],
        'both': [('FedMixer baseline', False, 'legacy', 'fedmixer', {}),
                 ('FedMixer-FCL', True, 'legacy', 'fedmixer', {})],
        'all': [
            ('FedAvg', False, 'legacy', 'fedavg', {}),
            ('FedAvg-FCL', True, 'legacy', 'fedavg', {}),
            ('FedMixer-FCL', True, 'legacy', 'fedmixer', {}),
            ('FedMixer-DistributionMemory (Proposed)', True,
             'proposed', 'fedmixer', {}),
        ],
        'ablation': [
            ('Proposed: full', True, 'proposed', 'fedmixer', {}),
            ('Proposed: no history distillation', True, 'proposed', 'fedmixer',
             {'history_enabled': False}),
            ('Proposed: no representation loss', True, 'proposed', 'fedmixer',
             {'representation_enabled': False}),
            ('Proposed: no mutual distillation', True, 'proposed', 'fedmixer',
             {'mutual_enabled': False}),
            ('Proposed: no retention weighting', True, 'proposed', 'fedmixer',
             {'retention_enabled': False}),
        ],
    }
    selected = list(variants[mode])
    if mode == 'all' and include_ablations:
        selected.extend(variants['ablation'][1:])
    return selected


def run_variant(round_batches, test_rounds, device, continual, seed,
                method='legacy', strategy='fedmixer', trainer_options=None):
    seed_everything(seed)
    clients = []
    for _ in round_batches[0]:
        model = make_model(device)
        clients.append(ContinualClient(
            model, torch.optim.Adam(model.parameters(), lr=0.001), device,
            memory_size=200, memory_batch_size=16,
            memory_weight=0.1, distill_weight=0.1, drift_threshold=1.0))

    def forward(model, inputs):
        return model(inputs, None, None, None)

    trainer = ContinualFedMixer(
        clients, forward, method=method, strategy=strategy,
        server=FedMixerServer(random_state=seed),
        **(trainer_options or {}))
    history = []
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    error_matrix = [[{} for _ in test_rounds] for _ in clients]
    for round_index in range(len(round_batches)):
        history.extend(trainer.fit(
            [round_batches[round_index]], continual=continual,
            start_round=round_index + 1))
        with torch.no_grad():
            for client_index, client in enumerate(clients):
                client.model.eval()
                for test_index in range(round_index + 1):
                    batches = test_rounds[test_index][client_index]
                    if batches is None:
                        error_matrix[client_index][round_index][test_index] = None
                        continue
                    batch_errors = [
                        (F.mse_loss(predictions, targets.to(device)).item(),
                         F.l1_loss(predictions, targets.to(device)).item())
                        for inputs, targets in batches
                        for predictions in [forward(client.model, inputs.to(device))]
                    ]
                    error_matrix[client_index][round_index][test_index] = (
                        (sum(error[0] for error in batch_errors) / len(batch_errors),
                         sum(error[1] for error in batch_errors) / len(batch_errors))
                        if batch_errors else None)
    diagonal = [error_matrix[client][index][index]
                for client in range(len(clients)) for index in range(len(test_rounds))
                if error_matrix[client][index][index] is not None]
    if not diagonal:
        raise ValueError('no chronological evaluation windows were available')
    forgetting_differences = {'mse': [], 'rmse': [], 'mae': []}
    for client in range(len(clients)):
        for task_index in range(max(0, len(test_rounds) - 1)):
            initial = error_matrix[client][task_index][task_index]
            final = error_matrix[client][-1][task_index]
            if initial is None or final is None:
                continue
            forgetting_differences['mse'].append(final[0] - initial[0])
            forgetting_differences['rmse'].append(
                final[0] ** 0.5 - initial[0] ** 0.5)
            forgetting_differences['mae'].append(final[1] - initial[1])
    ace_mse = statistics.fmean(error[0] for error in diagonal)
    ace_mae = statistics.fmean(error[1] for error in diagonal)
    forgetting_mse = statistics.fmean(forgetting_differences['mse']) \
        if forgetting_differences['mse'] else 0.0
    forgetting_mae = statistics.fmean(forgetting_differences['mae']) \
        if forgetting_differences['mae'] else 0.0
    forgetting_rmse = statistics.fmean(forgetting_differences['rmse']) \
        if forgetting_differences['rmse'] else 0.0
    round_mse = []
    for round_index in range(len(test_rounds)):
        values = [error_matrix[client][round_index][round_index][0]
                  for client in range(len(clients))
                  if error_matrix[client][round_index][round_index] is not None]
        round_mse.append(statistics.fmean(values) if values else None)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started
    peak_cuda_bytes = (torch.cuda.max_memory_allocated(device)
                       if device.type == 'cuda' else None)
    memory = history[-1]['memory_bytes'] if history else {}
    return {
        'seed': seed,
        'ACE_MAE': ace_mae,
        'ACE_RMSE': ace_mse ** 0.5,
        'ACE_MSE': ace_mse,
        'forgetting_MAE': forgetting_mae,
        'forgetting_RMSE': forgetting_rmse,
        'forgetting_MSE': forgetting_mse,
        'BWT_MAE': -forgetting_mae,
        'BWT_RMSE': -forgetting_rmse,
        'BWT_MSE': -forgetting_mse,
        'round_MSE': round_mse,
        'runtime_seconds': runtime_seconds,
        'peak_cuda_bytes': peak_cuda_bytes,
        'memory': memory,
        'final_history': history[-1] if history else None,
    }


def aggregate_seed_results(results):
    scalar_keys = (
        'ACE_MAE', 'ACE_RMSE', 'ACE_MSE', 'forgetting_MAE', 'forgetting_RMSE',
        'forgetting_MSE', 'BWT_MAE', 'BWT_RMSE', 'BWT_MSE', 'runtime_seconds')
    successes = [result for result in results if 'error' not in result]
    aggregate = {'successful_seeds': len(successes),
                 'failed_seeds': [result for result in results if 'error' in result]}
    for key in scalar_keys:
        values = [float(result[key]) for result in successes]
        aggregate[key] = {
            'mean': statistics.fmean(values) if values else None,
            'std': statistics.stdev(values) if len(values) > 1 else
                   (0.0 if values else None),
        }
    for key in ('total_persistent_bytes', 'algorithmic_memory_bytes'):
        values = [result['memory'].get(key) for result in successes]
        values = [float(value) for value in values if value is not None]
        aggregate[key] = {
            'mean': statistics.fmean(values) if values else None,
            'std': statistics.stdev(values) if len(values) > 1 else
                   (0.0 if values else None),
        }
    peak_values = [result['peak_cuda_bytes'] for result in successes
                   if result['peak_cuda_bytes'] is not None]
    aggregate['peak_cuda_bytes'] = {
        'mean': statistics.fmean(peak_values) if peak_values else None,
        'std': statistics.stdev(peak_values) if len(peak_values) > 1 else
               (0.0 if peak_values else None),
    }
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data/electricity_federated')
    parser.add_argument('--clients', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0,
                        help='rows per client; use 0 for the full prepared dataset')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode', choices=[
        'baseline', 'continual', 'both', 'all', 'ablation'], default='all')
    parser.add_argument('--include-ablations', action='store_true',
                        help='append proposed loss/retention ablations to --mode all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument('--output', default='./output/output.log')
    parser.add_argument('--results-json', default='./output/fedmixer_comparison.json')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('CUDA requested but unavailable in this PyTorch environment; using CPU.')
        device = torch.device('cpu')
    round_batches, test_rounds = build_rounds(args.data_dir, args.clients, args.rounds, args.limit)

    if not args.seeds:
        parser.error('provide at least one seed')
    variants = comparison_variants(args.mode, args.include_ablations)
    individual_results = {}
    output_lines = []
    for name, continual, method, strategy, trainer_options in variants:
        seed_results = []
        for seed in args.seeds:
            try:
                result = run_variant(
                    round_batches, test_rounds, device, continual, seed=seed,
                    method=method, strategy=strategy,
                    trainer_options=trainer_options)
                seed_results.append(result)
                output_lines.append(
                    '{} seed={}: ACE_MAE={:.6f} ACE_RMSE={:.6f} '
                    'forgetting_MAE={:.6f} BWT_MAE={:.6f} runtime={:.2f}s '
                    'algorithmic_memory_bytes={} total_persistent_bytes={} '
                    'peak_cuda_bytes={}'.format(
                        name, seed, result['ACE_MAE'], result['ACE_RMSE'],
                        result['forgetting_MAE'], result['BWT_MAE'],
                        result['runtime_seconds'],
                        result['memory'].get('algorithmic_memory_bytes'),
                        result['memory'].get('total_persistent_bytes'),
                        result['peak_cuda_bytes']))
            except Exception as error:  # keep the failed seed visible in saved results
                failure = {'seed': seed, 'error': '{}: {}'.format(
                    type(error).__name__, error)}
                seed_results.append(failure)
                output_lines.append('{} seed={} FAILED: {}'.format(
                    name, seed, failure['error']))
        individual_results[name] = {
            'seeds': seed_results,
            'aggregate': aggregate_seed_results(seed_results),
            'settings': {'method': method, 'strategy': strategy,
                         'continual': continual, **trainer_options},
        }
        aggregate = individual_results[name]['aggregate']
        mae = aggregate['ACE_MAE']
        rmse = aggregate['ACE_RMSE']
        output_lines.append(
            '{} aggregate (n={} successful, {} failed): '
            'ACE_MAE={:.6f}±{:.6f} ACE_RMSE={:.6f}±{:.6f} '
            'forgetting_MAE={:.6f}±{:.6f} BWT_MAE={:.6f}±{:.6f}'.format(
                name, aggregate['successful_seeds'],
                len(aggregate['failed_seeds']),
                mae['mean'] if mae['mean'] is not None else float('nan'),
                mae['std'] if mae['std'] is not None else float('nan'),
                rmse['mean'] if rmse['mean'] is not None else float('nan'),
                rmse['std'] if rmse['std'] is not None else float('nan'),
                aggregate['forgetting_MAE']['mean'] or 0.0,
                aggregate['forgetting_MAE']['std'] or 0.0,
                aggregate['BWT_MAE']['mean'] or 0.0,
                aggregate['BWT_MAE']['std'] or 0.0))

    payload = {
        'dataset': 'Electricity' if 'electricity' in args.data_dir.lower()
                   else str(args.data_dir),
        'mode': args.mode,
        'seeds': args.seeds,
        'client_count': args.clients,
        'rounds': args.rounds,
        'limit': args.limit,
        'device': str(device),
        'timing_method': 'wall clock across training and chronological evaluation; CUDA synchronized',
        'memory_method': 'persistent tensor storage separated into algorithmic and total; CUDA peak allocation reported separately',
        'results': individual_results,
    }
    os.makedirs(os.path.dirname(args.results_json) or '.', exist_ok=True)
    with open(args.results_json, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'a', encoding='utf-8') as handle:
        handle.write('\n=== mode: {} ===\n'.format(args.mode))
        handle.write('\n'.join(output_lines) + '\n')
    print('\n'.join(output_lines))
    print('text results saved to {}'.format(args.output))
    print('individual and aggregate seed results saved to {}'.format(args.results_json))


if __name__ == '__main__':
    main()
