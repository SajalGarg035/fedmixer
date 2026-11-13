"""Controlled synthetic validation for the federated continual-learning methods."""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continual_federated import ContinualClient, ContinualFedMixer, FedMixerServer


class TinyForecast(nn.Module):
    """Small forecaster used only to validate the synthetic FCL mechanism."""

    def __init__(self, sequence_length=8, prediction_length=4, hidden=16):
        super().__init__()
        self.encoder = nn.Linear(sequence_length, hidden)
        self.head = nn.Linear(hidden, prediction_length)

    def encode(self, inputs):
        return torch.tanh(self.encoder(inputs.squeeze(-1)))

    def forward_with_representation(self, inputs):
        representation = self.encode(inputs)
        prediction = self.head(representation).unsqueeze(-1)
        return prediction, representation

    def forward(self, inputs):
        return self.forward_with_representation(inputs)[0]


def regime_value(time_index, gradual_mix):
    low = torch.sin(2 * torch.pi * time_index / 8.0)
    high = torch.sin(2 * torch.pi * time_index / 4.0)
    return (1 - gradual_mix) * low + gradual_mix * high


def make_task_batch(seed, task_index, client_index, training=True,
                    batch_size=16, batch_index=0):
    generator = torch.Generator().manual_seed(
        seed * 100_003 + task_index * 1_009 + client_index * 37 +
        int(training) + batch_index * 10_007)
    starts = torch.rand(batch_size, generator=generator) * 24.0
    client_phase = client_index * 0.19
    if task_index <= 0:
        gradual_mix = 0.0
    elif task_index <= 3:
        gradual_mix = (0.18, 0.38, 0.68)[task_index - 1]
    elif task_index == 4:
        gradual_mix = 1.0
    else:
        gradual_mix = 0.0  # recurring stable regime
    gain = 0.75 + 0.12 * client_index
    t_x = starts[:, None] + torch.arange(8).float()[None, :] + client_phase
    t_y = starts[:, None] + torch.arange(8, 12).float()[None, :] + client_phase
    x = gain * regime_value(t_x, gradual_mix)
    y = gain * regime_value(t_y, gradual_mix)
    if not training:
        x = x + 0.01 * torch.randn(x.shape, generator=generator)
        y = y + 0.01 * torch.randn(y.shape, generator=generator)
    return x.unsqueeze(-1), y.unsqueeze(-1)


def make_tasks(seed, clients=4, rounds=6):
    training_rounds = []
    evaluation_rounds = []
    for task_index in range(rounds):
        train_clients = []
        test_clients = []
        for client_index in range(clients):
            # One missing client-task combination; evaluation uses the same mask.
            if task_index == 3 and client_index == clients - 1:
                train_clients.append(None)
                test_clients.append(None)
            else:
                # Multiple distinct local batches let the frozen previous-round
                # teacher and synthetic-probe targets exert a gradient after the
                # student's first local optimizer update.
                train_clients.append([
                    make_task_batch(seed, task_index, client_index, True,
                                    batch_index=batch_index)
                    for batch_index in range(2)])
                test_clients.append([
                    make_task_batch(seed, task_index, client_index, False)])
        training_rounds.append(train_clients)
        evaluation_rounds.append(test_clients)
    return training_rounds, evaluation_rounds


def run_method(seed, training_rounds, evaluation_rounds, method,
               strategy, continual, options=None):
    random.seed(seed)
    torch.manual_seed(seed)
    clients = []
    for _ in range(len(training_rounds[0])):
        model = TinyForecast()
        clients.append(ContinualClient(
            model, torch.optim.Adam(model.parameters(), lr=0.01),
            torch.device('cpu'), memory_size=32, memory_batch_size=8,
            memory_weight=0.1, distill_weight=0.1,
            replay_enabled=method == 'legacy'))

    trainer = ContinualFedMixer(
        clients, lambda model, inputs: model(inputs), method=method,
        strategy=strategy, server=FedMixerServer(random_state=seed),
        **(options or {}))
    errors = [
        [[None for _ in evaluation_rounds] for _ in clients]
        for _ in training_rounds]
    started = time.perf_counter()
    loss_history = []
    for trained_round, client_batches in enumerate(training_rounds):
        loss_history.extend(trainer.fit(
            [client_batches], continual=continual,
            start_round=trained_round + 1))
        for client_index, client in enumerate(clients):
            client.model.eval()
            with torch.no_grad():
                for task_index in range(trained_round + 1):
                    batches = evaluation_rounds[task_index][client_index]
                    if batches is None:
                        continue
                    values = []
                    for inputs, targets in batches:
                        prediction = client.model(inputs)
                        values.append((F.mse_loss(prediction, targets).item(),
                                       F.l1_loss(prediction, targets).item()))
                    if values:
                        errors[trained_round][client_index][task_index] = (
                            statistics.fmean(value[0] for value in values),
                            statistics.fmean(value[1] for value in values))

    diagonal = [errors[task][client][task]
                for task in range(len(training_rounds))
                for client in range(len(clients))
                if errors[task][client][task] is not None]
    forgetting = []
    for client_index in range(len(clients)):
        for task_index in range(len(training_rounds) - 1):
            first = errors[task_index][client_index][task_index]
            last = errors[-1][client_index][task_index]
            if first is not None and last is not None:
                forgetting.append((last[0] - first[0], last[1] - first[1]))
    elapsed = time.perf_counter() - started
    memory = trainer._memory_accounting()
    component_names = ('forecast', 'history', 'representation', 'mutual',
                       'weighted_history', 'weighted_representation',
                       'weighted_mutual')
    component_values = {
        name: [component[name] for item in loss_history
               for component in item['loss_components']]
        for name in component_names
    }
    return {
        'seed': seed,
        'ACE_MSE': statistics.fmean(value[0] for value in diagonal),
        'ACE_MAE': statistics.fmean(value[1] for value in diagonal),
        'ACE_RMSE': statistics.fmean(value[0] for value in diagonal) ** 0.5,
        'forgetting_MSE': statistics.fmean(value[0] for value in forgetting)
                          if forgetting else 0.0,
        'forgetting_MAE': statistics.fmean(value[1] for value in forgetting)
                          if forgetting else 0.0,
        'BWT_MSE': -statistics.fmean(value[0] for value in forgetting)
                   if forgetting else 0.0,
        'BWT_MAE': -statistics.fmean(value[1] for value in forgetting)
                   if forgetting else 0.0,
        'runtime_seconds': elapsed,
        'mean_loss_components': {
            name: statistics.fmean(values) if values else 0.0
            for name, values in component_values.items()},
        'memory': memory,
        'distribution_entries': [len(memory) for memory in trainer.distribution_memories],
        'raw_replay_entries': [len(client.memory) if client.memory is not None else 0
                               for client in clients],
        'missing_task': {'round_index': 3, 'client_index': len(clients) - 1},
    }


def summarize(results):
    keys = ('ACE_MSE', 'ACE_MAE', 'ACE_RMSE', 'forgetting_MSE',
            'forgetting_MAE', 'BWT_MSE', 'BWT_MAE', 'runtime_seconds')
    summary = {}
    for key in keys:
        values = [result[key] for result in results]
        summary[key] = {
            'mean': statistics.fmean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument('--output', default='./output/synthetic_fcl_results.json')
    args = parser.parse_args()
    methods = [
        ('FedAvg', 'legacy', 'fedavg', False, {'mutual_enabled': False}),
        ('FedAvg-FCL', 'legacy', 'fedavg', True, {'mutual_enabled': False}),
        ('FedMixer-FCL', 'legacy', 'fedmixer', True, {}),
        ('Proposed', 'proposed', 'fedmixer', True, {}),
        ('Proposed: no history distillation', 'proposed', 'fedmixer', True,
         {'history_enabled': False}),
        ('Proposed: no representation loss', 'proposed', 'fedmixer', True,
         {'representation_enabled': False}),
        ('Proposed: no mutual distillation', 'proposed', 'fedmixer', True,
         {'mutual_enabled': False}),
        ('Proposed: no retention weighting', 'proposed', 'fedmixer', True,
         {'retention_enabled': False}),
    ]
    data_by_seed = {seed: make_tasks(seed) for seed in args.seeds}
    results = {}
    for name, method, strategy, continual, options in methods:
        individual = []
        for seed in args.seeds:
            train, test = data_by_seed[seed]
            individual.append(run_method(seed, train, test, method,
                                         strategy, continual, options))
        results[name] = {'seeds': individual, 'aggregate': summarize(individual)}

    payload = {
        'experiment': 'controlled synthetic time-series FCL; not a real-world benchmark',
        'scenarios': [
            'stable regime', 'gradual seasonal mixture shift',
            'abrupt high-frequency shift', 'recurring stable regime',
            'heterogeneous client phase and gain', 'one missing client-task pair'],
        'seeds': args.seeds,
        'client_count': 4,
        'round_count': 6,
        'results': results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    for name, result in results.items():
        print('{}: {}'.format(name, result['aggregate']))
    print('synthetic results saved to {}'.format(path))


if __name__ == '__main__':
    main()
