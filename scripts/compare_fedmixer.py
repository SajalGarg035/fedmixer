"""Compare FedMixer with and without continual replay/drift on Electricity clients."""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continual_federated import ContinualClient, ContinualFedMixer
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
        values = load_csv_series(path, target='load', limit=limit)
        train_rounds, test_rounds = make_rounds(
            values, 1, rounds, 96, 96, batch_size=32,
            return_evaluation=True, standardize=True)
        client_batches.append((train_rounds, test_rounds))
    train_batches = [[client_batches[index][0][round_index]
                      for index in range(client_count)] for round_index in range(rounds)]
    test_batches = [[client_batches[index][1][round_index]
                     for index in range(client_count)] for round_index in range(rounds)]
    return train_batches, test_batches


def run_variant(round_batches, test_rounds, device, continual, seed):
    torch.manual_seed(seed)
    clients = []
    for _ in round_batches[0]:
        model = make_model(device)
        clients.append(ContinualClient(
            model, torch.optim.Adam(model.parameters(), lr=0.001), device,
            memory_size=200, memory_batch_size=16,
            memory_weight=0.1, distill_weight=0.1, drift_threshold=1.0))

    def forward(model, inputs):
        return model(inputs, None, None, None)

    trainer = ContinualFedMixer(clients, forward)
    history = []
    error_matrix = [[{} for _ in test_rounds] for _ in clients]
    for round_index in range(len(round_batches)):
        history.extend(trainer.fit(
            [round_batches[round_index]], continual=continual,
            start_round=round_index + 1))
        with torch.no_grad():
            for client_index, client in enumerate(clients):
                for test_index in range(round_index + 1):
                    batches = test_rounds[test_index][client_index]
                    errors = [
                        F.mse_loss(forward(client.model, inputs.to(device)), targets.to(device)).item()
                        for inputs, targets in batches
                    ]
                    error_matrix[client_index][round_index][test_index] = (
                        sum(errors) / len(errors) if errors else float('nan'))
    diagonal = [error_matrix[client][index][index]
                for client in range(len(clients)) for index in range(len(test_rounds))]
    final_errors = [error_matrix[client][-1][index]
                    for client in range(len(clients)) for index in range(len(test_rounds) - 1)]
    historical_diagonal = [error_matrix[client][index][index]
                           for client in range(len(clients))
                           for index in range(len(test_rounds) - 1)]
    ace = sum(diagonal) / len(diagonal)
    bwt = (sum(final_errors) / len(final_errors)
           - sum(historical_diagonal) / len(historical_diagonal)) if historical_diagonal else 0.0
    return history, ace, bwt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data/electricity_federated')
    parser.add_argument('--clients', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--limit', type=int, default=4000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode', choices=['baseline', 'continual', 'both'], default='both')
    parser.add_argument('--output', default='./output/output.log')
    args = parser.parse_args()
    device = torch.device(args.device)
    round_batches, test_rounds = build_rounds(args.data_dir, args.clients, args.rounds, args.limit)

    variants = {
        'baseline': [('FedMixer baseline', False)],
        'continual': [('Continual FedMixer', True)],
        'both': [('FedMixer baseline', False), ('Continual FedMixer', True)],
    }
    output_lines = []
    for name, continual in variants[args.mode]:
        history, ace, bwt = run_variant(round_batches, test_rounds, device, continual, seed=2021)
        output_lines.append('{}: ACE_MSE={:.6f} BWT={:.6f} final_clusters={} memory={}'.format(
            name, ace, bwt, history[-1]['clusters'], history[-1]['memory']))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'a', encoding='utf-8') as handle:
        handle.write('\n=== mode: {} ===\n'.format(args.mode))
        handle.write('\n'.join(output_lines) + '\n')
    print('\n'.join(output_lines))
    print('results saved to {}'.format(args.output))


if __name__ == '__main__':
    main()