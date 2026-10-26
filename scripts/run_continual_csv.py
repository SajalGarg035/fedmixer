"""Run compact continual FedMixer training on any numeric CSV target column."""

import argparse
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from continual_federated import ContinualClient, ContinualFedMixer
from continual_federated.csv_data import load_csv_series, make_rounds
from models.TimeMixer import Model


def make_model():
    configs = SimpleNamespace(
        task_name='long_term_forecast', seq_len=4, label_len=1, pred_len=1,
        down_sampling_window=1, down_sampling_layers=0, channel_independence=1,
        e_layers=1, moving_avg=3, enc_in=1, c_out=1, use_future_temporal_feature=0,
        d_model=8, embed='timeF', freq='h', dropout=0.0, use_norm=1,
        decomp_method='moving_avg', d_ff=16, top_k=3, factor=1,
        activation='gelu', down_sampling_method='avg',
    )
    return Model(configs).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='./data/ETT/ETTh1_tiny.csv')
    parser.add_argument('--target', default='OT')
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--clients', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=2)
    args = parser.parse_args()

    values = load_csv_series(args.csv, args.target, args.limit)
    round_batches = make_rounds(values, args.clients, args.rounds)
    clients = []
    for _ in range(args.clients):
        model = make_model()
        clients.append(ContinualClient(
            model, torch.optim.SGD(model.parameters(), lr=0.01),
            torch.device('cpu'), memory_size=32, memory_batch_size=4,
            memory_weight=0.2, distill_weight=0.1, drift_threshold=0.5))

    def forward(model, inputs):
        return model(inputs, None, None, None)

    trainer = ContinualFedMixer(clients, forward)
    for result in trainer.fit(round_batches):
        print('round={} clusters={} losses={} drifted={} memory={}'.format(
            result['round'], result['clusters'],
            [round(loss, 5) for loss in result['losses']],
            result['drifted'], result['memory']))


if __name__ == '__main__':
    main()