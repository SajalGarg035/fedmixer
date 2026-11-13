import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from continual_federated import (ContinualClient, ContinualFedMixer,
                                 DistributionMemory, load_season_rounds)


class RepresentedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs):
        return self.linear(inputs)

    def encode(self, inputs):
        return self.linear(inputs).reshape(inputs.shape[0], -1)

    def forward_with_representation(self, inputs):
        prediction = self.forward(inputs)
        return prediction, self.encode(inputs)


class ReplayFreeDistributionTests(unittest.TestCase):
    def test_data_fcl_loader_preserves_chronology_and_missing_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'windows' / 'station-a').mkdir(parents=True)
            (root / 'windows' / 'station-b').mkdir(parents=True)
            manifest = [
                {'status': 'included', 'station_id': 'station-a',
                 'season_id': 'a1', 'season_start': '2020-01-01'},
                {'status': 'included', 'station_id': 'station-a',
                 'season_id': 'a2', 'season_start': '2020-04-01'},
                {'status': 'included', 'station_id': 'station-b',
                 'season_id': 'b2', 'season_start': '2020-04-01'},
            ]
            (root / 'manifest.json').write_text(json.dumps(manifest),
                                                encoding='utf-8')
            for station, season, value in (
                    ('station-a', 'a1', 1.0), ('station-a', 'a2', 2.0),
                    ('station-b', 'b2', 3.0)):
                np.savez(
                    root / 'windows' / station / (season + '.npz'),
                    train_x=np.full((2, 4, 1), value, dtype=np.float32),
                    train_y=np.full((2, 1, 1), value, dtype=np.float32),
                    test_x=np.full((1, 4, 1), value, dtype=np.float32),
                    test_y=np.full((1, 1, 1), value, dtype=np.float32),
                    season_start='2020-01-01' if season == 'a1' else '2020-04-01',
                    season_id=season)
            train, test, stations, metadata = load_season_rounds(root, batch_size=2)
            self.assertEqual(stations, ['station-a', 'station-b'])
            self.assertEqual([item['season_start'] for item in metadata],
                             ['2020-01-01', '2020-04-01'])
            self.assertIsNone(train[0][1])
            self.assertIsNotNone(test[1][1])

    def test_distribution_memory_statistics_retrieval_and_round_trip(self):
        memory = DistributionMemory(max_entries=2)
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        summary = memory.summarize(values)
        self.assertTrue(torch.allclose(summary['mean'], torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.allclose(
            summary['log_variance'].exp(), torch.tensor([1.0, 1.0])))
        memory.add('task-1', summary, temporal_signature=torch.ones(2))
        retrieved = memory.retrieve(summary, top_k=1)
        self.assertEqual(retrieved[0]['entry']['task_id'], 'task-1')
        self.assertAlmostEqual(retrieved[0]['weight'], 1.0)
        restored = DistributionMemory(max_entries=2)
        restored.load_state_dict(memory.state_dict())
        self.assertEqual(restored.entries[0]['task_id'], 'task-1')
        self.assertEqual(restored.byte_count, memory.byte_count)
        self.assertNotIn('x', restored.entries[0])
        self.assertNotIn('y', restored.entries[0])

    def test_proposed_mode_has_no_raw_replay_and_freezes_reference(self):
        torch.manual_seed(4)
        clients = []
        for _ in range(2):
            model = RepresentedLinear()
            clients.append(ContinualClient(
                model, torch.optim.SGD(model.parameters(), lr=0.01),
                torch.device('cpu'), replay_enabled=False))

        def forward(model, inputs):
            return model(inputs)

        client_batches = []
        for offset in (0.0, 1.0):
            inputs = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1) + offset
            targets = inputs * 0.5
            client_batches.append([(inputs, targets)])
        trainer = ContinualFedMixer(
            clients, forward, method='proposed',
            distribution_memory_kwargs={'max_entries': 2})
        history = trainer.fit([client_batches, client_batches])

        self.assertEqual([len(client.memory or []) for client in clients], [0, 0])
        self.assertEqual([len(memory) for memory in trainer.distribution_memories], [2, 2])
        self.assertIsNotNone(clients[0].previous_teacher)
        reference_before = [
            parameter.detach().clone()
            for parameter in trainer.reference_encoder.parameters()]
        trainer.fit([client_batches])
        for before, parameter in zip(reference_before,
                                     trainer.reference_encoder.parameters()):
            self.assertTrue(torch.equal(before, parameter.detach()))
        self.assertTrue(all(torch.isfinite(torch.tensor(item['total']))
                            for item in history[-1]['loss_components']))

    def test_inactive_station_is_excluded_from_aggregation(self):
        clients = []
        for _ in range(2):
            model = RepresentedLinear()
            clients.append(ContinualClient(
                model, torch.optim.SGD(model.parameters(), lr=0.01),
                torch.device('cpu'), replay_enabled=False))

        def forward(model, inputs):
            return model(inputs)

        inputs = torch.ones(4, 1, 1)
        batches = [[(inputs, inputs)], None]
        trainer = ContinualFedMixer(
            clients, forward, method='proposed',
            generate_public_anchors=False)
        history = trainer.fit([batches])
        self.assertEqual(history[0]['clusters'], [[0]])
        self.assertEqual(len(trainer.distribution_memories[0]), 1)
        self.assertEqual(len(trainer.distribution_memories[1]), 0)


if __name__ == '__main__':
    unittest.main()
