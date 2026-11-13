import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from continual_federated import (ContinualClient, ContinualFedMixer,
                                 DistributionMemory, load_season_rounds,
                                 MultiscaleSpectralTrendDetector)
from models.TimeMixer import Model
from scripts.compare_fedmixer import aggregate_seed_results, comparison_variants


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


class RecordingLinear(RepresentedLinear):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, inputs):
        self.seen.append(inputs.detach().clone())
        return super().forward(inputs)


class ReplayFreeDistributionTests(unittest.TestCase):
    def test_comparison_ablation_variants_and_seed_aggregation(self):
        variants = comparison_variants('ablation')
        self.assertEqual(len(variants), 5)
        options = {name: trainer_options for name, _, _, _, trainer_options in variants}
        self.assertFalse(options['Proposed: no history distillation']['history_enabled'])
        self.assertFalse(options['Proposed: no representation loss']['representation_enabled'])
        self.assertFalse(options['Proposed: no mutual distillation']['mutual_enabled'])
        self.assertFalse(options['Proposed: no retention weighting']['retention_enabled'])
        aggregate = aggregate_seed_results([
            {'seed': 1, 'ACE_MAE': 1.0, 'ACE_RMSE': 2.0, 'ACE_MSE': 4.0,
             'forgetting_MAE': 0.1, 'forgetting_RMSE': 0.2,
             'forgetting_MSE': 0.4, 'BWT_MAE': -0.1, 'BWT_RMSE': -0.2,
             'BWT_MSE': -0.4, 'runtime_seconds': 3.0,
             'memory': {'total_persistent_bytes': 10,
                        'algorithmic_memory_bytes': 4},
             'peak_cuda_bytes': None},
            {'seed': 2, 'error': 'synthetic failure'},
        ])
        self.assertEqual(aggregate['successful_seeds'], 1)
        self.assertEqual(aggregate['failed_seeds'][0]['seed'], 2)
        self.assertEqual(aggregate['ACE_MAE'], {'mean': 1.0, 'std': 0.0})

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

    def test_distribution_memory_merges_recurring_regimes_and_preserves_rare(self):
        memory = DistributionMemory(max_entries=2, merge_similarity=0.99)
        regime_a = memory.summarize(torch.tensor([[0.0, 0.0], [0.01, 0.0]]))
        regime_b = memory.summarize(torch.tensor([[5.0, 4.0], [5.1, 4.0]]))
        memory.add('A-first', regime_a, round=1)
        memory.add('B-rare', regime_b, round=2)
        memory.add('A-return', regime_a, round=5)
        self.assertEqual(len(memory), 2)
        a_entry = next(entry for entry in memory.entries if entry['task_id'] == 'A-first')
        b_entry = next(entry for entry in memory.entries if entry['task_id'] == 'B-rare')
        self.assertEqual(a_entry['occurrences'], 2)
        self.assertEqual(a_entry['last_task_id'], 'A-return')
        self.assertEqual(b_entry['occurrences'], 1)
        self.assertEqual(a_entry['sample_count'], 4)

    def test_distribution_memory_rejects_invalid_summaries_and_handles_empty_or_tiny_scores(self):
        memory = DistributionMemory(max_entries=2)
        self.assertEqual(memory.retrieve({'mean': torch.ones(2),
                                          'log_variance': torch.zeros(2)}), [])
        with self.assertRaises(ValueError):
            memory.summarize(torch.tensor([[float('nan'), 0.0]]))
        with self.assertRaises(ValueError):
            memory.add('bad', {'mean': torch.ones(2),
                               'log_variance': torch.zeros(2),
                               'sample_count': 1}, input_window=torch.ones(1))
        summary = memory.summarize(torch.zeros(2, 2))
        memory.add('one', summary)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            memory.retrieve({'mean': torch.ones(3),
                             'log_variance': torch.zeros(3)})
        distant = {'mean': torch.full((2,), 1e10),
                   'log_variance': torch.zeros(2), 'sample_count': 1}
        retrieved = memory.retrieve(distant, top_k=1)
        self.assertTrue(torch.isfinite(torch.tensor(retrieved[0]['weight'])))
        self.assertGreaterEqual(retrieved[0]['weight'], 0)
        self.assertAlmostEqual(sum(entry['weight'] for entry in retrieved), 1.0,
                               places=6)
        self.assertAlmostEqual(memory.retention_weight(retrieved), memory.min_similarity)
        with self.assertRaises(ValueError):
            memory.retrieve(distant, top_k=0)

    def test_distribution_memory_keeps_capacity_and_never_stores_raw_samples(self):
        memory = DistributionMemory(max_entries=2, merge_similarity=1.0)
        for task, offset in enumerate((0.0, 4.0, 8.0, 12.0), start=1):
            memory.add(task, memory.summarize(torch.tensor(
                [[offset, 1.0], [offset + 0.1, 1.2]])))
            self.assertLessEqual(len(memory), 2)
        self.assertTrue(all(not any(key in entry for key in ('x', 'y', 'inputs', 'targets'))
                            for entry in memory.entries))

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
        self.assertEqual([len(memory) for memory in trainer.distribution_memories], [1, 1])
        self.assertEqual([memory.entries[0]['occurrences']
                          for memory in trainer.distribution_memories], [2, 2])
        self.assertIsNotNone(clients[0].previous_teacher)
        reference_before = [
            parameter.detach().clone()
            for parameter in trainer.reference_encoder.parameters()]
        trainer.fit([client_batches])
        self.assertEqual([memory.entries[0]['occurrences']
                          for memory in trainer.distribution_memories], [3, 3])
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

    def test_reference_encoder_uses_synchronized_global_state_with_missing_clients(self):
        torch.manual_seed(19)
        clients = []
        for _ in range(4):
            model = RepresentedLinear()
            clients.append(ContinualClient(
                model, torch.optim.SGD(model.parameters(), lr=0.1),
                torch.device('cpu'), replay_enabled=False))
        inactive_state = [
            {name: value.detach().clone() for name, value in client.model.state_dict().items()}
            for client in (clients[0], clients[2])]
        data_two = torch.arange(1, 3, dtype=torch.float32).reshape(2, 1, 1)
        data_four = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1)
        trainer = ContinualFedMixer(
            clients, lambda model, x: model(x), method='proposed',
            generate_public_anchors=False)
        trainer.fit([[None, [(data_two, torch.zeros_like(data_two))], None,
                      [(data_four, torch.zeros_like(data_four))]]])
        for old_state, inactive_client in zip(inactive_state, (clients[0], clients[2])):
            for name, value in inactive_client.model.state_dict().items():
                self.assertTrue(torch.equal(old_state[name], value))
        expected = {}
        state_two = clients[1].model.state_dict()
        state_four = clients[3].model.state_dict()
        for name in state_two:
            expected[name] = (state_two[name].float() * 2 +
                              state_four[name].float() * 4) / 6
            self.assertTrue(torch.allclose(
                trainer.reference_encoder.state_dict()[name].float(), expected[name]))
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in trainer.reference_encoder.parameters()))
        representation_one = trainer._encode(trainer.reference_encoder, data_four)
        representation_two = trainer._encode(trainer.reference_encoder, data_four)
        self.assertEqual(representation_one.shape, representation_two.shape)
        self.assertTrue(torch.equal(representation_one, representation_two))

    def test_post_mixing_representation_retains_time_and_hidden_features(self):
        config = type('Config', (), dict(
            task_name='long_term_forecast', seq_len=8, label_len=2, pred_len=2,
            down_sampling_window=2, down_sampling_layers=1,
            channel_independence=1, e_layers=1, moving_avg=3, enc_in=1,
            c_out=1, use_future_temporal_feature=0, d_model=8, embed='timeF',
            freq='h', dropout=0.0, use_norm=1, decomp_method='moving_avg',
            d_ff=16, top_k=3, factor=1, activation='gelu',
            down_sampling_method='avg'))()
        model = Model(config).eval()
        inputs = torch.randn(3, 8, 1)
        with torch.no_grad():
            prediction, representation = model.forward_with_representation(inputs)
        self.assertEqual(prediction.shape, (3, 2, 1))
        self.assertEqual(representation.shape[0], 3)
        self.assertGreater(representation.shape[1], 2)
        self.assertTrue(torch.isfinite(representation).all())

        with torch.no_grad():
            _, batch_one = model.forward_with_representation(inputs[:1])
            _, reversed_representation = model.forward_with_representation(
                inputs[:1].flip(dims=(1,)))
        self.assertEqual(batch_one.shape[0], 1)
        self.assertGreater(batch_one.shape[1], 8)
        self.assertFalse(torch.allclose(batch_one, reversed_representation))
        with self.assertRaisesRegex(ValueError, 'seq_len'):
            model.forward_with_representation(torch.randn(1, 7, 1))

    def test_channel_independent_features_return_one_representation_per_window(self):
        config = type('Config', (), dict(
            task_name='long_term_forecast', seq_len=8, label_len=2, pred_len=2,
            down_sampling_window=2, down_sampling_layers=1,
            channel_independence=1, e_layers=1, moving_avg=3, enc_in=3,
            c_out=3, use_future_temporal_feature=0, d_model=8, embed='timeF',
            freq='h', dropout=0.0, use_norm=1, decomp_method='moving_avg',
            d_ff=16, top_k=3, factor=1, activation='gelu',
            down_sampling_method='avg'))()
        model = Model(config).eval()
        with torch.no_grad():
            prediction, representation = model.forward_with_representation(
                torch.randn(1, 8, 3))
        self.assertEqual(prediction.shape, (1, 2, 3))
        self.assertEqual(representation.shape[0], 1)
        self.assertTrue(torch.isfinite(representation).all())

    def test_similarity_weighted_topk_retention_and_eviction(self):
        memory = DistributionMemory(max_entries=2, min_similarity=0.1)
        for task, shift in enumerate((0.0, 0.5, 2.0), start=1):
            summary = memory.summarize(torch.tensor([[shift, 0.0],
                                                     [shift + 0.1, 0.2]]))
            memory.add(task, summary)
        self.assertEqual(len(memory), 2)
        self.assertNotIn(2, [entry['task_id'] for entry in memory.entries])
        query = memory.entries[0]
        retrieved = memory.retrieve(query, top_k=2)
        weighted = sum(item['similarity'] * item['weight'] for item in retrieved)
        expected = 0.1 + 0.9 * weighted
        self.assertAlmostEqual(memory.retention_weight(retrieved), expected, places=6)
        self.assertAlmostEqual(sum(item['weight'] for item in retrieved), 1.0, places=6)

    def test_round_drift_aggregates_batches_and_separates_pathways(self):
        detector = MultiscaleSpectralTrendDetector(
            seasonal_threshold=0.05, trend_threshold=2.1,
            scales=1, minimum_length=4)
        stable_trend_a = torch.linspace(0, 1, 32).view(1, 32, 1)
        detector.update(stable_trend_a)
        season = torch.sin(torch.arange(32) * 2 * torch.pi / 8).view(1, 32, 1)
        seasonal, trend = detector.signatures(season)
        flags = detector.update_signatures(seasonal, trend)[:2]
        self.assertTrue(flags[0])
        self.assertFalse(flags[1])

        detector = MultiscaleSpectralTrendDetector(
            seasonal_threshold=2.1, trend_threshold=0.05,
            scales=1, minimum_length=4)
        detector.update(stable_trend_a)
        new_trend = -stable_trend_a
        seasonal, trend = detector.signatures(new_trend)
        flags = detector.update_signatures(seasonal, trend)[:2]
        self.assertFalse(flags[0])
        self.assertTrue(flags[1])

    def test_round_drift_uses_sample_weighted_online_aggregation(self):
        model = RepresentedLinear()
        client = ContinualClient(
            model, torch.optim.SGD(model.parameters(), lr=0.0),
            torch.device('cpu'), drift_threshold=0.1, trend_threshold=0.1)
        stable = torch.sin(torch.arange(32) * 2 * torch.pi / 8).view(1, 32, 1)
        outlier = torch.sin(torch.arange(32) * 2 * torch.pi / 3).view(1, 32, 1)
        client.drift_detector.update(stable)
        stable_batch = stable.repeat(8, 1, 1)
        stable_signature = client.drift_detector.signatures(stable_batch)
        outlier_signature = client.drift_detector.signatures(outlier)
        client.begin_round()
        client.accumulate_drift(stable_batch)
        client.accumulate_drift(outlier)
        flags = client.finalize_round_drift()
        expected_season = (stable_signature[0] * 8 + outlier_signature[0]) / 9
        expected_trend = (stable_signature[1] * 8 + outlier_signature[1]) / 9
        self.assertTrue(torch.allclose(client.last_signatures[0], expected_season))
        self.assertTrue(torch.allclose(client.last_signatures[1], expected_trend))
        self.assertEqual(flags, (False, False))
        self.assertEqual(client.persistent_drift_bytes,
                         client.last_signatures[0].untyped_storage().nbytes() +
                         client.last_signatures[1].untyped_storage().nbytes() +
                         client.drift_detector.previous_seasonal.untyped_storage().nbytes() +
                         client.drift_detector.previous_trend.untyped_storage().nbytes())

    def test_drift_detector_stable_abrupt_gradual_and_recurring_regimes(self):
        t = torch.linspace(0, 1, 32).view(1, 32, 1)
        detector = MultiscaleSpectralTrendDetector(
            seasonal_threshold=0.15, trend_threshold=0.1,
            scales=1, minimum_length=4)
        self.assertEqual(detector.update(t)[:2], (False, False))
        self.assertEqual(detector.update(t.clone())[:2], (False, False))
        abrupt = -t
        self.assertEqual(detector.update(abrupt)[:2], (False, True))
        self.assertEqual(detector.update(t)[:2], (False, True))
        self.assertEqual(detector.update(t)[:2], (False, False))

        gradual = MultiscaleSpectralTrendDetector(
            seasonal_threshold=0.15, trend_threshold=0.1,
            scales=1, minimum_length=4)
        gradual.update(t)
        detections = []
        for step in range(1, 11):
            detections.append(gradual.update(t + step * 0.025)[:2])
        self.assertTrue(any(flags[1] for flags in detections))

    def test_reclustering_only_when_pathway_drift_or_initialization(self):
        clients = []
        for _ in range(2):
            model = RepresentedLinear()
            clients.append(ContinualClient(
                model, torch.optim.SGD(model.parameters(), lr=0.01),
                torch.device('cpu'), replay_enabled=False))
        inputs = torch.ones(4, 1, 1)
        trainer = ContinualFedMixer(
            clients, lambda model, x: model(x), method='proposed',
            generate_public_anchors=False)
        first = trainer.fit([[[ (inputs, inputs)], [(inputs, inputs)]]])[0]
        second = trainer.fit([[[ (inputs, inputs)], [(inputs, inputs)]]])[0]
        self.assertEqual(first['reclustered'], {'seasonal': True, 'trend': True})
        self.assertEqual(second['reclustered'], {'seasonal': False, 'trend': False})

    def test_reclustering_is_pathway_specific(self):
        ramp = torch.linspace(0, 1, 32).view(1, 32, 1).repeat(4, 1, 1)
        seasonal = torch.sin(torch.arange(32) * 2 * torch.pi / 8)
        seasonal = seasonal.view(1, 32, 1).repeat(4, 1, 1)
        reverse = -ramp
        targets = torch.zeros(4, 32, 1)

        def make_trainer(seasonal_threshold, trend_threshold):
            model = RepresentedLinear()
            client = ContinualClient(
                model, torch.optim.SGD(model.parameters(), lr=0.0),
                torch.device('cpu'), replay_enabled=False,
                drift_threshold=seasonal_threshold,
                trend_threshold=trend_threshold)
            return ContinualFedMixer(
                [client], lambda current, x: current(x), method='proposed',
                generate_public_anchors=False)

        seasonal_trainer = make_trainer(0.05, 2.1)
        seasonal_trainer.fit([[[ (ramp, targets)]]])
        seasonal_result = seasonal_trainer.fit([[[ (seasonal, targets)]]])[0]
        self.assertEqual(seasonal_result['reclustered'],
                         {'seasonal': True, 'trend': False})

        trend_trainer = make_trainer(2.1, 0.05)
        trend_trainer.fit([[[ (ramp, targets)]]])
        trend_result = trend_trainer.fit([[[ (reverse, targets)]]])[0]
        self.assertEqual(trend_result['reclustered'],
                         {'seasonal': False, 'trend': True})

    def test_loss_switches_independently_disable_terms(self):
        inputs = torch.ones(3, 1, 1)
        targets = torch.zeros_like(inputs)
        for enabled in ('history', 'representation', 'mutual'):
            model = RepresentedLinear()
            client = ContinualClient(
                model, torch.optim.SGD(model.parameters(), lr=0.0),
                torch.device('cpu'), replay_enabled=False)
            client.set_previous_teacher()
            with torch.no_grad():
                for parameter in client.model.parameters():
                    parameter.add_(1.0)
            args = {'history_enabled': True, 'representation_enabled': True,
                    'mutual_enabled': True}
            args[enabled + '_enabled'] = False
            result = client.train_batch_replay_free(
                inputs, targets, lambda m, x: m(x), retention_weight=1.0,
                distillation_inputs=inputs,
                distillation_targets=torch.zeros_like(inputs), **args)
            key = {'history': 'history', 'representation': 'representation',
                   'mutual': 'mutual'}[enabled]
            self.assertEqual(result[key], 0.0)
            self.assertGreater(result['total'], result['forecast'])

    def test_checkpoint_save_restore_and_schema_validation(self):
        clients = []
        for _ in range(2):
            model = RepresentedLinear()
            clients.append(ContinualClient(
                model, torch.optim.SGD(model.parameters(), lr=0.01),
                torch.device('cpu'), replay_enabled=False))
        inputs = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1)
        trainer = ContinualFedMixer(
            clients, lambda model, x: model(x), method='proposed',
            generate_public_anchors=False)
        trainer.fit([[[ (inputs, inputs)], [(inputs, inputs)]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trainer.pt'
            torch.save(trainer.state_dict(), path)
            state = torch.load(path, map_location='cpu', weights_only=False)
            restored_clients = []
            for _ in range(2):
                model = RepresentedLinear()
                restored_clients.append(ContinualClient(
                    model, torch.optim.SGD(model.parameters(), lr=0.01),
                    torch.device('cpu'), replay_enabled=False))
            restored = ContinualFedMixer(
                restored_clients, lambda model, x: model(x), method='proposed',
                generate_public_anchors=False)
            restored.load_state_dict(state)
            self.assertEqual([len(m) for m in restored.distribution_memories],
                             [len(m) for m in trainer.distribution_memories])
            for left, right in zip(trainer.reference_encoder.parameters(),
                                   restored.reference_encoder.parameters()):
                self.assertTrue(torch.equal(left, right))
            for key in ('schema_version', 'method', 'distribution_memories'):
                state.pop(key, None)
            with self.assertRaises(ValueError):
                restored.load_state_dict({'schema_version': 99, 'method': 'proposed'})

    def test_teacher_and_student_use_identical_distillation_input(self):
        model = RecordingLinear()
        client = ContinualClient(
            model, torch.optim.SGD(model.parameters(), lr=0.0),
            torch.device('cpu'), replay_enabled=False)
        client.set_previous_teacher()
        teacher_before = {name: value.detach().clone()
                          for name, value in client.previous_teacher.state_dict().items()}
        inputs = torch.tensor([[[2.0]], [[3.0]]])
        client.train_batch_replay_free(
            inputs, torch.zeros_like(inputs), lambda current, x: current(x),
            retention_weight=1.0)
        student_input = client.model.seen[-1]
        teacher_input = client.previous_teacher.seen[-1]
        self.assertTrue(torch.equal(student_input, teacher_input))
        self.assertTrue(torch.equal(student_input, inputs))
        for name, value in client.previous_teacher.state_dict().items():
            self.assertTrue(torch.equal(value, teacher_before[name]))
            self.assertFalse(value.requires_grad)

    def test_teacher_is_frozen_during_training_and_replaced_after_round(self):
        torch.manual_seed(33)
        clients = []
        for _ in range(2):
            model = RepresentedLinear()
            clients.append(ContinualClient(
                model, torch.optim.SGD(model.parameters(), lr=0.2),
                torch.device('cpu'), replay_enabled=False))
        inputs = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1)
        targets = torch.zeros_like(inputs)
        trainer = ContinualFedMixer(
            clients, lambda current, x: current(x), method='proposed',
            generate_public_anchors=False)
        trainer.fit([[[(inputs, targets)], [(inputs, targets)]]])
        previous = [{name: value.detach().clone()
                     for name, value in client.previous_teacher.state_dict().items()}
                    for client in clients]
        trainer.fit([[[(inputs * 2, targets)], [(inputs * 2, targets)]]])
        for client, old_teacher in zip(clients, previous):
            for name, value in client.previous_teacher.state_dict().items():
                self.assertTrue(torch.equal(value, client.model.state_dict()[name]))
                self.assertFalse(torch.equal(old_teacher[name], value))
                self.assertFalse(value.requires_grad)

    def test_teacher_models_are_omitted_when_teacher_losses_are_disabled(self):
        clients = []
        for _ in range(2):
            model = RepresentedLinear()
            clients.append(ContinualClient(
                model, torch.optim.SGD(model.parameters(), lr=0.01),
                torch.device('cpu'), replay_enabled=False))
        inputs = torch.ones(4, 1, 1)
        trainer = ContinualFedMixer(
            clients, lambda current, x: current(x), method='proposed',
            generate_public_anchors=False, history_enabled=False,
            representation_enabled=False)
        result = trainer.fit([[[(inputs, inputs)], [(inputs, inputs)]]])[0]
        self.assertTrue(all(client.previous_teacher is None for client in clients))
        self.assertEqual(result['memory_bytes']['teacher_bytes'], 0)

    def test_proposed_memory_accounting_includes_reference_and_teachers(self):
        clients = []
        for _ in range(2):
            model = RepresentedLinear()
            clients.append(ContinualClient(
                model, torch.optim.SGD(model.parameters(), lr=0.01),
                torch.device('cpu'), replay_enabled=False))
        inputs = torch.ones(4, 1, 1)
        trainer = ContinualFedMixer(
            clients, lambda model, x: model(x), method='proposed',
            generate_public_anchors=False)
        result = trainer.fit([[[ (inputs, inputs)], [(inputs, inputs)]]])[0]
        self.assertGreater(result['memory_bytes']['reference_encoder_bytes'], 0)
        self.assertGreater(result['memory_bytes']['teacher_bytes'], 0)
        self.assertEqual(result['memory_bytes']['algorithmic_memory_bytes'],
                         result['memory_bytes']['distribution_bytes'] +
                         result['memory_bytes']['reference_encoder_bytes'] +
                         result['memory_bytes']['teacher_bytes'] +
                         result['memory_bytes']['gradient_projection_bytes'] +
                         result['memory_bytes']['drift_state_bytes'])
        self.assertEqual(result['memory_bytes']['total_persistent_bytes'],
                         result['memory_bytes']['algorithmic_memory_bytes'] +
                         result['memory_bytes']['client_model_bytes'] +
                         result['memory_bytes']['optimizer_state_bytes'])


if __name__ == '__main__':
    unittest.main()
