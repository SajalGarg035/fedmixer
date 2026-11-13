"""Temporal-round orchestration for continual federated forecasting."""

import torch
from copy import deepcopy

from .distribution_memory import DistributionMemory
from .server import FedMixerServer


class ContinualFedMixer:
    """Run legacy replay or replay-free distribution training per round.

    ``round_batches`` is a list of rounds. Each round contains one list of
    ``(inputs, targets)`` batches per client. The forecasting model and its
    unchanged forward path are supplied by the caller.
    """

    def __init__(self, clients, forward_fn, public_inputs=None, server=None,
                 method='legacy', distribution_memory_kwargs=None,
                 history_weight=0.3, representation_weight=0.1,
                 generate_public_anchors=True):
        if not clients:
            raise ValueError('clients must not be empty')
        self.clients = clients
        self.forward_fn = forward_fn
        self.public_inputs = public_inputs
        self.generate_public_anchors = generate_public_anchors
        self.server = server or FedMixerServer()
        if method not in ('legacy', 'proposed'):
            raise ValueError("method must be 'legacy' or 'proposed'")
        self.method = method
        self.history_weight = history_weight
        self.representation_weight = representation_weight
        self.reference_encoder = None
        memory_kwargs = dict(distribution_memory_kwargs or {})
        self.distribution_top_k = int(memory_kwargs.pop('top_k', 3))
        self.distribution_memories = (
            [DistributionMemory(**memory_kwargs) for _ in clients]
            if method == 'proposed' else [])
        if method == 'proposed':
            for client, memory in zip(self.clients, self.distribution_memories):
                client.memory = None
                client.replay_enabled = False
                client.distribution_memory = memory

    def fit(self, round_batches, continual=True, start_round=1):
        history = []
        for round_index, client_batches in enumerate(round_batches, start=start_round):
            if len(client_batches) != len(self.clients):
                raise ValueError('each round must contain one entry per client')
            active_indices = [
                index for index, batches in enumerate(client_batches)
                if batches is not None and len(batches) > 0]
            if not active_indices:
                raise ValueError('round {} has no active clients'.format(round_index))

            if self.public_inputs is None and self.generate_public_anchors:
                first_batch = next(
                    (batch for index in active_indices
                     for batch in client_batches[index]),
                    None)
                if first_batch is not None:
                    self.public_inputs = self.server.generate_synthetic_fourier_probes(
                        first_batch[0].shape[1], first_batch[0].shape[2],
                        batch_size=min(8, first_batch[0].shape[0]),
                        device=first_batch[0].device)
            if self.public_inputs is not None and self.server.clusters is not None:
                public_targets = self._distillation_targets()
            else:
                public_targets = [None] * len(self.clients)

            public_targets = public_targets if self.public_inputs is not None else (
                [None] * len(self.clients))
            losses = []
            loss_components = []
            sample_counts = []
            drifted = []
            seasonal_signatures = []
            trend_signatures = []
            seasonal_drift = []
            trend_drift = []
            for client_index in active_indices:
                client = self.clients[client_index]
                batches = client_batches[client_index]
                client.begin_round()
                client_loss = []
                component_totals = {
                    'forecast': 0.0, 'history': 0.0, 'representation': 0.0,
                    'mutual': 0.0, 'total': 0.0}
                sample_count = 0
                client_drift = False
                retention_weight = self._retention_weight(client_index, batches)
                for inputs, targets in batches:
                    if continual:
                        client_drift = client.observe(inputs) or client_drift
                    if self.method == 'proposed':
                        components = client.train_batch_replay_free(
                            inputs, targets, self.forward_fn,
                            retention_weight=retention_weight,
                            history_weight=self.history_weight,
                            representation_weight=self.representation_weight,
                            distillation_inputs=self.public_inputs,
                            distillation_targets=public_targets[client_index])
                        client_loss.append(components['total'])
                        for name in component_totals:
                            component_totals[name] += components[name]
                    else:
                        client_loss.append(client.train_batch(
                            inputs, targets, self.forward_fn,
                            self.public_inputs, public_targets[client_index],
                            use_replay=continual))
                    sample_count += len(inputs)
                seasonal_signature, trend_signature = client.last_signatures
                seasonal_signatures.append(
                    seasonal_signature if seasonal_signature is not None else torch.zeros(1))
                trend_signatures.append(
                    trend_signature if trend_signature is not None else torch.zeros(1))
                seasonal_drift.append(client.last_drift[0])
                trend_drift.append(client.last_drift[1])
                losses.append(sum(client_loss) / max(1, len(client_loss)))
                if self.method == 'proposed':
                    divisor = max(1, len(batches))
                    loss_components.append({
                        name: value / divisor for name, value in component_totals.items()})
                sample_counts.append(sample_count)
                drifted.append(client_drift)

            states = [{name: value.detach().clone()
                       for name, value in self.clients[index].model.state_dict().items()}
                      for index in active_indices]
            seasonal_recluster = True
            trend_recluster = True
            self.server.recluster_pathways(
                seasonal_signatures, trend_signatures,
                seasonal=seasonal_recluster, trend=trend_recluster)
            cluster_states = self.server.aggregate_decoupled(
                states, seasonal_signatures, trend_signatures, sample_counts,
                force_recluster=False)
            for local_index, state in enumerate(cluster_states):
                self.clients[active_indices[local_index]].model.load_state_dict(state)

            seasonal_clusters = [
                [active_indices[index] for index in indices]
                for indices in self.server.seasonal_clusters]
            trend_clusters = [
                [active_indices[index] for index in indices]
                for indices in self.server.trend_clusters]
            self.server.seasonal_clusters = seasonal_clusters
            self.server.trend_clusters = trend_clusters
            self.server.clusters = seasonal_clusters

            if self.method == 'proposed':
                self._record_distribution_summaries(
                    client_batches, seasonal_signatures, round_index,
                    active_indices)
                for client_index in active_indices:
                    self.clients[client_index].set_previous_teacher()

            history.append({
                'round': round_index,
                'clusters': [list(indices) for indices in self.server.clusters],
                'seasonal_clusters': [list(indices) for indices in self.server.seasonal_clusters],
                'trend_clusters': [list(indices) for indices in self.server.trend_clusters],
                'losses': losses,
                'drifted': drifted,
                'seasonal_drift': seasonal_drift,
                'trend_drift': trend_drift,
                'memory': [
                    len(client.memory) if client.memory is not None else 0
                    for client in self.clients],
                'distribution_entries': [
                    len(memory) for memory in self.distribution_memories],
                'distribution_bytes': [
                    memory.byte_count for memory in self.distribution_memories],
                'loss_components': loss_components,
            })
        return history

    def _encode(self, model, inputs):
        if hasattr(model, 'encode'):
            return model.encode(inputs)
        return self.forward_fn(model, inputs).reshape(inputs.shape[0], -1)

    def _current_summary(self, batches):
        representations = []
        with torch.no_grad():
            model = self.reference_encoder
            for inputs, _ in batches:
                device = next(model.parameters()).device
                representations.append(self._encode(model, inputs.to(device)).cpu())
        if not representations:
            return None
        return DistributionMemory.summarize(torch.cat(representations, dim=0))

    def _retention_weight(self, client_index, batches):
        if self.method != 'proposed' or self.reference_encoder is None:
            return 0.0 if self.method == 'proposed' else 1.0
        summary = self._current_summary(batches)
        retrieved = self.distribution_memories[client_index].retrieve(
            summary, top_k=self.distribution_top_k)
        return self.distribution_memories[client_index].retention_weight(retrieved)

    def _record_distribution_summaries(self, client_batches, signatures, round_index,
                                       active_indices):
        if self.reference_encoder is None:
            self.reference_encoder = deepcopy(self.clients[0].model)
            self.reference_encoder.eval()
            for parameter in self.reference_encoder.parameters():
                parameter.requires_grad_(False)
        for local_index, client_index in enumerate(active_indices):
            batches = client_batches[client_index]
            summary = self._current_summary(batches)
            if summary is None:
                continue
            signature = signatures[local_index]
            self.distribution_memories[client_index].add(
                task_id=round_index, summary=summary,
                temporal_signature=signature, round=round_index)

    def state_dict(self):
        return {
            'method': self.method,
            'reference_encoder': None if self.reference_encoder is None else
                self.reference_encoder.state_dict(),
            'distribution_memories': [
                memory.state_dict() for memory in self.distribution_memories],
        }

    def load_state_dict(self, state):
        if state.get('method') != self.method:
            raise ValueError('trainer checkpoint method does not match')
        reference_state = state.get('reference_encoder')
        if reference_state is not None:
            if self.reference_encoder is None:
                self.reference_encoder = deepcopy(self.clients[0].model)
            self.reference_encoder.load_state_dict(reference_state)
            self.reference_encoder.eval()
            for parameter in self.reference_encoder.parameters():
                parameter.requires_grad_(False)
        for memory, memory_state in zip(
                self.distribution_memories, state.get('distribution_memories', [])):
            memory.load_state_dict(memory_state)

    def _distillation_targets(self):
        seasonal_lookup = {
            client: indices[0] for indices in self.server.seasonal_clusters
            for client in indices}
        trend_lookup = {
            client: indices[0] for indices in self.server.trend_clusters
            for client in indices}
        with torch.no_grad():
            targets = []
            for client_index in range(len(self.clients)):
                representative_indices = list(dict.fromkeys(
                    index for index in (
                        seasonal_lookup.get(client_index, client_index),
                        trend_lookup.get(client_index, client_index))
                    if index is not None))
                predictions = []
                scale_predictions = []
                for representative_index in representative_indices:
                    model = self.clients[representative_index].model
                    device = next(model.parameters()).device
                    inputs = self.public_inputs.to(device)
                    if hasattr(model, 'forecast_multiscale'):
                        output = model.forecast_multiscale(inputs)
                        predictions.append(output[0])
                        scale_predictions.append(torch.stack(output[1], dim=1))
                    else:
                        predictions.append(self.forward_fn(model, inputs))
                prediction = torch.stack(predictions).mean(dim=0).detach()
                if scale_predictions:
                    scales = torch.stack(scale_predictions).mean(dim=0).detach()
                    targets.append((prediction, scales))
                else:
                    targets.append(prediction)
        return targets