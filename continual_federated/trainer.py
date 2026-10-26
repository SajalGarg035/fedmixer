"""Temporal-round orchestration for continual federated forecasting."""

import torch

from .server import FedMixerServer


class ContinualFedMixer:
    """Run local replay, clustered averaging, and mutual distillation per round.

    ``round_batches`` is a list of rounds. Each round contains one list of
    ``(inputs, targets)`` batches per client. The forecasting model and its
    unchanged forward path are supplied by the caller.
    """

    def __init__(self, clients, forward_fn, public_inputs=None, server=None):
        if not clients:
            raise ValueError('clients must not be empty')
        self.clients = clients
        self.forward_fn = forward_fn
        self.public_inputs = public_inputs
        self.server = server or FedMixerServer()

    def fit(self, round_batches, continual=True, start_round=1):
        history = []
        for round_index, client_batches in enumerate(round_batches, start=start_round):
            if len(client_batches) != len(self.clients):
                raise ValueError('each round must contain one entry per client')

            if self.public_inputs is None:
                first_batch = next(
                    (batch for client_round in client_batches for batch in client_round),
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

            losses = []
            sample_counts = []
            drifted = []
            seasonal_signatures = []
            trend_signatures = []
            seasonal_drift = []
            trend_drift = []
            for client_index, (client, batches) in enumerate(zip(self.clients, client_batches)):
                client.begin_round()
                client_loss = []
                sample_count = 0
                client_drift = False
                for inputs, targets in batches:
                    if continual:
                        client_drift = client.observe(inputs) or client_drift
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
                sample_counts.append(sample_count)
                drifted.append(client_drift)

            states = [{name: value.detach().clone()
                       for name, value in client.model.state_dict().items()}
                      for client in self.clients]
            seasonal_recluster = round_index == 1 or (
                continual and any(client.last_seasonal_drift for client in self.clients))
            trend_recluster = round_index == 1 or (
                continual and any(client.last_trend_drift for client in self.clients))
            self.server.recluster_pathways(
                seasonal_signatures, trend_signatures,
                seasonal=seasonal_recluster, trend=trend_recluster)
            cluster_states = self.server.aggregate_decoupled(
                states, seasonal_signatures, trend_signatures, sample_counts,
                force_recluster=False)
            for client_index, state in enumerate(cluster_states):
                self.clients[client_index].model.load_state_dict(state)

            history.append({
                'round': round_index,
                'clusters': [list(indices) for indices in self.server.clusters],
                'seasonal_clusters': [list(indices) for indices in self.server.seasonal_clusters],
                'trend_clusters': [list(indices) for indices in self.server.trend_clusters],
                'losses': losses,
                'drifted': drifted,
                'seasonal_drift': seasonal_drift,
                'trend_drift': trend_drift,
                'memory': [len(client.memory) for client in self.clients],
            })
        return history

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
                representative_indices = list(dict.fromkeys((
                    seasonal_lookup[client_index], trend_lookup[client_index])))
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