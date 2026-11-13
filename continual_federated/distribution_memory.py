"""Client-local representation-distribution memory for replay-free learning."""

from copy import deepcopy

import torch


class DistributionMemory:
    """Bounded memory of diagonal Gaussian representation summaries.

    Entries contain statistics and metadata only. In particular, input and
    target windows are never retained by this class.
    """

    schema_version = 1

    def __init__(self, max_entries=8, temperature=1.0, variance_weight=0.5,
                 min_similarity=0.1, epsilon=1e-6):
        if max_entries < 1:
            raise ValueError('max_entries must be positive')
        if temperature <= 0 or variance_weight < 0:
            raise ValueError('temperature must be positive and variance_weight non-negative')
        if not 0 <= min_similarity <= 1 or epsilon <= 0:
            raise ValueError('min_similarity must be in [0, 1] and epsilon positive')
        self.max_entries = max_entries
        self.temperature = float(temperature)
        self.variance_weight = float(variance_weight)
        self.min_similarity = float(min_similarity)
        self.epsilon = float(epsilon)
        self.entries = []

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def summarize(representations, metadata=None, epsilon=1e-6):
        values = representations.detach().float().reshape(representations.shape[0], -1)
        if values.shape[0] == 0:
            raise ValueError('representations must contain at least one sample')
        mean = values.mean(dim=0).cpu()
        variance = values.var(dim=0, unbiased=False).clamp_min(epsilon).cpu()
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise ValueError('representations must be finite')
        result = {
            'mean': mean,
            'log_variance': variance.log(),
            'sample_count': int(values.shape[0]),
        }
        if metadata:
            result.update(deepcopy(metadata))
        return result

    def add(self, task_id, summary, temporal_signature=None, **metadata):
        mean = summary['mean'].detach().float().cpu().clone()
        log_variance = summary['log_variance'].detach().float().cpu().clone()
        if mean.ndim != 1 or log_variance.shape != mean.shape:
            raise ValueError('summary mean and log_variance must be matching vectors')
        if not torch.isfinite(mean).all() or not torch.isfinite(log_variance).all():
            raise ValueError('summary statistics must be finite')
        entry = {
            'task_id': task_id,
            'mean': mean,
            'log_variance': log_variance,
            'sample_count': int(summary.get('sample_count', 0)),
            'temporal_signature': None if temporal_signature is None else
                torch.as_tensor(temporal_signature).detach().float().cpu().clone(),
            'schema_version': self.schema_version,
        }
        entry.update(deepcopy(metadata))
        self.entries.append(entry)
        self._evict_if_needed()

    def similarity(self, query_summary, stored_summary):
        mean_distance = (query_summary['mean'].float() -
                         stored_summary['mean'].float()).pow(2).mean()
        variance_distance = (query_summary['log_variance'].float() -
                             stored_summary['log_variance'].float()).pow(2).mean()
        distance = mean_distance + self.variance_weight * variance_distance
        similarity = torch.exp(-distance / self.temperature)
        return float(similarity.clamp(0, 1).item())

    def retrieve(self, query_summary, top_k=None):
        if not self.entries:
            return []
        count = min(top_k or len(self.entries), len(self.entries))
        ranked = sorted(
            ((self.similarity(query_summary, entry), entry)
             for entry in self.entries),
            key=lambda item: item[0], reverse=True)
        selected = ranked[:count]
        scores = torch.tensor([score for score, _ in selected], dtype=torch.float32)
        weights = scores / scores.sum().clamp_min(self.epsilon)
        return [
            {'entry': entry, 'similarity': float(score), 'weight': float(weight)}
            for (score, entry), weight in zip(selected, weights)
        ]

    def retention_weight(self, retrieved):
        if not retrieved:
            return self.min_similarity
        return self.min_similarity + (1 - self.min_similarity) * max(
            item['similarity'] for item in retrieved)

    @property
    def byte_count(self):
        total = 0
        for entry in self.entries:
            for value in entry.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total

    def _evict_if_needed(self):
        while len(self.entries) > self.max_entries:
            if len(self.entries) <= 1:
                del self.entries[0]
                continue
            scores = torch.zeros(len(self.entries))
            for index, entry in enumerate(self.entries):
                similarities = [
                    self.similarity(entry, other)
                    for other_index, other in enumerate(self.entries)
                    if index != other_index
                ]
                scores[index] = max(similarities)
            del self.entries[int(scores.argmax().item())]

    def state_dict(self):
        return {
            'schema_version': self.schema_version,
            'max_entries': self.max_entries,
            'temperature': self.temperature,
            'variance_weight': self.variance_weight,
            'min_similarity': self.min_similarity,
            'epsilon': self.epsilon,
            'entries': deepcopy(self.entries),
        }

    def load_state_dict(self, state):
        if state.get('schema_version') != self.schema_version:
            raise ValueError('unsupported distribution memory schema version')
        self.max_entries = int(state['max_entries'])
        self.temperature = float(state['temperature'])
        self.variance_weight = float(state['variance_weight'])
        self.min_similarity = float(state['min_similarity'])
        self.epsilon = float(state['epsilon'])
        self.entries = deepcopy(state['entries'])
        self._evict_if_needed()
