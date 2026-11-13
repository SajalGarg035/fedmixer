"""Client-local representation-distribution memory for replay-free learning."""

from copy import deepcopy
import math

import torch


class DistributionMemory:
    """Bounded memory of diagonal Gaussian representation summaries.

    Entries contain statistics and metadata only. In particular, input and
    target windows are never retained by this class.
    """

    schema_version = 2

    def __init__(self, max_entries=8, temperature=1.0, variance_weight=0.5,
                 min_similarity=0.1, epsilon=1e-6,
                 merge_similarity=0.995):
        if int(max_entries) < 1:
            raise ValueError('max_entries must be positive')
        if (not math.isfinite(float(temperature)) or temperature <= 0 or
                not math.isfinite(float(variance_weight)) or variance_weight < 0):
            raise ValueError('temperature must be positive and variance_weight non-negative')
        if (not 0 <= min_similarity <= 1 or not math.isfinite(float(epsilon)) or
                epsilon <= 0):
            raise ValueError('min_similarity must be in [0, 1] and epsilon positive')
        if not math.isfinite(float(merge_similarity)) or not 0 <= merge_similarity <= 1:
            raise ValueError('merge_similarity must be in [0, 1]')
        self.max_entries = int(max_entries)
        self.temperature = float(temperature)
        self.variance_weight = float(variance_weight)
        self.min_similarity = float(min_similarity)
        self.epsilon = float(epsilon)
        self.merge_similarity = float(merge_similarity)
        self.entries = []

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def summarize(representations, metadata=None, epsilon=1e-6):
        if not torch.is_tensor(representations) or representations.ndim < 2:
            raise ValueError('representations must be a tensor with a batch dimension')
        values = representations.detach().to(torch.float64).reshape(
            representations.shape[0], -1)
        if values.shape[0] == 0:
            raise ValueError('representations must contain at least one sample')
        if values.shape[1] == 0 or not torch.isfinite(values).all():
            raise ValueError('representations must be non-empty and finite')
        variance, mean = torch.var_mean(values, dim=0, unbiased=False)
        mean = mean.to(torch.float32).cpu()
        variance = variance.clamp_min(epsilon).to(torch.float32).cpu()
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise ValueError('representations must be finite')
        result = {
            'mean': mean,
            'log_variance': variance.log(),
            'sample_count': int(values.shape[0]),
        }
        if metadata:
            raise ValueError('summary metadata is not supported; pass scalar entry metadata')
        return result

    def add(self, task_id, summary, temporal_signature=None, **metadata):
        if not isinstance(task_id, (str, int, float)) or (
                isinstance(task_id, float) and not math.isfinite(task_id)):
            raise ValueError('task_id must be a finite scalar identifier')
        mean = summary['mean'].detach().float().cpu().clone()
        log_variance = summary['log_variance'].detach().float().cpu().clone()
        if mean.ndim != 1 or log_variance.shape != mean.shape:
            raise ValueError('summary mean and log_variance must be matching vectors')
        if not torch.isfinite(mean).all() or not torch.isfinite(log_variance).all():
            raise ValueError('summary statistics must be finite')
        sample_count = int(summary.get('sample_count', 0))
        if sample_count < 1:
            raise ValueError('summary sample_count must be positive')
        safe_metadata = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(
                    value, (str, int, float, bool, type(None))):
                raise ValueError('distribution metadata must contain scalar values only')
            if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
                raise ValueError('distribution metadata scalars must be finite')
            safe_metadata[key] = value
        if temporal_signature is not None:
            temporal_signature = torch.as_tensor(temporal_signature).detach().float().cpu()
            if temporal_signature.ndim > 1 or not torch.isfinite(temporal_signature).all():
                raise ValueError('temporal_signature must be a finite vector')
        entry = {
            'task_id': task_id,
            'mean': mean,
            'log_variance': log_variance,
            'sample_count': sample_count,
            'occurrences': 1,
            'last_task_id': task_id,
            'temporal_signature': None if temporal_signature is None else
                temporal_signature.clone(),
            'schema_version': self.schema_version,
        }
        entry.update(safe_metadata)
        if not self._merge_recurring_entry(entry):
            self.entries.append(entry)
            self._evict_if_needed()

    def _merge_recurring_entry(self, incoming):
        if not self.entries:
            return False
        similarities = [self.similarity(incoming, entry) for entry in self.entries]
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        if similarities[best_index] < self.merge_similarity:
            return False
        existing = self.entries[best_index]
        first_count = existing['sample_count']
        second_count = incoming['sample_count']
        total = first_count + second_count
        first_mean = existing['mean']
        second_mean = incoming['mean']
        mean = (first_mean * first_count + second_mean * second_count) / total
        first_var = existing['log_variance'].exp()
        second_var = incoming['log_variance'].exp()
        variance = (
            first_count * (first_var + (first_mean - mean).square()) +
            second_count * (second_var + (second_mean - mean).square())) / total
        existing['mean'] = mean
        existing['log_variance'] = variance.clamp_min(self.epsilon).log()
        existing['sample_count'] = total
        existing['occurrences'] = int(existing.get('occurrences', 1)) + 1
        existing['last_task_id'] = incoming['task_id']
        for key, value in incoming.items():
            if key not in ('mean', 'log_variance', 'sample_count', 'occurrences',
                           'last_task_id', 'task_id', 'schema_version'):
                existing[key] = value
        return True

    def similarity(self, query_summary, stored_summary):
        if (query_summary['mean'].shape != stored_summary['mean'].shape or
                query_summary['log_variance'].shape !=
                stored_summary['log_variance'].shape):
            raise ValueError('query and stored representation dimensions differ')
        mean_distance = (query_summary['mean'].float() -
                         stored_summary['mean'].float()).pow(2).mean()
        variance_distance = (query_summary['log_variance'].float() -
                             stored_summary['log_variance'].float()).pow(2).mean()
        distance = mean_distance + self.variance_weight * variance_distance
        if not torch.isfinite(distance):
            raise ValueError('distribution distance must be finite')
        similarity = torch.exp(-distance / self.temperature)
        return float(similarity.clamp(0, 1).item())

    def retrieve(self, query_summary, top_k=None):
        if not self.entries:
            return []
        if top_k is not None and top_k < 1:
            raise ValueError('top_k must be positive')
        count = min(len(self.entries) if top_k is None else top_k,
                    len(self.entries))
        ranked = sorted(
            ((self.similarity(query_summary, entry), entry)
             for entry in self.entries),
            key=lambda item: item[0], reverse=True)
        selected = ranked[:count]
        scores = torch.tensor([score for score, _ in selected], dtype=torch.float32)
        score_sum = scores.sum()
        weights = (scores / score_sum if score_sum > self.epsilon else
                   torch.full_like(scores, 1.0 / len(selected)))
        return [
            {'entry': entry, 'similarity': float(score), 'weight': float(weight)}
            for (score, entry), weight in zip(selected, weights)
        ]

    def retention_weight(self, retrieved):
        if not retrieved:
            return self.min_similarity
        weight_sum = sum(float(item['weight']) for item in retrieved)
        if weight_sum <= self.epsilon:
            return self.min_similarity
        weighted_similarity = sum(
            float(item['weight']) * float(item['similarity'])
            for item in retrieved) / weight_sum
        return self.min_similarity + (1 - self.min_similarity) * weighted_similarity

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
            nearest = []
            for index, entry in enumerate(self.entries):
                similarities = [
                    self.similarity(entry, other)
                    for other_index, other in enumerate(self.entries)
                    if index != other_index]
                nearest.append(max(similarities, default=-1.0))
            # Evict only from the most redundant pair. When their similarity is
            # tied, retain the more frequently observed regime, then the older
            # representative. This keeps recurring and unique regimes useful.
            most_redundant = max(nearest)
            candidates = [index for index, score in enumerate(nearest)
                          if score == most_redundant]
            victim = min(
                candidates,
                key=lambda index: (
                    int(self.entries[index].get('occurrences', 1)),
                    -index))
            del self.entries[victim]

    def state_dict(self):
        return {
            'schema_version': self.schema_version,
            'max_entries': self.max_entries,
            'temperature': self.temperature,
            'variance_weight': self.variance_weight,
            'min_similarity': self.min_similarity,
            'epsilon': self.epsilon,
            'merge_similarity': self.merge_similarity,
            'entries': deepcopy(self.entries),
        }

    def load_state_dict(self, state):
        if state.get('schema_version') not in (1, self.schema_version):
            raise ValueError('unsupported distribution memory schema version')
        loaded = DistributionMemory(
            max_entries=state['max_entries'],
            temperature=state['temperature'],
            variance_weight=state['variance_weight'],
            min_similarity=state['min_similarity'],
            epsilon=state['epsilon'],
            merge_similarity=state.get('merge_similarity', 0.995))
        entries = deepcopy(state['entries'])
        for entry in entries:
            if any(key.lower() in {'x', 'y', 'inputs', 'targets',
                                   'input_window', 'target_window'}
                   for key in entry):
                raise ValueError('distribution checkpoint contains raw replay samples')
            mean = entry.get('mean')
            log_variance = entry.get('log_variance')
            if (not torch.is_tensor(mean) or not torch.is_tensor(log_variance) or
                    mean.ndim != 1 or log_variance.shape != mean.shape or
                    not torch.isfinite(mean).all() or
                    not torch.isfinite(log_variance).all()):
                raise ValueError('distribution checkpoint contains an invalid summary')
            if int(entry.get('sample_count', 0)) < 1:
                raise ValueError('distribution checkpoint sample_count must be positive')
            if not isinstance(entry.get('task_id'), (str, int, float)):
                raise ValueError('distribution checkpoint task_id is invalid')
            temporal_signature = entry.get('temporal_signature')
            if temporal_signature is not None and (
                    not torch.is_tensor(temporal_signature) or
                    temporal_signature.ndim > 1 or
                    not torch.isfinite(temporal_signature).all()):
                raise ValueError('distribution checkpoint temporal signature is invalid')
            for key, value in entry.items():
                if torch.is_tensor(value) and key not in (
                        'mean', 'log_variance', 'temporal_signature'):
                    raise ValueError('distribution checkpoint contains unsupported tensors')
                if key not in ('mean', 'log_variance', 'temporal_signature') and not isinstance(
                        value, (str, int, float, bool, type(None))):
                    raise ValueError('distribution checkpoint metadata must be scalar')
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError('distribution checkpoint metadata is non-finite')
        loaded.entries = entries
        for entry in loaded.entries:
            entry.setdefault('occurrences', 1)
            entry.setdefault('last_task_id', entry['task_id'])
            entry['schema_version'] = self.schema_version
        loaded._evict_if_needed()
        self.__dict__.update(loaded.__dict__)
