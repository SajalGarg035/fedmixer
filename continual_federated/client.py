import torch
import torch.nn.functional as F
from copy import deepcopy

from .distribution_memory import DistributionMemory
from .drift import MultiscaleSpectralTrendDetector
from .memory import OrthogonalSubspaceProjector, SpectralCoresetMemory


class ContinualClient:
    """Client-side continual training wrapper around an unchanged forecasting model."""

    def __init__(self, model, optimizer, device, memory_size=200,
                 memory_batch_size=16, memory_weight=0.1, distill_weight=0.1,
                 drift_threshold=0.15, trend_threshold=0.2,
                 replay_enabled=True, distribution_memory=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.replay_enabled = replay_enabled
        self.memory_size = memory_size
        self.memory = None
        self.distribution_memory = distribution_memory
        self.previous_teacher = None
        self.subspace_projector = OrthogonalSubspaceProjector()
        self.memory_batch_size = memory_batch_size
        self.memory_weight = memory_weight
        self.distill_weight = distill_weight
        downsampling_window = getattr(getattr(model, 'configs', None),
                                      'down_sampling_window', 2)
        self.drift_detector = MultiscaleSpectralTrendDetector(
            seasonal_threshold=drift_threshold,
            trend_threshold=trend_threshold,
            downsampling_window=max(1, downsampling_window))
        self.last_drift_distance = 0.0
        self.last_signatures = (None, None)
        self.last_drift = (False, False)
        self.last_seasonal_drift = False
        self.last_trend_drift = False
        self._seasonal_signature_sum = None
        self._trend_signature_sum = None
        self._round_sample_count = 0
        self.last_drift_distances = (0.0, 0.0)

    def begin_round(self):
        self.last_seasonal_drift = False
        self.last_trend_drift = False
        self.last_drift = (False, False)
        self.last_signatures = (None, None)
        self._seasonal_signature_sum = None
        self._trend_signature_sum = None
        self._round_sample_count = 0

    def accumulate_drift(self, inputs):
        seasonal, trend = self.drift_detector.signatures(inputs)
        count = int(inputs.shape[0])
        if self._seasonal_signature_sum is None:
            self._seasonal_signature_sum = torch.zeros_like(seasonal)
            self._trend_signature_sum = torch.zeros_like(trend)
        if (seasonal.shape != self._seasonal_signature_sum.shape or
                trend.shape != self._trend_signature_sum.shape):
            raise ValueError('drift signature dimensions changed within a round')
        self._seasonal_signature_sum.add_(seasonal, alpha=count)
        self._trend_signature_sum.add_(trend, alpha=count)
        self._round_sample_count += count

    def finalize_round_drift(self):
        if self._round_sample_count == 0:
            self.last_signatures = (None, None)
            self.last_drift = (False, False)
            return self.last_drift
        seasonal = self._seasonal_signature_sum / self._round_sample_count
        trend = self._trend_signature_sum / self._round_sample_count
        (seasonal_drift, trend_drift, seasonal, trend,
         seasonal_distance, trend_distance) = self.drift_detector.update_signatures(
            seasonal, trend)
        self.last_drift = (seasonal_drift, trend_drift)
        self.last_seasonal_drift = seasonal_drift
        self.last_trend_drift = trend_drift
        self.last_signatures = (seasonal, trend)
        self.last_drift_distance = float(max(seasonal_distance, trend_distance))
        self.last_drift_distances = (float(seasonal_distance), float(trend_distance))
        self._seasonal_signature_sum = None
        self._trend_signature_sum = None
        self._round_sample_count = 0
        return self.last_drift

    @property
    def persistent_drift_bytes(self):
        signatures = (*self.last_signatures,
                      self.drift_detector.previous_seasonal,
                      self.drift_detector.previous_trend)
        seen = set()
        total = 0
        for value in signatures:
            if torch.is_tensor(value):
                key = (value.device.type, value.device.index,
                       value.untyped_storage().data_ptr(),
                       value.untyped_storage().nbytes())
                if key not in seen:
                    seen.add(key)
                    total += value.untyped_storage().nbytes()
        return total

    @property
    def gradient_projection_bytes(self):
        tensors = [*self.subspace_projector._bases.values(),
                   *self.subspace_projector._pending.values()]
        return sum(value.numel() * value.element_size() for value in tensors)

    def observe(self, inputs):
        seasonal, trend, seasonal_signature, trend_signature = self.drift_detector.update(inputs)
        self.last_drift = (seasonal, trend)
        self.last_seasonal_drift = self.last_seasonal_drift or seasonal
        self.last_trend_drift = self.last_trend_drift or trend
        self.last_signatures = (seasonal_signature, trend_signature)
        self.last_drift_distance = float(seasonal or trend)
        return seasonal or trend

    def forward_with_scales(self, inputs, forward_fn):
        if hasattr(self.model, 'forecast_multiscale'):
            output = self.model.forecast_multiscale(inputs)
            prediction, scale_predictions = output[0], output[1]
            return prediction, torch.stack(scale_predictions, dim=1)
        return forward_fn(self.model, inputs), None

    @staticmethod
    def _representation(model, inputs, forward_fn):
        if hasattr(model, 'forward_with_representation'):
            return model.forward_with_representation(inputs)
        prediction = forward_fn(model, inputs)
        return prediction, prediction.reshape(prediction.shape[0], -1)

    def set_previous_teacher(self):
        self.previous_teacher = deepcopy(self.model).to(self.device)
        self.previous_teacher.eval()
        for parameter in self.previous_teacher.parameters():
            parameter.requires_grad_(False)

    def train_batch_replay_free(self, inputs, targets, forward_fn,
                                retention_weight=0.0, history_weight=0.3,
                                representation_weight=0.1,
                                distillation_inputs=None,
                                distillation_targets=None,
                                history_enabled=True,
                                representation_enabled=True,
                                mutual_enabled=True):
        if not torch.isfinite(torch.tensor(retention_weight)) or not 0 <= retention_weight <= 1:
            raise ValueError('retention_weight must be finite and in [0, 1]')
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        predictions, representation = self._representation(
            self.model, inputs, forward_fn)
        forecast_loss = F.mse_loss(predictions, targets)
        history_loss = forecast_loss.new_zeros(())
        representation_loss = forecast_loss.new_zeros(())
        if self.previous_teacher is not None and (history_enabled or representation_enabled):
            with torch.no_grad():
                teacher_predictions, teacher_representation = self._representation(
                    self.previous_teacher, inputs, forward_fn)
            if history_enabled:
                history_loss = F.mse_loss(predictions, teacher_predictions.detach())
            if representation_enabled:
                representation_loss = self._moment_loss(
                    representation, teacher_representation.detach())
        weighted_history = history_weight * retention_weight * history_loss
        weighted_representation = (representation_weight * retention_weight
                                   * representation_loss)
        loss = forecast_loss + weighted_history + weighted_representation
        mutual_loss = forecast_loss.new_zeros(())
        if (mutual_enabled and distillation_inputs is not None
                and distillation_targets is not None):
            anchor_predictions, _ = self._representation(
                self.model, distillation_inputs.to(self.device), forward_fn)
            if isinstance(distillation_targets, tuple):
                distillation_targets = distillation_targets[0]
            mutual_loss = F.mse_loss(
                anchor_predictions, distillation_targets.to(self.device))
            loss = loss + self.distill_weight * mutual_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('replay-free training loss is not finite')
        loss.backward()
        self.optimizer.step()
        return {
            'total': loss.detach().item(),
            'forecast': forecast_loss.detach().item(),
            'history': history_loss.detach().item(),
            'representation': representation_loss.detach().item(),
            'mutual': mutual_loss.detach().item(),
            'weighted_history': weighted_history.detach().item(),
            'weighted_representation': weighted_representation.detach().item(),
            'weighted_mutual': (self.distill_weight * mutual_loss).detach().item(),
        }

    @staticmethod
    def _moment_loss(student, teacher):
        student = student.reshape(student.shape[0], -1)
        teacher = teacher.reshape(teacher.shape[0], -1)
        if student.shape[1] != teacher.shape[1]:
            raise ValueError('student and teacher representation widths differ')
        mean_loss = F.mse_loss(student.mean(dim=0), teacher.mean(dim=0))
        student_log_var = student.var(dim=0, unbiased=False).clamp_min(1e-6).log()
        teacher_log_var = teacher.var(dim=0, unbiased=False).clamp_min(1e-6).log()
        return mean_loss + 0.5 * F.mse_loss(student_log_var, teacher_log_var)

    def train_batch(self, inputs, targets, forward_fn, distillation_inputs=None,
                    distillation_targets=None, use_replay=True,
                    mutual_enabled=True):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        if self.last_drift[0] or self.last_drift[1]:
            self.subspace_projector.consolidate_pending()
        predictions, scale_predictions = self.forward_with_scales(inputs, forward_fn)
        loss = F.mse_loss(predictions, targets)

        if use_replay and self.replay_enabled and self.memory is None:
            self.memory = SpectralCoresetMemory(self.memory_size)
        if use_replay and self.memory is not None:
            replay = self.memory.sample(self.memory_batch_size, self.device)
            if replay is not None:
                replay_inputs, replay_targets, old_predictions = replay
                replay_predictions, replay_scales = self.forward_with_scales(
                    replay_inputs, forward_fn)
                loss = loss + self.memory_weight * F.mse_loss(replay_predictions, replay_targets)
                if old_predictions is not None:
                    old_predictions = old_predictions.to(self.device)
                    current_output = replay_scales if replay_scales is not None else replay_predictions
                    loss = loss + self.memory_weight * F.mse_loss(current_output, old_predictions)

        if (mutual_enabled and distillation_inputs is not None
                and distillation_targets is not None):
            distill_predictions, distill_scales = self.forward_with_scales(
                distillation_inputs.to(self.device), forward_fn)
            target_predictions = distillation_targets
            target_scales = None
            if isinstance(distillation_targets, tuple):
                target_predictions, target_scales = distillation_targets
            loss = loss + self.distill_weight * F.mse_loss(
                distill_predictions, target_predictions.to(self.device))
            if distill_scales is not None and target_scales is not None:
                loss = loss + self.distill_weight * F.mse_loss(
                    distill_scales, target_scales.to(self.device))

        loss.backward()
        if not (self.last_drift[0] or self.last_drift[1]):
            self.subspace_projector.accumulate(self.model.named_parameters())
        # A drift batch must retain its new-regime gradient. Projecting it
        # against the just-consolidated stable basis can remove all usable
        # directions for small parameters and prevent adaptation.
        if not (self.last_drift[0] or self.last_drift[1]):
            self.subspace_projector.project_gradients(self.model.named_parameters())
        self.optimizer.step()
        if use_replay and self.memory is not None:
            old_output = scale_predictions if scale_predictions is not None else predictions
            self.memory.add_batch(inputs, targets, old_output)
        return loss.detach().item()
