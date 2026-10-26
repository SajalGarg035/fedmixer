import torch
import torch.nn.functional as F

from .drift import MultiscaleSpectralTrendDetector
from .memory import OrthogonalSubspaceProjector, SpectralCoresetMemory


class ContinualClient:
    """Client-side continual training wrapper around an unchanged forecasting model."""

    def __init__(self, model, optimizer, device, memory_size=200,
                 memory_batch_size=16, memory_weight=0.1, distill_weight=0.1,
                 drift_threshold=0.15, trend_threshold=0.2):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.memory = SpectralCoresetMemory(memory_size)
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

    def begin_round(self):
        self.last_seasonal_drift = False
        self.last_trend_drift = False

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

    def train_batch(self, inputs, targets, forward_fn, distillation_inputs=None,
                    distillation_targets=None, use_replay=True):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        if self.last_drift[0] or self.last_drift[1]:
            self.subspace_projector.consolidate_pending()
        predictions, scale_predictions = self.forward_with_scales(inputs, forward_fn)
        loss = F.mse_loss(predictions, targets)

        if use_replay:
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

        if distillation_inputs is not None and distillation_targets is not None:
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
        self.subspace_projector.project_gradients(self.model.named_parameters())
        self.optimizer.step()
        if use_replay:
            old_output = scale_predictions if scale_predictions is not None else predictions
            self.memory.add_batch(inputs, targets, old_output)
        return loss.detach().item()