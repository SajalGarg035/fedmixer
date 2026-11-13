import torch


class MeanDriftDetector:
    """Detect a shift in the mean of consecutive time-series chunks."""

    def __init__(self, threshold=1.0):
        if threshold < 0:
            raise ValueError('threshold must be non-negative')
        self.threshold = threshold
        self.previous_mean = None

    def update(self, current_chunk):
        current_mean = current_chunk.detach().float().mean(
            dim=tuple(range(current_chunk.ndim - 1))).cpu()
        drift = False
        distance = 0.0
        if self.previous_mean is not None:
            distance = torch.linalg.vector_norm(current_mean - self.previous_mean).item()
            drift = distance > self.threshold
        self.previous_mean = current_mean.cpu()
        return drift, distance

    def reset(self):
        self.previous_mean = None


class MultiscaleSpectralTrendDetector:
    """Detect frequency and coarse trend changes in streaming windows."""

    def __init__(self, seasonal_threshold=0.15, trend_threshold=0.2,
                 scales=3, downsampling_window=2, trend_order=2,
                 minimum_length=16):
        if seasonal_threshold < 0 or trend_threshold < 0:
            raise ValueError('drift thresholds must be non-negative')
        if scales < 1 or downsampling_window < 1 or trend_order < 1:
            raise ValueError('scales, downsampling_window, and trend_order must be positive')
        self.seasonal_threshold = seasonal_threshold
        self.trend_threshold = trend_threshold
        self.scales = scales
        self.downsampling_window = downsampling_window
        self.trend_order = trend_order
        self.minimum_length = minimum_length
        self.previous_seasonal = None
        self.previous_trend = None

    @staticmethod
    def _seasonal_trend(values):
        if values.shape[1] < 3:
            return values - values.mean(dim=1, keepdim=True), values
        kernel = min(5, values.shape[1] if values.shape[1] % 2 else values.shape[1] - 1)
        kernel = max(3, kernel)
        padding = kernel // 2
        padded = torch.nn.functional.pad(
            values.transpose(1, 2), (padding, padding), mode='replicate')
        trend = torch.nn.functional.avg_pool1d(padded, kernel, stride=1)
        trend = trend.transpose(1, 2)
        return values - trend, trend

    def _multiscale(self, values):
        current = values.float()
        result = []
        for _ in range(self.scales):
            result.append(current)
            if current.shape[1] < self.downsampling_window * 2:
                break
            current = torch.nn.functional.avg_pool1d(
                current.transpose(1, 2), self.downsampling_window,
                stride=self.downsampling_window).transpose(1, 2)
        return result

    @staticmethod
    def _spectral_signature(seasonal):
        spectrum = torch.fft.rfft(seasonal, dim=1).abs().square().mean(dim=(0, 2))
        spectrum = spectrum[1:]
        if spectrum.numel() == 0:
            return torch.zeros(1)
        return spectrum / spectrum.sum().clamp_min(1e-8)

    def _trend_signature(self, trend):
        length = trend.shape[1]
        positions = torch.linspace(-1, 1, length, device=trend.device)
        basis = [torch.ones_like(positions)]
        for degree in range(1, self.trend_order + 1):
            basis.append(positions ** degree)
        basis = torch.stack(basis, dim=1)
        coefficients = torch.einsum('btc,tp->bpc', trend, basis)
        # Keep polynomial degree and channel identity. Averaging those axes
        # collapses univariate trend signatures to one scalar and misses most
        # changes in slope, curvature, and level.
        return coefficients.mean(dim=0).flatten()

    @staticmethod
    def _wasserstein(first, second):
        length = max(first.numel(), second.numel())
        first = torch.nn.functional.pad(first, (0, length - first.numel()))
        second = torch.nn.functional.pad(second, (0, length - second.numel()))
        return torch.cumsum(first, dim=0).sub(torch.cumsum(second, dim=0)).abs().mean()

    def signatures(self, current_chunk):
        values = current_chunk.detach().float()
        if values.ndim == 2:
            values = values.unsqueeze(-1)
        if values.ndim != 3:
            raise ValueError('current_chunk must have shape [batch, time, channels]')
        seasonal_signatures = []
        trend_signatures = []
        for scale in self._multiscale(values):
            seasonal, trend = self._seasonal_trend(scale)
            seasonal_signatures.append(self._spectral_signature(seasonal))
            trend_signatures.append(self._trend_signature(trend))
        seasonal_signature = torch.cat(seasonal_signatures).cpu()
        trend_signature = torch.cat(trend_signatures).cpu()
        return seasonal_signature, trend_signature

    def update_signatures(self, seasonal_signature, trend_signature):
        seasonal_signature = torch.as_tensor(seasonal_signature).float().cpu()
        trend_signature = torch.as_tensor(trend_signature).float().cpu()
        seasonal_distance = 0.0
        trend_distance = 0.0
        seasonal_drift = False
        trend_drift = False
        if self.previous_seasonal is not None:
            seasonal_distance = self._wasserstein(
                seasonal_signature, self.previous_seasonal).item()
            seasonal_drift = seasonal_distance > self.seasonal_threshold
            trend_delta = torch.linalg.vector_norm(
                trend_signature - self.previous_trend)
            trend_scale = (torch.linalg.vector_norm(trend_signature) +
                           torch.linalg.vector_norm(self.previous_trend) + 1e-8)
            trend_distance = (trend_delta / trend_scale).item()
            trend_drift = trend_distance > self.trend_threshold
        # Keep a stable-regime anchor until a change crosses its threshold.
        # This lets small gradual shifts accumulate instead of chasing each
        # immediately preceding round. On detection, accept the new regime.
        if self.previous_seasonal is None or seasonal_drift:
            self.previous_seasonal = seasonal_signature.clone()
        if self.previous_trend is None or trend_drift:
            self.previous_trend = trend_signature.clone()
        return (seasonal_drift, trend_drift, seasonal_signature,
                trend_signature, seasonal_distance, trend_distance)

    def update(self, current_chunk):
        return self.update_signatures(*self.signatures(current_chunk))[:4]

    def reset(self):
        self.previous_seasonal = None
        self.previous_trend = None

    @property
    def byte_count(self):
        return sum(value.numel() * value.element_size()
                   for value in (self.previous_seasonal, self.previous_trend)
                   if torch.is_tensor(value))
