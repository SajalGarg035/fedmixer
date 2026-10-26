import torch


class SpectralCoresetMemory:
    """Replay memory that preserves spectrally diverse windows and old outputs."""

    def __init__(self, capacity=200):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self._items = []

    def __len__(self):
        return len(self._items)

    def add_batch(self, inputs, targets, logits=None):
        inputs = inputs.detach().cpu()
        targets = targets.detach().cpu()
        logits = None if logits is None else logits.detach().cpu()
        for index in range(inputs.shape[0]):
            item = (inputs[index].clone(), targets[index].clone())
            if logits is not None:
                item += (logits[index].clone(),)
            signature = self._signature(inputs[index])
            if len(self._items) < self.capacity:
                self._items.append((item, signature))
            else:
                self._items.append((item, signature))
                self._evict_redundant()

    @staticmethod
    def _signature(values, bins=16):
        values = values.float()
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        spectrum = torch.fft.rfft(values, dim=0).abs().mean(dim=-1)
        spectrum = spectrum[1:]
        if spectrum.numel() == 0:
            return torch.zeros(1)
        spectrum = spectrum / spectrum.norm().clamp_min(1e-8)
        if spectrum.numel() > bins:
            spectrum = spectrum[:bins]
        return torch.nn.functional.pad(spectrum, (0, max(0, bins - spectrum.numel())))

    def _evict_redundant(self):
        signatures = torch.stack([entry[1] for entry in self._items])
        similarity = signatures @ signatures.T
        similarity.fill_diagonal_(-1)
        redundant = similarity.max(dim=1).values.argmax().item()
        del self._items[redundant]

    def sample(self, batch_size, device):
        if not self._items:
            return None
        count = min(batch_size, len(self._items))
        indices = torch.randperm(len(self._items))[:count].tolist()
        items = [self._items[index][0] for index in indices]
        inputs = torch.stack([item[0] for item in items]).to(device)
        targets = torch.stack([item[1] for item in items]).to(device)
        logits = None
        if len(items[0]) == 3:
            logits = torch.stack([item[2] for item in items]).to(device)
        return inputs, targets, logits


ReplayMemory = SpectralCoresetMemory


class OrthogonalSubspaceProjector:
    """Bounded-rank gradient projection for retaining historical directions."""

    def __init__(self, max_rank=8, decay=0.9):
        if max_rank < 1 or not 0 < decay <= 1:
            raise ValueError('max_rank must be positive and decay must be in (0, 1]')
        self.max_rank = max_rank
        self.decay = decay
        self._bases = {}
        self._pending = {}

    @staticmethod
    def _gradient_basis(parameter, max_rank):
        if parameter.grad is None or parameter.grad.ndim == 0:
            return None
        matrix = parameter.grad.detach().float().reshape(parameter.shape[0], -1)
        _, _, vectors = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(max_rank, vectors.shape[0])
        return vectors[:rank].T.contiguous()

    def _merge_basis(self, name, basis, target):
        previous = target.get(name)
        if previous is not None:
            basis = torch.cat((previous * self.decay, basis), dim=1)
        basis, _ = torch.linalg.qr(basis, mode='reduced')
        target[name] = basis[:, :self.max_rank].detach()

    def consolidate(self, parameters):
        """Compatibility method for explicitly supplied historical gradients."""
        for name, parameter in parameters:
            basis = self._gradient_basis(parameter, self.max_rank)
            if basis is not None:
                self._merge_basis(name, basis, self._bases)

    def accumulate(self, parameters):
        """Collect directions from stable batches for later drift consolidation."""
        for name, parameter in parameters:
            basis = self._gradient_basis(parameter, self.max_rank)
            if basis is not None:
                self._merge_basis(name, basis, self._pending)

    def consolidate_pending(self):
        """Lock stable-regime directions before the first batch of a new regime."""
        for name, basis in self._pending.items():
            self._merge_basis(name, basis, self._bases)
        self._pending.clear()

    def project_gradients(self, parameters):
        for name, parameter in parameters:
            basis = self._bases.get(name)
            if basis is None or parameter.grad is None:
                continue
            matrix = parameter.grad.reshape(parameter.shape[0], -1)
            if matrix.shape[1] == basis.shape[0]:
                basis = basis.to(matrix.device)
                projection = torch.eye(matrix.shape[1], device=matrix.device)
                projection = projection - basis @ basis.T
                parameter.grad.copy_((matrix @ projection).reshape_as(parameter.grad))