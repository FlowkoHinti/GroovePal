from abc import ABC, abstractmethod

import torch
from omegaconf import DictConfig


def sort_params_by_name(model, params):
    """Utility to keep the order of params, since xLSTM implementation uses sets.
    Makes Params deterministic"""

    name_by_param = {p: n for n, p in model.named_parameters()}
    params = [p for p in params if getattr(p, "requires_grad", True)]
    return sorted(params, key=lambda p: name_by_param.get(p, ""))


class BaseDNALearner(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(self.cfg.train.device if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def _setup_dataset(self):
        """Set up training/validation/test datasets and data loaders."""
        pass

    @abstractmethod
    def _setup_embedding(self):
        """Initialize embedding configurations."""
        pass

    @abstractmethod
    def _setup_model(self):
        """Instantiate and initialize the model."""
        pass

    @abstractmethod
    def _setup_criterion(self):
        """Instantiate and initialize the loss function."""
        pass

    @abstractmethod
    def _setup_optimizer(self):
        """Initializes the optimizer with weight decay applied to appropriate parameters."""
        pass

    @abstractmethod
    def _setup_training_objects(self):
        """Set up optimizer, scheduler, loss function, and learner state."""
        pass

    @abstractmethod
    def _setup_callbacks(self):
        """Register callbacks used during training."""
        pass

    @abstractmethod
    def compute_loss(self, logits, targets) -> torch.Tensor:
        """Compute and return the training loss."""
        pass

    @abstractmethod
    def train(self):
        """Main training loop."""
        pass

    @abstractmethod
    def test(self):
        """Evaluation on the test set."""
        pass
