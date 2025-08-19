from dataclasses import dataclass, field
from typing import Optional, Dict, List

from torch import nn, optim
from torch.utils.data import DataLoader


@dataclass
class LearnerState:
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[optim.lr_scheduler._LRScheduler]
    train_loader: DataLoader
    val_loader: DataLoader
    criterion: nn.Module
    max_epochs: int
    step_based_scheduler: bool
    eval_metrics: List[str] = field(default_factory=list)

    # Dynamic values updated during training
    start_epoch: int = 0
    epoch: int = 0
    global_step: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
