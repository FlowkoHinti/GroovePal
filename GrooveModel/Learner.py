# loss function for masked language modelling
# insert mask tokens into sequence and let model predict sequence without mask tokens I_mask -> I
# teacher forcing can be a great optimisation as well
# add loss / training function for the various setups
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from dacite import from_dict, Config as DaciteConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.text import Perplexity

from GrooveModel import Tokenizers, CollateFunctions
from GrooveModel.Callbacks import CheckpointCallback, EarlyStoppingCallback, PlotLossCurvesCallback, \
    PlotMetricsCallback, LRLoggerCallback, GradientClippingCallback, EpochSummaryCallback, CallbackManager
from GrooveModel.Datasets import DNANextTokenDataset
from GrooveModel.Embeddings import MultiTaskDNAEmbeddingConfig
from GrooveModel.LearnerState import LearnerState
from GrooveModel.Models import MultiTaskDNAxLSTM, MultiTaskDNAModelConfig
from GrooveModel.TrainLoop import run_training_loop
from GrooveModel.Utils.Logger import setup_logger
from GrooveModel.Utils.SpecialTokens import SpecialTokens
from GrooveModel.xlstm.experiments.lr_scheduler import LinearWarmupCosineAnnealing


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
    def compute_metrics(self, logits, targets) -> dict[str, float]:
        """Compute and return evaluation metrics."""
        pass

    @abstractmethod
    def train(self):
        """Main training loop."""
        pass

    @abstractmethod
    def test(self):
        """Evaluation on the test set."""
        pass


class MultiTaskDNALearner(BaseDNALearner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.logger = setup_logger("MultiTaskDNALearner")

        self._setup_dataset()
        self._setup_embedding()
        self._setup_model()
        self._setup_optimizer()
        self._setup_training_objects()
        self._setup_callbacks()

    def _setup_dataset(self):
        ds_conf = self.cfg.dataset

        self.train_dataset = DNANextTokenDataset(
            ds_conf, split="train", tokenizer=Tokenizers.MultiTaskDNATokenizer
        )
        self.val_dataset = DNANextTokenDataset(
            ds_conf, split="validation", tokenizer=Tokenizers.MultiTaskDNATokenizer
        )
        # self.test_dataset = DNANextTokenDataset(
        #     ds_conf, split="test", tokenizer=Tokenizers.MultiTaskDNATokenizer
        # )

        common_loader_args = {
            "batch_size": self.cfg.train.batch_size,
            "num_workers": ds_conf.get("num_workers", 4),
            "collate_fn": CollateFunctions.pad_truncate_batch,
        }

        self.train_loader = DataLoader(self.train_dataset, shuffle=ds_conf.get("shuffle", True), **common_loader_args)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **common_loader_args)
        # self.test_loader = DataLoader(self.test_dataset, shuffle=False, **common_loader_args)

    def _setup_embedding(self):
        self.embedding_config = from_dict(
            MultiTaskDNAEmbeddingConfig,
            OmegaConf.to_container(self.cfg.embedding, resolve=True),
            config=DaciteConfig(strict=True)
        )

    def _setup_model(self):
        model_config = from_dict(
            MultiTaskDNAModelConfig,
            OmegaConf.to_container(self.cfg.model, resolve=True),
            config=DaciteConfig(strict=True)
        )

        self.model = MultiTaskDNAxLSTM(model_config, self.embedding_config)
        self.model.reset_parameters()
        self.model.to(self.device)

    def _setup_optimizer(self):
        # TODO: CHECK IF THIS WORKS
        optim_groups = self.model._create_weight_decay_optim_groups()

        self.optimizer = torch.optim.AdamW(
            [
                {"params": optim_groups[0], "weight_decay": self.cfg.train.weight_decay},
                {"params": optim_groups[1], "weight_decay": 0.0},
            ],
            lr=self.cfg.train.initial_lr,
        )

    def _setup_training_objects(self):
        match self.cfg.train.lr_scheduler.get("type", None):
            case "linear_warmup_cosine_annealing":
                steps_per_epoch = math.ceil(len(self.train_dataset) / self.cfg.train.batch_size)
                total_steps = steps_per_epoch * self.cfg.train.epochs
                warmup_steps = int(self.cfg.train.lr_scheduler.lr_warmup_steps_ratio * total_steps)

                self.scheduler = LinearWarmupCosineAnnealing(
                    optimizer=self.optimizer,
                    warmup_steps=warmup_steps,
                    decay_until_step=total_steps,
                    max_lr=self.cfg.train.initial_lr,
                    min_lr=self.cfg.train.lr_scheduler.lr_decay_factor * self.cfg.train.initial_lr,
                )

            case "step_lr":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=self.cfg.train.lr_scheduler.get("lr_step_size", 10),
                    gamma=self.cfg.train.lr_scheduler.get("lr_gamma", 0.5)
                )

            case _:
                self.scheduler = None

        self.criterion = nn.CrossEntropyLoss(ignore_index=SpecialTokens.PAD)

        self.learner = LearnerState(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=self.criterion,
            max_epochs=self.cfg.train.epochs,
            eval_metrics=self.cfg.train.metrics,
        )

    def _setup_callbacks(self):
        self.callbacks = [
            CheckpointCallback(
                save_dir=self.cfg.train.save_dir,
                model_name=self.cfg.train.model_name,
                logger=self.logger,
                monitor=self.cfg.train.get("monitor", "val_loss"),
                mode=self.cfg.train.get("mode", "min"),
                load_best=self.cfg.train.get("load_best", False),
            ),
            EarlyStoppingCallback(
                patience=self.cfg.train.get("patience", 5),
                logger=self.logger
            ),
            PlotLossCurvesCallback(
                save_dir=self.cfg.train.save_dir,
                model_name=self.cfg.train.model_name,
                logger=self.logger
            ),
            PlotMetricsCallback(
                save_dir=self.cfg.train.save_dir,
                model_name=self.cfg.train.model_name,
                logger=self.logger
            ),
            LRLoggerCallback(logger=self.logger),
            GradientClippingCallback(
                max_norm=self.cfg.train.get("gradient_clip_norm", 1.0),
                logger=self.logger
            ),
            EpochSummaryCallback(
                save_dir=self.cfg.train.save_dir,
                model_name=self.cfg.train.model_name,
                logger=self.logger
            )
        ]
        self.callback_manager = CallbackManager(self.callbacks)

    def compute_loss(self, logits_dict: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        num_heads = len(logits_dict)

        # Compute and accumulate loss per head
        for i, (head, logits) in enumerate(logits_dict.items()):
            logits = logits.view(-1, logits.size(-1))  # Flatten logits: (B*T, C)
            target = targets[:, :, i].view(-1)  # Flatten target: (B*T,)
            loss = self.learner.criterion(logits, target)  # Compute loss for this head
            total_loss += loss

        return total_loss / num_heads

    def compute_metrics(self, logits_dict: dict[str, torch.Tensor], targets: torch.Tensor) -> dict[str, float]:
        results = {}
        metric_sums = {}
        metric_counts = {}

        for i, (head, logits) in enumerate(logits_dict.items()):
            head_results = {}

            target_classes = targets[:, :, i]  # (B, T)
            target_flat = target_classes.view(-1)
            logits_flat = logits.view(-1, logits.size(-1))  # (N, V)

            for metric_name in self.learner.eval_metrics:
                if metric_name.startswith("top_k_accuracy@"):
                    k = int(metric_name.split("@")[1])
                    metric = MulticlassAccuracy(num_classes=logits.size(-1), top_k=k).to(self.device)
                    value = metric(logits_flat.to(self.device), target_flat.to(self.device))

                elif metric_name == "perplexity":
                    metric = Perplexity(ignore_index=SpecialTokens.PAD).to(self.device)
                    value = metric(logits.to(self.device), target_classes.to(self.device))

                else:
                    raise ValueError(f"Unsupported metric: {metric_name}")

                value_scalar = value.item() if hasattr(value, 'item') else float(value)
                head_results[f"{head}/{metric_name}"] = value_scalar

                metric_sums.setdefault(metric_name, 0.0)
                metric_counts.setdefault(metric_name, 0)
                metric_sums[metric_name] += value_scalar
                metric_counts[metric_name] += 1

            results.update(head_results)

        for metric_name in metric_sums:
            avg_value = metric_sums[metric_name] / metric_counts[metric_name]
            results[f"avg_{metric_name}"] = avg_value

        return results

    def train(self):
        run_training_loop(
            learner=self.learner,
            callback_manager=self.callback_manager,
            device=self.device,
            compute_loss_fn=self.compute_loss,
            compute_metrics_fn=self.compute_metrics,
            logger=self.logger,
            use_mixed_precision=self.cfg.train.get("use_mixed_precision", False),
        )

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.logger.info("Starting test evaluation...")

        pass
