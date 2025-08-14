# loss function for masked language modelling
# insert mask tokens into sequence and let model predict sequence without mask tokens I_mask -> I
# teacher forcing can be a great optimisation as well

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from dacite import from_dict, Config as DaciteConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from GrooveModel import Tokenizers, CollateFunctions
from GrooveModel.Callbacks import CheckpointCallback, EarlyStoppingCallback, PlotLossCurvesCallback, \
    PlotMetricsCallback, LRLoggerCallback, GradientClippingCallback, EpochSummaryCallback, CallbackManager
from GrooveModel.Datasets import DNANextTokenDataset
from GrooveModel.Embeddings import MultiTaskDNAEmbeddingConfig
from GrooveModel.LearnerState import LearnerState
from GrooveModel.Metrics import MultiTaskDNAMetrics
from GrooveModel.Models import MultiTaskDNAxLSTM, MultiTaskDNAModelConfig
from GrooveModel.TrainLoop import run_training_loop
from GrooveModel.Utils.Logger import setup_logger
from GrooveModel.Utils.SpecialTokens import SpecialTokens
from GrooveModel.xlstm.experiments.lr_scheduler import LinearWarmupCosineAnnealing


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
            ds_conf, split="train", tokenizer=Tokenizers.MultiTaskDnaTokenizer
        )
        self.val_dataset = DNANextTokenDataset(
            ds_conf, split="validation", tokenizer=Tokenizers.MultiTaskDnaTokenizer
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
        # use your existing group creator (don’t modify it)
        decay, no_decay = self.model._create_weight_decay_optim_groups()
        # make order deterministic going forward
        decay = sort_params_by_name(self.model, list(decay))
        no_decay = sort_params_by_name(self.model, list(no_decay))

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.train.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.train.initial_lr,
        )

    def _setup_training_objects(self):
        # pick scheduler
        sched_type = self.cfg.train.lr_scheduler.get("type", None)
        if sched_type == "linear_warmup_cosine_annealing":
            steps_per_epoch = len(self.train_loader)
            total_steps = steps_per_epoch * self.cfg.train.epochs
            warmup_steps = int(self.cfg.train.lr_scheduler.lr_warmup_steps_ratio * total_steps)

            self.scheduler = LinearWarmupCosineAnnealing(
                optimizer=self.optimizer,
                warmup_steps=warmup_steps,
                decay_until_step=total_steps,
                max_lr=self.cfg.train.initial_lr,
                min_lr=self.cfg.train.lr_scheduler.lr_decay_factor * self.cfg.train.initial_lr,
            )
            step_based = True
        elif sched_type == "step_lr":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.train.lr_scheduler.get("lr_step_size", 10),
                gamma=self.cfg.train.lr_scheduler.get("lr_gamma", 0.5),
            )
            step_based = False
        else:
            self.scheduler = None
            step_based = False

        self.criterion = nn.CrossEntropyLoss(ignore_index=SpecialTokens.PAD)

        self.metrics = MultiTaskDNAMetrics(
            metric_names=self.cfg.train.metrics,
            device=self.device,
            ignore_index=SpecialTokens.PAD,
        )

        self.learner = LearnerState(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=self.criterion,
            max_epochs=self.cfg.train.epochs,
            step_based_scheduler=step_based,
            eval_metrics=self.cfg.train.metrics,
        )

    def _setup_callbacks(self):
        self.callbacks = [
            CheckpointCallback(
                save_dir=self.cfg.train.save_dir,
                model_name=self.cfg.train.model_name,
                device=self.device,
                logger=self.logger,
                monitor=self.cfg.train.get("monitor", "val_loss"),
                mode=self.cfg.train.get("mode", "min"),
                load_best_on_start=(self.cfg.train.get("load_best_on_start", False)),
            ),
            EarlyStoppingCallback(
                monitor=self.cfg.train.get("monitor", "val_loss"),
                patience=self.cfg.train.get("patience", 5),
                min_delta=self.cfg.train.get("min_delta", 0.0),
                mode=self.cfg.train.get("mode", "min"),
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
        # TODO: MAYBE CALCULATE DIFFERENTLY PER HEAD -> close velocities are ok -> instruments should be correctly matched
        # TODO: MAYBE DONT AVG THE LOSS OVER ALL HEADS BUT ADD WEIGHTS TO EACH SEPARATELY
        total_loss = 0.0
        num_heads = len(logits_dict)

        # Compute and accumulate loss per head
        for i, (head, logits) in enumerate(logits_dict.items()):
            logits = logits.view(-1, logits.size(-1))  # Flatten logits: (B*T, C)
            target = targets[:, :, i].view(-1)  # Flatten target: (B*T,)
            loss = self.learner.criterion(logits, target)  # Compute loss for this head
            total_loss += loss

        return total_loss / num_heads

    def train(self):
        run_training_loop(
            learner=self.learner,
            callback_manager=self.callback_manager,
            device=self.device,
            compute_loss_fn=self.compute_loss,
            metrics=self.metrics,
            logger=self.logger,
            use_mixed_precision=self.cfg.train.get("use_mixed_precision", False),
        )

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.logger.info("Starting test evaluation...")

        pass
