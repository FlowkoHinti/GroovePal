# loss function for masked language modelling
# insert mask tokens into sequence and let model predict sequence without mask tokens I_mask -> I
# teacher forcing can be a great optimisation as well
# add loss / training function for the various setups

import math
import torch
import torch.nn as nn
import torch.optim as optim
from dacite import from_dict, Config as DaciteConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics import Perplexity
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

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


class MultiTaskDNALearner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = setup_logger("MultiTaskDNALearner")
        self.device = torch.device(self.cfg.train.device if torch.cuda.is_available() else "cpu")

        self._setup_dataset()
        self._setup_embedding()
        self._setup_model()
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

        self.model = MultiTaskDNAxLSTM(model_config, self.embedding_config).to(self.device)

    def _setup_training_objects(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.train.initial_lr)

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.cfg.train.get("lr_step_size", 10),
            gamma=self.cfg.train.get("lr_gamma", 0.5)
        ) if self.cfg.train.get("use_scheduler", True) else None

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
        metric_dict = {
            'accuracy': MulticlassAccuracy,
            'precision': MulticlassPrecision,
            'recall': MulticlassRecall,
            'f1': MulticlassF1Score,
            'perplexity': Perplexity,
        }

        results = {}

        # Loop through each head
        for i, (head, logits) in enumerate(logits_dict.items()):
            head_results = {}

            # Get predictions as class indices
            pred_classes = torch.argmax(logits, dim=-1)  # Shape: (B, T)
            target_classes = targets[:, :, i]  # Shape: (B, T)

            pred_flat = pred_classes.view(-1)  # Flatten for metric calculation
            target_flat = target_classes.view(-1)

            # Compute each requested metric
            # TODO: HANDLING FOR METRICS WITHOUT NUM CLASSES
            for metric_name in self.learner.eval_metrics:
                metric_cls = metric_dict[metric_name]
                metric = metric_cls(num_classes=logits.size(-1)).to(self.device)
                value = metric(pred_flat.to(self.device), target_flat.to(self.device))
                head_results[f"{head}/{metric_name}"] = value.item() if hasattr(value, 'item') else float(value)

            results.update(head_results)

        return results

    def train(self):
        run_training_loop(
            learner=self.learner,
            callback_manager=self.callback_manager,
            device=self.device,
            compute_loss_fn=self.compute_loss,
            compute_metrics_fn=self.compute_metrics,
            logger=self.logger
        )

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.logger.info("Starting test evaluation...")

        pass
