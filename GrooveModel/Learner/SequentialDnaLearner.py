from pathlib import Path

import torch
from dacite import from_dict, Config as DaciteConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from GrooveModel import CollateFunctions
from GrooveModel.Callbacks.Callback import CallbackManager
from GrooveModel.Callbacks.CheckpointCallback import CheckpointCallback
from GrooveModel.Callbacks.EarlyStoppingCallback import EarlyStoppingCallback
from GrooveModel.Callbacks.EpochSummaryCallback import EpochSummaryCallback
from GrooveModel.Callbacks.GradientClippingCallback import GradientClippingCallback
from GrooveModel.Callbacks.LRLoggerCallback import LRLoggerCallback
from GrooveModel.Callbacks.PlotLossCurveCallback import PlotLossCurvesCallback
from GrooveModel.Callbacks.PlotMetricsCallback import PlotMetricsCallback
from GrooveModel.Datasets import DNANextTokenDataset
from GrooveModel.Embedding.SequentialDnaEmbedding import SequentialDNAEmbeddingConfig
from GrooveModel.Learner.Learner import BaseDNALearner, sort_params_by_name
from GrooveModel.Learner.LearnerState import LearnerState
from GrooveModel.Metrics import SequentialDNAMetrics
from GrooveModel.Models import ModelConfigxLstm, SequentialDNAxLSTM
from GrooveModel.Tokenizer.SequentialDnaTokenizer import SequentialDnaTokenizer
from GrooveModel.TrainLoop import run_training_loop
from GrooveModel.Utils.Logger import setup_logger
from GrooveModel.Utils.SpecialTokens import SpecialTokens
from GrooveModel.xlstm.experiments.lr_scheduler import LinearWarmupCosineAnnealing


class SequentialDnaLearner(BaseDNALearner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.logger = setup_logger("SequentialDnaLearner")

        self._setup_dataset()
        self._setup_embedding()
        self._setup_model()
        self._setup_criterion()
        self._setup_optimizer()
        self._setup_training_objects()
        self._setup_callbacks()

    def _setup_dataset(self):
        ds_conf = self.cfg.dataset

        self.train_dataset = DNANextTokenDataset(
            ds_conf, split="train", tokenizer=SequentialDnaTokenizer
        )
        self.val_dataset = DNANextTokenDataset(
            ds_conf, split="validation", tokenizer=SequentialDnaTokenizer
        )
        self.test_dataset = DNANextTokenDataset(
            ds_conf, split="test", tokenizer=SequentialDnaTokenizer
        )

        num_workers = ds_conf.get("num_workers", 4)
        use_persistent = ds_conf.get("persistent_workers", True) and num_workers > 0

        common_loader_args = {
            "batch_size": self.cfg.train.batch_size,
            "num_workers": num_workers,
            "persistent_workers": use_persistent,
            "pin_memory": ds_conf.get("pin_memory", True),
            "collate_fn": CollateFunctions.pad_truncate_batch,
            "drop_last": ds_conf.get("drop_last", True),
        }

        # Only set prefetch_factor when workers are used
        if num_workers > 0:
            common_loader_args["prefetch_factor"] = ds_conf.get("prefetch_factor", 1)

        if self.device.type == "cuda":
            common_loader_args["pin_memory_device"] = str(self.device)

        self.train_loader = DataLoader(self.train_dataset, shuffle=ds_conf.get("shuffle", True), **common_loader_args)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **common_loader_args)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **common_loader_args)

    def _setup_embedding(self):
        # inject vocab size
        vocab_size = len(SequentialDnaTokenizer.vocab)
        self.cfg.embedding.vocab_size = vocab_size

        self.embedding_config = from_dict(
            SequentialDNAEmbeddingConfig,
            OmegaConf.to_container(self.cfg.embedding, resolve=True),
            config=DaciteConfig(strict=True)
        )

    def _setup_model(self):
        model_config = from_dict(
            ModelConfigxLstm,
            OmegaConf.to_container(self.cfg.model, resolve=True),
            config=DaciteConfig(strict=True)
        )

        self.model = SequentialDNAxLSTM(model_config,
                                        self.embedding_config,
                                        absolute_grid_units=self.cfg.dataset.tokenizer.absolute_grid_units)
        # Load pretrained weights if given
        if self.cfg.train.get("finetune", None):
            pretrained_path = Path(
                self.cfg.train.save_dir) / f"{self.cfg.train.finetune}" / "checkpoints" / f"{self.cfg.train.finetune}_best.pt"
            self.logger.info(f"Finetuning pretrained model from {pretrained_path}")
            pretrained = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(pretrained["model_state_dict"])
        else:
            self.model.reset_parameters()
        self.model.to(self.device)

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=SpecialTokens.PAD, label_smoothing=0.1)

    def _setup_optimizer(self):
        decay, no_decay = self.model._create_weight_decay_optim_groups()
        # make order deterministic going forward
        decay = sort_params_by_name(self.model, list(decay))
        no_decay = sort_params_by_name(self.model, list(no_decay))
        loss_params = [p for p in self.criterion.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.train.weight_decay, "name": "decay"},
                {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"},
                {"params": loss_params, "weight_decay": 0.0, "name": "loss_weights"},
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

        self.metrics = SequentialDNAMetrics(
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
                loss_label="Cross Entropy Loss",
                logger=self.logger
            ),
            PlotMetricsCallback(
                save_dir=self.cfg.train.save_dir,
                model_name=self.cfg.train.model_name,
                logger=self.logger,
                head_colors={
                    'instrument': 'crimson',
                    'velocity': 'dodgerblue',
                    'offset': 'limegreen',
                    'beat_unit': 'goldenrod',
                    'grid_factor': 'rebeccapurple',
                    'bpm': 'chocolate',
                    'time_signature': 'seagreen',
                }
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

    def compute_loss(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)
        return self.criterion(outputs_flat, targets_flat)

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
