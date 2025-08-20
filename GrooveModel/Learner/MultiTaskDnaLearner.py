import numpy as np
import torch
from dacite import from_dict, Config as DaciteConfig
from omegaconf import DictConfig, OmegaConf
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
from GrooveModel.Embedding.MultiTaskDnaEmbedding import MultiTaskDNAEmbeddingConfig
from GrooveModel.Learner.Learner import BaseDNALearner, sort_params_by_name
from GrooveModel.Learner.LearnerState import LearnerState
from GrooveModel.Loss import UncertaintyWeightedMultiTaskLoss
from GrooveModel.Metrics import MultiTaskDNAMetrics
from GrooveModel.Models import MultiTaskDNAModelConfig, MultiTaskDNAxLSTM
from GrooveModel.Tokenizer.MultiTaskDnaTokenizer import MultiTaskDnaTokenizer
from GrooveModel.TrainLoop import run_training_loop
from GrooveModel.Utils.DNAOffset import normalize_offset_tensor
from GrooveModel.Utils.DNAVelocity import normalize_velocity_tensor
from GrooveModel.Utils.Logger import setup_logger
from GrooveModel.Utils.SpecialTokens import SpecialTokens
from GrooveModel.xlstm.experiments.lr_scheduler import LinearWarmupCosineAnnealing


class MultiTaskDNALearner(BaseDNALearner):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.logger = setup_logger("MultiTaskDNALearner")

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
            ds_conf, split="train", tokenizer=MultiTaskDnaTokenizer
        )
        self.val_dataset = DNANextTokenDataset(
            ds_conf, split="validation", tokenizer=MultiTaskDnaTokenizer
        )
        # self.test_dataset = DNANextTokenDataset(
        #     ds_conf, split="test", tokenizer=MultiTaskDNATokenizer
        # )

        num_workers = ds_conf.get("num_workers", 4)
        use_persistent = ds_conf.get("persistent_workers", True) and num_workers > 0

        common_loader_args = {
            "batch_size": self.cfg.train.batch_size,
            "num_workers": num_workers,
            "persistent_workers": use_persistent,
            "pin_memory": ds_conf.get("pin_memory", True),  # True if training on GPU
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

    def _setup_criterion(self):
        self.criterion = UncertaintyWeightedMultiTaskLoss(
            tasks={
                "instrument": {"type": "ce"},
                "velocity": {"type": "mae"},
                "beat_unit": {"type": "ce"},
                "offset": {"type": "mae"},
                "grid_factor": {"type": "ce"},
                "bpm": {"type": "ce"},
                "time_signature": {"type": "ce"},
            },
            reduction="mean",
            ce_kwargs={"instrument": {"ignore_index": SpecialTokens.PAD},
                       "beat_unit": {"ignore_index": SpecialTokens.PAD},
                       "grid_factor": {"ignore_index": SpecialTokens.PAD},
                       "bpm": {"ignore_index": SpecialTokens.PAD},
                       "time_signature": {"ignore_index": SpecialTokens.PAD}},
            device=self.device,
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
        """
        CE + regression with UncertaintyWeightedMultiTaskLoss.
        """

        class_logits, reg_outputs = outputs  # Dict[str,(B,T,C)], Dict[str,(B,T,1)]
        B, T, _ = targets.shape

        # --- indices in targets ---
        output_head_indices = {
            "instrument": 0,
            "velocity": 1,
            "beat_unit": 2,
            "offset": 3,
            "grid_factor": 4,
            "bpm": 5,
            "time_signature": 6,
        }

        # --- CE targets (int tokens as-is) ---
        ce_targets = {
            "instrument": targets[:, :, output_head_indices["instrument"]].view(-1),  # Flatten Targets (B x T)
            "beat_unit": targets[:, :, output_head_indices["beat_unit"]].view(-1),
            "grid_factor": targets[:, :, output_head_indices["grid_factor"]].view(-1),
            "bpm": targets[:, :, output_head_indices["bpm"]].view(-1),
            "time_signature": targets[:, :, output_head_indices["time_signature"]].view(-1),
        }

        # --- decode regression targets from token IDs (vectorized) ---
        vel_ids = targets[:, :, output_head_indices["velocity"]]  # (B,T)
        off_ids = targets[:, :, output_head_indices["offset"]]  # (B,T)

        vel_mask = vel_ids != SpecialTokens.PAD
        off_mask = off_ids != SpecialTokens.PAD

        vel_pred = reg_outputs["velocity"].squeeze(-1)  # (B,T) in [0,1]
        off_pred = reg_outputs["offset"].squeeze(-1)  # (B,T) in [-1,1]

        # velocity token -> [0,1]
        vel_tgt = normalize_velocity_tensor(vel_ids, dtype=vel_pred.dtype)

        # offset token -> [-1,1]
        off_tgt = normalize_offset_tensor(off_ids, dtype=off_pred.dtype)

        # zero loss on PAD: copy preds into targets (no grad path through targets)
        if (~vel_mask).any():
            vel_tgt = vel_tgt.clone()
            vel_tgt[~vel_mask] = vel_pred.detach()[~vel_mask]
        if (~off_mask).any():
            off_tgt = off_tgt.clone()
            off_tgt[~off_mask] = off_pred.detach()[~off_mask]

        # --- feed multitask criterion ---
        loss_outputs = {
            "instrument": class_logits["instrument"].view(-1, class_logits["instrument"].size(-1)),
            # Flatten logits (B, T)
            "beat_unit": class_logits["beat_unit"].view(-1, class_logits["beat_unit"].size(-1)),
            "grid_factor": class_logits["grid_factor"].view(-1, class_logits["grid_factor"].size(-1)),
            "bpm": class_logits["bpm"].view(-1, class_logits["bpm"].size(-1)),
            "time_signature": class_logits["time_signature"].view(-1, class_logits["time_signature"].size(-1)),
            "velocity": vel_pred,
            "offset": off_pred,
        }
        loss_targets = {
            **ce_targets,
            "velocity": vel_tgt,
            "offset": off_tgt,
        }

        total_loss, _ = self.learner.criterion(loss_outputs, loss_targets, diagnostics=False)
        return total_loss

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
