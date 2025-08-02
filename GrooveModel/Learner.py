# loss function for masked language modelling
# insert mask tokens into sequence and let model predict sequence without mask tokens I_mask -> I
# teacher forcing can be a great optimisation as well
# add loss / training function for the various setups

import torch
import torch.nn as nn
import torch.optim as optim
from dacite import from_dict, Config as DaciteConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


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
            ds_conf, split="val", tokenizer=Tokenizers.MultiTaskDNATokenizer
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=ds_conf.get("shuffle", True),
            num_workers=ds_conf.get("num_workers", 4),
            collate_fn=CollateFunctions.pad_truncate_batch
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=ds_conf.get("num_workers", 4),
            collate_fn=CollateFunctions.pad_truncate_batch
        )

    def _setup_embedding(self):
        schema = OmegaConf.structured(MultiTaskDNAEmbeddingConfig())
        self.embedding_config = OmegaConf.merge(schema, self.cfg.embedding)

    def _setup_model(self):
        cfg_omega = OmegaConf.merge(OmegaConf.structured(MultiTaskDNAModelConfig()), self.cfg.model)
        model_config = from_dict(
            MultiTaskDNAModelConfig,
            OmegaConf.to_container(cfg_omega, resolve=True),
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
        )

    def _setup_callbacks(self):
        self.callbacks = [
            CheckpointCallback(
                save_dir=self.cfg.train.save_dir,
                model_name=self.cfg.train.model_name,
                logger=self.logger,
                monitor=self.cfg.train.get("monitor", "val_loss"),
                mode=self.cfg.train.get("mode", "min")
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

    #TODO: FIX
    def compute_loss(self, preds, targets):
        return self.learner.criterion(preds, targets)

    # TODO: FIX
    def compute_metrics(self, preds, targets):
        correct = (preds.argmax(dim=1) == targets).sum().item()
        total = targets.size(0)
        return {"accuracy": correct / total}

    def train(self):
        run_training_loop(
            learner=self.learner,
            callback_manager=self.callback_manager,
            device=self.device,
            compute_loss_fn=self.compute_loss,
            compute_metrics_fn=self.compute_metrics,
            logger=self.logger
        )
