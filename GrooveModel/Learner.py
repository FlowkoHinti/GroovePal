# loss function for masked language modelling
# insert mask tokens into sequence and let model predict sequence without mask tokens I_mask -> I
# teacher forcing can be a great optimisation as well
# add loss / training function for the various setups
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from GrooveModel.Callbacks import CheckpointCallback, EarlyStoppingCallback, PlotLossCurvesCallback, \
    PlotMetricsCallback, LRLoggerCallback, GradientClippingCallback, EpochSummaryCallback, CallbackManager
from GrooveModel.LearnerState import LearnerState
from GrooveModel.Utils.Logger import setup_logger

# LOOK AT xLSTM Experiments/main.py

# add callbacks/metrics like in fastai

# loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)


class DummyTextDataset(Dataset):
    def __init__(self, vocab_size=1000, num_samples=500, seq_len=30, num_classes=2):
        self.data = [
            (torch.randint(1, vocab_size, (seq_len,)), torch.randint(0, num_classes, (1,)).item())
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def train_loop(learner: LearnerState, callback_manager, device):
    callback_manager.call("on_train_begin", learner)

    for epoch in range(learner.start_epoch, learner.max_epochs):
        learner.epoch = epoch
        callback_manager.call("on_epoch_begin", learner)

        learner.model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(tqdm(learner.train_loader, desc=f"Epoch {epoch + 1}")):
            learner._current_batch = batch_idx
            callback_manager.call("on_batch_begin", learner)

            x, y = x.to(device), y.to(device)
            learner.optimizer.zero_grad()
            preds = learner.model(x)
            loss = learner.criterion(preds, y)
            loss.backward()
            learner.optimizer.step()

            total_loss += loss.item()
            callback_manager.call("on_batch_end", learner)

        learner.train_loss = total_loss / len(learner.train_loader)

        # Validation
        learner.model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in learner.val_loader:
                x, y = x.to(device), y.to(device)
                preds = learner.model(x)
                loss = learner.criterion(preds, y)
                total_val_loss += loss.item()
                correct += (preds.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        learner.val_loss = total_val_loss / len(learner.val_loader)
        learner.metrics['accuracy'] = correct / total

        if learner.scheduler:
            learner.scheduler.step()

        callback_manager.call("on_epoch_end", learner)

        if callback_manager.state.get("early_stop"):
            logger.warning("Early stopping triggered.")
            break

    callback_manager.call("on_train_end", learner)

BASE_PATH = Path(__file__).parent.parent
MODEL_DIR = BASE_PATH / "Models"

#setup logger
logger = setup_logger(name='TrainerLogger')

# Model and training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

vocab_size = 1000
model = LSTMClassifier(vocab_size, embed_dim=64, hidden_dim=128, num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(DummyTextDataset(), batch_size=32, shuffle=True)
val_loader = DataLoader(DummyTextDataset(), batch_size=32)

# Trainer state
learner = LearnerState(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    max_epochs=10
)

# Callbacks
callbacks = [
    CheckpointCallback(save_dir=MODEL_DIR, model_name="lstm", logger=logger),
    EarlyStoppingCallback(patience=3, logger=logger),
    PlotLossCurvesCallback(save_dir=MODEL_DIR, model_name="lstm", logger=logger),
    PlotMetricsCallback(save_dir=MODEL_DIR, model_name="lstm", logger=logger, metric_colors={"accuracy": "green"}),
    LRLoggerCallback(logger=logger),
    GradientClippingCallback(max_norm=1.0, logger=logger),
    EpochSummaryCallback(save_dir=MODEL_DIR, model_name="lstm", logger=logger)
]

manager = CallbackManager(callbacks)

# Start training
train_loop(learner, manager, device)
