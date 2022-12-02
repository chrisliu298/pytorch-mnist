import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchinfo import summary


class BaseModel(LightningModule):
    """A base class for all models."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def on_train_start(self):
        """Log model summary at the start of training."""
        summary(self, input_size=(1, 28, 28), verbose=self.cfg.verbose)

    def evaluate(self, batch, stage=None):
        """Evaluate model on a batch of data."""
        x, y = batch
        logits = self(x)
        pred = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, y)
        acc = torch.sum(pred == y).float() / len(y)
        self.log(f"{stage}_loss", loss, logger=True)
        self.log(f"{stage}_acc", acc, logger=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        """Train model on a batch of data."""
        loss, acc = self.evaluate(batch, stage="train")
        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, outputs):
        """Log at the end of each training epoch."""
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["train_acc"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss, logger=True)
        self.log("avg_train_acc", avg_acc, logger=True)

    def validation_step(self, batch, batch_idx):
        """Validate model on a batch of data."""
        loss, acc = self.evaluate(batch, stage="val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        """Log at the end of each validation epoch."""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss, logger=True)
        self.log("avg_val_acc", avg_acc, logger=True)

    def test_step(self, batch, batch_idx):
        """Test model on a batch of data."""
        loss, acc = self.evaluate(batch, stage="test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        """Log at the end of each test epoch."""
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_loss", avg_loss, logger=True)
        self.log("avg_test_acc", avg_acc, logger=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        if self.cfg.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.wd,
            )
        elif self.cfg.optimizer == "adam":
            optimizer = optim.AdamW(
                self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd
            )
        return optimizer


class FCN(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.layer_dims = [int(x) for x in cfg.layer_dims.split("x")]
        self.num_classes = self.layer_dims[-1]
        self.build_layers(self.layer_dims)

    def build_layers(self, layer_dims):
        """Build layers."""
        self._layers = []
        for i in range(len(layer_dims) - 1):
            layer = nn.Linear(layer_dims[i], layer_dims[i + 1])
            if self.cfg.init == "normal":
                nn.init.normal_(layer.weight, mean=0, std=(1 / layer_dims[i]) ** 0.5)
            nn.init.zeros_(layer.bias)
            self._layers.append(layer)
            self.add_module(f"layer_{i}", layer)
            if self.cfg.dropout > 0:
                dropout = nn.Dropout(self.cfg.dropout)
                self._layers.append(dropout)
                self.add_module(f"dropout_{i}", dropout)

    def forward(self, x):
        """Forward pass."""
        x = x.view(x.size(0), -1)
        for layer in self._layers:
            x = layer(x)
            if isinstance(layer, nn.Dropout):
                x = F.relu(x)
        return x
