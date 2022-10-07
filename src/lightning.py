from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch.nn import CrossEntropyLoss, Linear
from torch.nn import functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
import wandb
from datetime import datetime


if not torch.cuda.is_available():
    print("CUDA not available...")

now_ = datetime.now().isoformat()[:-7]
wandb_logger = WandbLogger(log_model="all")
wandb.define_metric("val_accuracy", summary="max")


class My_LitModule(LightningModule):
    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        """method used to define our model parameters"""
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = Linear(28 * 28, n_layer_1)
        self.layer_2 = Linear(n_layer_1, n_layer_2)
        self.layer_3 = Linear(n_layer_2, n_classes)

        self.loss = CrossEntropyLoss()
        self.lr = lr

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

        # watch gradient
        wandb.watch(
            self,
            log="gradients",
            log_freq=1000
        )

    def forward(self, x):
        """method used for inference input -> output"""

        # (b, 1, 28, 28) -> (b, 1*28*28)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        # let's do 3 x (linear + relu)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return preds

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)
        return preds

    def test_epoch_end(self, outputs):
        # shape: batch_size, channels, width, height
        dummy_input = torch.zeros((1, 1, 28, 28), device=self.device)
        file_name = f"artifacts/models/{now_}/model.onnx"
        torch.onnx.export(self, dummy_input, file_name)
        wandb.save(file_name)

    def configure_optimizers(self):
        """defines model optimizer"""
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y)
        return preds, loss, acc


class LogPredictionSamplesCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [
                f"Ground Truth: {y_i} - Prediction: {y_pred}"
                for y_i, y_pred in zip(y[:n], outputs[:n])
            ]

            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger.log_image(
                key="sample_images", images=images, caption=captions
            )

            # Option 2: log images and predictions as a W&B Table
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred]
                for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))
            ]
            wandb_logger.log_table(
                key="sample_table", columns=columns, data=data
            )


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data/mnist", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy", mode="max",
    dirpath=f"artifacts/models/{now_}")
trainer = Trainer(
    max_epochs=1,
    logger=wandb_logger,
    callbacks=[
        checkpoint_callback, LogPredictionSamplesCallback()]
)
model = My_LitModule()
dm = MNISTDataModule()

trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)  # use best trained model

# # reference can be retrieved in artifacts panel
# # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
# checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"

# # download checkpoint locally (if not already cached)
# run = wandb.init(project="MNIST")
# artifact = run.use_artifact(checkpoint_reference, type="model")
# artifact_dir = artifact.download()

# # load checkpoint
# model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
