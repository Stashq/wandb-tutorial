import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.pl_mnist_model import MNISTDataModule, My_LitModule

if not torch.cuda.is_available():
    print("CUDA not available...")

wandb_logger = WandbLogger(log_model="all")
wandb.define_metric("val_accuracy", summary="max")


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


checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy", mode="max", dirpath="artifacts/models"
)
trainer = Trainer(
    max_epochs=1,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, LogPredictionSamplesCallback()],
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
