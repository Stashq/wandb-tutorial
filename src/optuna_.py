import optuna
import torch
from optuna import Trial
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
import wandb
from src.pl_mnist_model import MNISTDataModule, My_LitModule


# Simple example
def objective(trial: Trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study = optuna.create_study()
study.optimize(objective, n_trials=10)
best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))


# Optuna + Lightning
wandb_logger = WandbLogger(log_model="all")
wandb.define_metric("val_accuracy", summary="max")


def objective(trial: Trial) -> float:
    n_layer_1 = trial.suggest_int("n_layer_1", 128, 512)
    n_layer_2 = trial.suggest_int("n_layer_2", 128, 512)
    lr = trial.suggest_float("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 48, 64])

    model = My_LitModule(n_layer_1=n_layer_1, n_layer_2=n_layer_2, lr=lr)
    datamodule = MNISTDataModule(batch_size=batch_size)

    trainer = Trainer(
        logger=wandb_logger,
        enable_checkpointing=False,
        max_epochs=5,
        gpus=1 if torch.cuda.is_available() else None,
    )
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_accuracy"].item()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
