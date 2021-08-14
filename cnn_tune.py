# %%

import os
import argparse
import random
import typing
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torchmetrics import R2Score
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback


# Custom imports
# from feat_eng.funcs import add_min, safe_log, get_corr_feats, min_max
from custom_metrics.metrics import mean_error, lin_ccc, model_efficiency_coefficient

import warnings

warnings.filterwarnings(action="ignore", category=DeprecationWarning)

# ------------------- TO DO ------------------------------------------------- #

"""
Tuning
"""

# ------------------- Settings ---------------------------------------------- #

# Set matploblib style
plt.style.use("seaborn-colorblind")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.format"] = "svg"

# Reset params if needed
# plt.rcParams.update(mpl.rcParamsDefault)


def seed_everything(SEED=43):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.backends.cudnn.benchmark = False


SEED = 43
seed_everything(SEED=SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

# ------------------- Organization ------------------------------------------ #

DATA_DIR = Path("data/")

# ------------------- Data -------------------------------------------------- #

#%%
# features = np.load("data/cnn_features.npy")
# targets = np.load("data/cnn_targets.npy")[:, 0]

#%%
class SoilDataset(Dataset):
    """Soil covariate dataset."""

    def __init__(self, features, targets, transform=None):
        super().__init__()

        self.features = torch.from_numpy(features).permute(0, 3, 1, 2)
        self.targets = torch.from_numpy(targets)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features[idx]
        targets = self.targets[idx]
        sample = {"features": features, "targets": targets}

        if self.transform:
            sample = self.transform(sample)

        return sample


class DataModule(pl.LightningDataModule):
    def __init__(self, feature_path, target_path, batch_size=128):
        super().__init__()
        self.feature_path = Path(os.getcwd()).joinpath(feature_path)
        self.target_path = Path(os.getcwd()).joinpath(target_path)
        self.batch_size = batch_size

    def prepare_data(self):
        # Load data
        self.features = np.load(self.feature_path)
        self.targets = np.load(self.target_path)[:, 0]

    def setup(self, stage=None):
        # Train-val-test split 7-2-1
        x_train_, x_test, y_train_, self.y_test = train_test_split(self.features, self.targets, test_size=2 / 10)
        x_train, x_val, self.y_train, y_val = train_test_split(x_train_, y_train_, test_size=2 / 8)

        feature_reshaper = FunctionTransformer(
            func=np.reshape,
            inverse_func=np.reshape,
            kw_args={"newshape": (-1, 43)},
            inv_kw_args={"newshape": (-1, 15, 15, 43)},
        )
        feature_inverse_reshaper = FunctionTransformer(func=np.reshape, kw_args={"newshape": (-1, 15, 15, 43)})

        target_reshaper = FunctionTransformer(func=np.reshape, kw_args={"newshape": (-1, 1)})

        # Preprocessing
        feature_transformer = Pipeline(
            steps=[
                ("reshaper", feature_reshaper),
                ("minmax_scaler", MinMaxScaler()),
                ("inverse_reshaper", feature_inverse_reshaper),
            ]
        )
        target_transformer = Pipeline(steps=[("reshaper", target_reshaper), ("minmax_scaler", MinMaxScaler())])

        self.train_data = SoilDataset(
            feature_transformer.fit_transform(x_train), target_transformer.fit_transform(self.y_train)
        )
        self.val_data = SoilDataset(feature_transformer.transform(x_val), target_transformer.transform(y_val))
        self.test_data = SoilDataset(feature_transformer.transform(x_test), target_transformer.transform(self.y_test))
        self.pred_data = SoilDataset(feature_transformer.transform(x_test), target_transformer.transform(self.y_test))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=4, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4, pin_memory=False)

    def predict_dataloader(self):
        return DataLoader(self.pred_data, batch_size=self.batch_size, num_workers=4, pin_memory=False)


class SoilCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config["lr"]
        self.l1_size = config["l1_size"]
        self.l2_size = config["l2_size"]
        self.l3_size = config["l3_size"]

        self.conv1 = nn.Conv2d(43, self.l1_size, 3, padding="same")
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(self.l1_size)
        # self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.l1_size, self.l2_size, 2, padding="same")
        self.bn2 = nn.BatchNorm2d(self.l2_size)
        self.pool2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.LazyLinear(self.l3_size)
        self.bn3 = nn.BatchNorm1d(self.l3_size)
        self.fc2 = nn.Linear(self.l3_size, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.fc2(x)
        return x  # prediction

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch.values()
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {"loss": loss, "y_pred": y_pred, "target": y}

    # def training_epoch_end(self, outputs):
    #     r2 = self.train_r2.compute()
    #     self.log('train_r2_epoch', r2, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch.values()
        y_pred = self.forward(x)
        val_loss = F.mse_loss(y_pred, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {"loss": val_loss, "y_pred": y_pred, "target": y}

    # def validation_epoch_end(self, outputs):
    #     r2 = self.val_r2.compute()
    #     self.log('val_r2_epoch', r2, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch.values()
        y_pred = self.forward(x)
        test_loss = F.mse_loss(y_pred, y)
        metrics = {"test_loss": test_loss}
        self.log_dict(metrics)
        return {"loss": test_loss, "y_pred": y_pred, "target": y}


#%%
# data = DataModule(DATA_DIR.joinpath("cnn_features.npy"), DATA_DIR.joinpath("cnn_targets.npy"))
# model = SoilCNN()

# data.prepare_data()
# data.setup()
# batch = next(iter(data.train_dataloader()))
# model.train()
# model.forward(batch["features"])

# #%%
# early_stopping = EarlyStopping("val_loss", patience=100, mode="min")
tune_callback = TuneReportCallback({"loss": "val_loss"}, on="validation_end")
# trainer = Trainer(
#     callbacks=[early_stopping],
#     deterministic=True,
#     # auto_lr_find=True,
#     logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
#     progress_bar_refresh_rate=0,
# )
# # trainer.tune(model)

# #%%
# trainer.fit(model, datamodule=data)


#%%
def train_tune(config, num_epochs=100, num_gpus=0):
    data = DataModule(
        "/home/peterp/GDrive/Thesis/cnn-soc-wagga/data/cnn_features.npy",
        "/home/peterp/GDrive/Thesis/cnn-soc-wagga/data/cnn_targets.npy",
        batch_size=config["batch_size"],
    )
    model = SoilCNN(config=config)
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        callbacks=[tune_callback],
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
    )

    # Necessary to call forward to initialize parameters for LazyLinear
    data.prepare_data()
    data.setup()
    batch = next(iter(data.train_dataloader()))
    model.train()
    model.forward(batch["features"])

    trainer.fit(model, datamodule=data)


def tuner(num_samples=50, num_epochs=150, gpus_per_trial=0):
    config = {
        "l1_size": tune.choice([8, 16, 32, 64, 128]),
        "l2_size": tune.choice([8, 16, 32, 64, 128]),
        "l3_size": tune.choice([8, 16, 32, 64, 128]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256]),
    }

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=15, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["l1_size", "l2_size", "l3_size", "lr"],
        metric_columns=["val_loss", "training_iteration"],
    )

    analysis = tune.run(
        tune.with_parameters(train_tune, num_epochs=num_epochs, num_gpus=gpus_per_trial),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_CNN",
    )

    print("Best hyperparameters found were: ", analysis.best_config)


tuner()


#%%
trainer.validate(model, datamodule=data)
#%%
# You are an idiot
test_backscaler = MinMaxScaler()
check_trans = test_backscaler.fit_transform(data.y_train.reshape(-1, 1))
y_pred = model(data.test_data.features).detach().numpy()
y_pred = test_backscaler.inverse_transform(y_pred.reshape(-1, 1))

# y_pred2 = trainer.predict(model=model, datamodule=data)
#%%

y_true = data.y_test.reshape(-1, 1)
# Calculate metrics
# r2 = np.exp(scaler_y.inverse_transform(rf.score(x_test, y_test)
#                                        .reshape(1, -1)))[0][0]
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
me = mean_error(y_true, y_pred)
mec = model_efficiency_coefficient(y_true, y_pred)
ccc = lin_ccc(y_true, y_pred)


# ------------------- Plotting ---------------------------------------------- #
#%%

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_true, y_pred, c=colors[0])
ax.plot(
    [y_true.min(), y_true.max()],
    [y_true.min(), y_true.max()],
    "--",
    lw=2,
    label="1:1 line",
    c=colors[1],
)
ax.set_xlabel("True")
ax.set_ylabel("Predicted")
# Regression line
y_true1, y_pred1 = y_true.reshape(-1, 1), y_pred.reshape(-1, 1)
ax.plot(
    y_true1,
    LinearRegression().fit(y_true1, y_pred1).predict(y_true1),
    c=colors[2],
    lw=2,
    label="Trend",
)
ax.legend(loc="upper left")
ax.text(
    -11,
    370,
    f"MSE: {mse:.3f}\nME:  {me:.3f}\nMEC: {mec:.3f}\nCCC: {ccc:.3f}",
    va="top",
    ha="left",
    linespacing=1.5,
    snap=True,
    bbox={"facecolor": "white", "alpha": 0, "pad": 5},
)
plt.tight_layout()
# plt.savefig('RF_x_trees.svg', bbox_inches='tight',
#             pad_inches=0)

# location for the zoomed portion
sub_ax = plt.axes([0.45, 0.45, 0.5, 0.5])
# plot the zoomed portion
sub_ax.scatter(y_true, y_pred, c=colors[0], s=10)
sub_ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--", lw=2, c=colors[1])
sub_ax.plot(
    y_true1,
    LinearRegression().fit(y_true1, y_pred1).predict(y_true1),
    c=colors[2],
    lw=2,
)
sub_ax.set_xlim([0, 60])
sub_ax.set_ylim([0, 60])

plt.show()

#%%
