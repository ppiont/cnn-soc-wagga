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
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action="ignore", category=DeprecationWarning)

# ------------------- TO DO ------------------------------------------------- #

"""
Make CNN
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
    def __init__(self, feature_path, target_path, batch_size=32):
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
        x_train_, x_test, y_train_, y_test = train_test_split(self.features, self.targets, test_size=1 / 10)
        x_train, x_val, y_train, y_val = train_test_split(x_train_, y_train_, test_size=2 / 9)

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
            feature_transformer.fit_transform(x_train), target_transformer.fit_transform(y_train)
        )
        self.val_data = SoilDataset(feature_transformer.transform(x_val), target_transformer.fit_transform(y_val))
        self.test_data = SoilDataset(feature_transformer.transform(x_test), target_transformer.fit_transform(y_test))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=4, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4, pin_memory=False)


class SoilCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(43, 16, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.Linear(64, 1),
        )

        # self.train_r2 = R2Score()
        # self.val_r2 = R2Score()

    def forward(self, x):
        return self.layers(x).squeeze()  # prediction

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
        # x, y = batch
        # y_pred = self.forward(x)
        # test_loss = F.mse_loss(y_pred, y)
        # print(test_loss)
        # return {'loss' : test_loss, 'y_pred' : y_pred, 'target' : y}
        metrics = self.validation_step(batch, batch_idx)
        metrics = {"test_loss": metrics["loss"]}
        self.log_dict(metrics)


#%%
data = DataModule(DATA_DIR.joinpath("cnn_features.npy"), DATA_DIR.joinpath("cnn_targets.npy"))
model = SoilCNN()

data.prepare_data()
data.setup()
batch = next(iter(data.train_dataloader()))
model.train()
model.forward(batch["features"])

early_stopping = EarlyStopping("val_loss", patience=50, mode="min")
trainer = Trainer(callbacks=[early_stopping])
trainer.fit(model, datamodule=data)

#%%
trainer.test(model, datamodule=data)
# %%
