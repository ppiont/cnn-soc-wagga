#%%
import os
import argparse
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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# torch.set_default_tensor_type(torch.FloatTensor)
seed_everything(43)

# TO DO #
# Clean up code
# Comment code
# Mixed precision
# RayTune
#%%
feats = np.load('data/cnn_features.npy')
targets = np.load('data/cnn_targets.npy')[:, 0]


#%%
trans = transforms.Compose()


class SoilDataset(Dataset):
    """Soil covariate dataset."""

    def __init__(self, features, targets, transform=None):
        super(SoilDataset, self).__init__()

        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(np.moveaxis(targets, -1, 1)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features
        targets = self.targets
        # sample = {'features': features, 'targets': targets}

        # if self.transform:
        #     sample = self.transform(sample)

        return features, targets


class DataModule(pl.LightningDataModule):
    def __init__(self, feature_dir, target_dir):
        super(DataModule, self).__init__()
        self.feature_dir = Path(os.getcwd()).joinpath(feature_dir)
        self.target_dir = = Path(os.getcwd()).joinpath(target_dir)
        self.batch_size = 64

    def prepare_data(self):
        # Load data
        self.targets = np.load(self.target_dir)
        self.features = np.load(self.feature_dir)

    def setup(self, stage=None):

        # Train-val-test split: 70-20-10
        train_split, test_split = train_test_split(self.features, self.targets, test_size=1/10)
        train_split, val_split = train_test_split(train_split, test_size=2/9)

        # Preprocessing
        target_col = self.input_data.columns[self.target_idx]
        spectral_col = self.input_data.columns[self.spectral_idx]
        categorical_col= self.input_data.columns[self.categorical_idx]
        spectral_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                            ('absorbance', AbsorbanceFromReflectance()),
                                            ('robust_scaler', RobustScaler()),
                                            ('minmax_scaler', MinMaxScaler())])
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
        preprocessor = ColumnTransformer(
            transformers=[('targets', 'passthrough', target_col),
                          ('spectral', spectral_transformer, spectral_col),
                          ('categorical', categorical_transformer, categorical_col)])

        self.train_data = SoilDataset(preprocessor.fit_transform(train_split))
        self.val_data = SoilDataset(preprocessor.transform(val_split))
        self.test_data = SoilDataset(preprocessor.transform(test_split))

    def train_dataloader(self):
  
        # Generating train_dataloader
        return DataLoader(self.train_data, batch_size = self.batch_size, num_workers=4, pin_memory=False)
  
    def val_dataloader(self):

        # Generating val_dataloader
        return DataLoader(self.val_data, batch_size = self.batch_size, num_workers=4, pin_memory=False)
  
    def test_dataloader(self):

        # Generating test_dataloader
        return DataLoader(self.test_data, batch_size = self.batch_size, num_workers=4, pin_memory=False) 


class SoilCNN(pl.LightningModule):
    
    def __init__(self):
        super(SoilRegressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(68, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1))

        # self.train_r2 = R2Score()
        # self.val_r2 = R2Score()

    def forward(self, x):
        prediction = self.layers(x)
        return prediction.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'loss' : loss, 'y_pred' : y_pred, 'target' : y}

    # def training_epoch_end(self, outputs):
    #     r2 = self.train_r2.compute()
    #     self.log('train_r2_epoch', r2, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        val_loss = F.mse_loss(y_pred, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss' : val_loss, 'y_pred' : y_pred, 'target' : y}

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
        metrics = {'test_loss': metrics['loss']}
        self.log_dict(metrics)


#%%
data_in = DataModule()
model = SoilRegressor()

early_stopping = EarlyStopping('val_loss', patience=1, mode='min')
trainer = Trainer(callbacks=[early_stopping])
trainer.fit(model, datamodule=data_in)

#%%
trainer.test(model, datamodule=data_in)
# %%
