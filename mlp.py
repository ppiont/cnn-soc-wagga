# -*- coding: utf-8 -*-

# Standard lib imports
import pathlib

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
import pdb  # Brug det


# Custom imports
from feat_eng.funcs import add_min, safe_log  # , get_corr_feats, min_max
from custom_metrics.metrics import (mean_error, lin_ccc,
                                    model_efficiency_coefficient)

# ------------------- TO DO ------------------------------------------------- #

"""
Bayesian Opt
 includes making NN layers / forward pass dynamic

K-fold to function - DONE (but sloppy)

Plot like RF summary

Streamline code

Regularize

Add avg best loss output for K-fold - DONE

train() eval() modes and regularization (dropout)


"""

# ------------------- Settings ---------------------------------------------- #


# Set matploblib style
plt.style.use('seaborn-colorblind')
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['figure.dpi'] = 450
plt.rcParams['savefig.transparent'] = True
plt.rcParams['savefig.format'] = 'svg'

# # Reset params if needed
# plt.rcParams.update(mpl.rcParamsDefault)


# ------------------- Organization ------------------------------------------ #


DATA_DIR = pathlib.Path('data/')
SEED = 43
torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------- Read and prep data ------------------------------------ #


train_data = np.load(DATA_DIR.joinpath('train.npy'))
test_data = np.load(DATA_DIR.joinpath('test.npy'))

x_train = train_data[:, 1:]
y_train = train_data[:, 0]

x_test = test_data[:, 1:]
y_test = test_data[:, 0]

# Normalize X
scaler_x = MinMaxScaler()
scaler_x.fit(x_train)
x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)

# Normalize y
scaler_y = MinMaxScaler()
scaler_y.fit(y_train.reshape(-1, 1))
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

# # Make tensors
# x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
# x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)


class Dataset(torch.utils.data.TensorDataset):
    """Characterize a PyTorch Dataset."""

    def __init__(self, features, targets):
        """Initialize with X and y."""
        self.features = features
        self.targets = targets

    def __len__(self):
        """Return total number of samples."""
        return len(self.targets)

    def __getitem__(self, index):
        """Generate one data sample."""
        return self.features[index], self.targets[index]


# train = Dataset(features=x_train, targets=y_train)
# test = Dataset(features=x_test, targets=y_test)

# Clear memory
# del(train_data, test_data, x_train, y_train, x_test, y_test)

# ------------------- NN setup ---------------------------------------------- #


class NeuralNet(nn.Module):
    """Neural Network class."""

    def __init__(self, input_dims=287, activation=nn.ReLU(),
                 dropout=nn.Dropout(p=0.5)):
        """Initialize as subclass of nn.Module, inherit its methods."""
        super(NeuralNet, self).__init__()

        self.input_dims = input_dims
        self.activation = activation
        self.dropout = dropout

        # Layer structure
        self.fc1 = nn.Linear(self.input_dims, 128)
        # self.b1 = nn.BatchNorm1d(256)
        # self.fc2 = nn.Linear(256, 128)
        self.b2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.b3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.b4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)
        # x = self.dropout(x)  # set mode?
        x = self.activation(x)
        # x = self.b1(x)
        # x = self.fc2(x)
        x = self.dropout(x)
        # x = self.activation(x)
        x = self.b2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.b3(x)
        x = self.fc4(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.b4(x)
        x = self.fc5(x)

        return x


def train_step(model, features, targets, optimizer, loss_fn):
    """Perform a single training step.

    Calulcates prediction, loss and gradients for a single batch
    and updates optimizer parameters accordingly.
    """
    # Set gradients to zero
    model.zero_grad()
    # Pass data through model
    output = model(features)
    # Calculate loss
    loss = loss_fn(output, targets)
    # Calculate gradients
    loss.backward()
    # Update parameters
    optimizer.step()

    return loss, output


def train_network(model, train_data, val_data, optimizer, loss_fn,
                  n_epochs=300, patience=20, print_progress=True):
    """Train a neural network model."""
    # Initalize loss as very high
    best_loss = 1e8

    # Create lists to hold train and val losses
    train_loss = []
    val_loss = []

    # Start training (loop over epochs)
    for epoch in range(n_epochs):

        # Initalize epoch train loss
        epoch_loss = 0
        # Loop over training batches
        model.train()
        for bidx, (features, targets) in enumerate(train_data):
            # Calculate loss and predictions
            loss, predictions = train_step(model, features, targets,
                                           optimizer, loss_fn)
            epoch_loss += loss

        # Save train epoch loss
        train_loss.append(epoch_loss.item())  # Divider med n_batches

        # Initialize val epoch loss
        val_epoch_loss = 0
        # Loop over validation batches
        model.eval()
        for bidx, (features, targets) in enumerate(val_data):
            output = model(features)
            val_epoch_loss += loss_fn(output, targets)
        # Save val epoch loss
        val_loss.append(val_epoch_loss.item())  # Divider med n_batches

        # Early stopping (check if val loss is an improvement on current best)
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss.item()
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

            # Check early stopping condition
            if epochs_no_improve == patience:
                print(f'Stopping after {epoch} epochs due to no improvement.')
                model.load_state_dict(best_model)
                break
        # Print progress at set epoch intervals if desired
        if print_progress:
            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1} Train Loss: {epoch_loss:.4}, ', end='')
                print(f'Val Loss: {val_epoch_loss:.4}')

    return train_loss, val_loss


def weight_reset(m):
    """Reset all weights in an NN."""
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


model = NeuralNet()
lr = 1e-5
regu = 1e-10  # SGD/RMSProp
# SGD/RMSProp
# L1 regu
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regu)
loss_fn = nn.MSELoss()
batch_size = 128
n_epochs = 2000
n_splits = 5
patience = 100


def k_fold_cv_train(X=x_train, y=y_train, model=model, optimizer=optimizer,
                    loss_fn=loss_fn, n_splits=5, batch_size=batch_size,
                    n_epochs=500, patience=100, shuffle=True, rng=SEED):
    """Train a NN with K-Fold cross-validation."""
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=rng)
    best_losses = []

    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):

        # Get training and val features
        x_train_fold = x_train[train_index]
        x_val_fold = x_train[val_index]

        # Get training and val targets
        y_train_fold = y_train[train_index]
        y_val_fold = y_train[val_index]

        train = Dataset(x_train_fold, y_train_fold)
        train_loader = DataLoader(train, batch_size=batch_size,
                                  shuffle=shuffle, drop_last=True)
        # Create val dataset and dataloader
        val = Dataset(x_val_fold, y_val_fold)
        val_loader = DataLoader(val, batch_size=batch_size,
                                shuffle=False, drop_last=True)

        # Train
        train_loss, val_loss = train_network(model=model,
                                             train_data=train_loader,
                                             val_data=val_loader,
                                             optimizer=optimizer,
                                             loss_fn=loss_fn,
                                             n_epochs=n_epochs,
                                             patience=patience)
        best_losses.append(val_loss)
        model.apply(weight_reset)

    return best_losses, train_loss, val_loss


# Set parameters
# model = NeuralNet()
# lr = 1e-2
# regu = 1e-6
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regu)
# loss_fn = nn.MSELoss()
# batch_size = 32
# n_epochs = 500
# n_splits = 5
# patience = 100


# Run training
kf_train = k_fold_cv_train(n_epochs=2000)


# Calc avg min val loss
avg_loss = np.mean([np.min(x) for x in kf_train[0]])


plt.figure()
plt.plot(kf_train[1], linewidth=2, label='Train Loss')
plt.plot(kf_train[2], linewidth=2, label='Val Loss')
plt.grid()
# plt.yscale('log')
plt.legend()
plt.show()
