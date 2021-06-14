# -*- coding: utf-8 -*-

#%%

# Standard lib imports
import os
import pathlib
import random

# Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_objective, plot_convergence, plot_evaluations
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
Use Torch Dataset.. you made a class for it dummy
[0.07358756448295099, 0.1, 0.14635936519340323, 1, 213]
^best params from "final" colab run
"""

# ------------------- Settings ---------------------------------------------- #


# Set matploblib style
plt.style.use('seaborn-colorblind')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['figure.dpi'] = 450
plt.rcParams['savefig.transparent'] = True
plt.rcParams['savefig.format'] = 'svg'

# Reset params if needed
# plt.rcParams.update(mpl.rcParamsDefault)


# ------------------- Organization ------------------------------------------ #


DATA_DIR = pathlib.Path('data/')


def seed_everything(SEED=43):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED']=str(SEED)
    # torch.backends.cudnn.benchmark = False


SEED = 43
seed_everything(SEED=SEED)

#%%
# ------------------- Read and prep data ------------------------------------ #


x_train = np.load(DATA_DIR.joinpath('x_train_cnn.npy'))
y_train = np.load(DATA_DIR.joinpath('y_train_cnn.npy'))

x_test = np.load(DATA_DIR.joinpath('x_test_cnn.npy'))
y_test = np.load(DATA_DIR.joinpath('y_test_cnn.npy'))

y_train, y_test = y_train[:, 0], y_test[:, 0]


#%%

input_dims=x_train.shape[-1]

n_channels = input_dims
data_transformed = np.zeros_like(x_train)

channel_scalers = []

for i in range(n_channels):
    mmx = MinMaxScaler()
    slc = x_train[:, :, :, i].reshape(x_train.shape[0], -1) # make it a bunch of row vectors
    transformed = mmx.fit_transform(slc)
    transformed = transformed.reshape(x_train.shape[0], 15, 15) # reshape it back to tiles
    data_transformed[:, :, :, i] = transformed # put it in the transformed array
    channel_scalers.append(mmx) # store the transform

#%%


# Normalize X
scaler_x = MinMaxScaler()
scaler_x.fit(x_train)
x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)

# Normalize y
scaler_y = MinMaxScaler()
scaler_y.fit(y_train.reshape(-1, 1))
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
# There is no reason to scale y_test actually
# y_test = scaler_y.transform(y_test.reshape(-1, 1))

# # Make tensors
# x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
# x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

#%%

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

    def __init__(self, input_dims=input_dims, n_layers=3, n_neurons=64,
                 activation=nn.ELU(), dropout_rate=0.5):
        """Initialize as subclass of nn.Module, inherit its methods."""
        super(NeuralNet, self).__init__()

        self.input_dims = input_dims
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # Layer structure
        # First layer
        self.in_layer = nn.Linear(self.input_dims, self.n_neurons)

        # Dense, Dropout, Activation and BN
        self.dense = nn.Linear(self.n_neurons, self.n_neurons)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = activation
        self.batchnorm = nn.BatchNorm1d(self.n_neurons)

        # Output layer
        self.out_layer = nn.Linear(self.n_neurons, 1)

    def forward(self, x):
        """Forward pass."""
                
        x = self.batchnorm(self.activation(self.dropout(self.in_layer(x))))

        for i in range(self.n_layers-1):
            x = self.batchnorm(self.activation(self.dropout(self.dense(x))))

        x = self.out_layer(x)

        return x


def train_step(model, features, targets, optimizer, loss_fn):
    """Perform a single training step.

    Calulcates prediction, loss and gradients for a single batch
    and updates optimizer parameters accordingly."""

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
                  n_epochs=2000, patience=100, print_progress=True):
    """Train a neural network model."""
    # Initalize loss as very high
    best_loss = 1e8

    # Create lists to hold train and val losses
    train_loss = []
    val_loss = []
    # Init epochs_no_improve
    epochs_no_improve = 0
    # best_model = copy.deepcopy(model.state_dict())
    
    # Start training (loop over epochs)
    for epoch in range(n_epochs):

        # Initalize epoch train loss
        train_epoch_loss = 0
        # Loop over training batches
        model.train()  # set model to training mode for training
        for bidx, (features, targets) in enumerate(train_data):
            # Calculate loss and predictions
            loss, predictions = train_step(model, features, targets,
                                           optimizer, loss_fn)
            train_epoch_loss += loss
        # Save train epoch loss
        train_loss.append(train_epoch_loss.item())

        # Initialize val epoch loss
        val_epoch_loss = 0
        # Loop over validation batches
        model.eval()  # set model to evaluation mode for validation
        for bidx, (features, targets) in enumerate(val_data):
            output = model(features)
            val_epoch_loss += loss_fn(output, targets)
        # Save val epoch loss
        val_loss.append(val_epoch_loss.item())

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
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1} Train Loss: {train_epoch_loss:.4}, ', end='')
                print(f'Val Loss: {val_epoch_loss:.4}')

    return train_loss, val_loss


def weight_reset(m):
    """Reset all weights in an NN."""
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

#%%
# ------------------- Cross-validation -------------------------------------- #

def kfold_cv_train(x_train, y_train, model, optimizer,loss_fn=nn.MSELoss(), 
                   n_splits=5, batch_size=312, n_epochs=2000, patience=100,
                   shuffle=True, rng=SEED):
    """Train a NN with K-Fold cross-validation."""
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=rng)
    best_losses = []

    for fold, (train_index, val_index) in enumerate(kfold.split(x_train, y_train)):
        # print(f'Starting fold {fold + 1}')
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
                                shuffle=False, drop_last=False)

        # Train
        train_loss, val_loss = train_network(model=model,
                                             train_data=train_loader,
                                             val_data=val_loader,
                                             optimizer=optimizer,
                                             loss_fn=loss_fn,
                                             n_epochs=n_epochs,
                                             patience=patience,
                                             print_progress=True)
        best_losses.append(min(val_loss))
        model.apply(weight_reset)

    return sum(best_losses)/n_splits, train_loss, val_loss


#%%

# ------------------- Bayesian optimization --------------------------------- #


class tqdm_skopt(object):
    """Progress bar object for functions with callbacks."""

    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        """Update bar with intermediate results."""
        self._bar.update()


# Set parameter search space
space = []
# space.append(Categorical(['relu', 'leakyrelu', 'elu'], name='activation'))
space.append(Real(1e-5, 1e-1, name='learning_rate'))
space.append(Real(1e-10, 1e-1, name='regularization'))
space.append(Real(0.0, 0.9, name='dropout_rate'))
space.append(Integer(int(32), int(312), name='batch_size', dtype=int))
space.append(Categorical(['relu', 'leakyrelu', 'prelu', 'elu', 'selu'], name='activation'))
space.append(Integer(int(1), int(5), name='n_layers', dtype=int))
space.append(Integer(int(16), int(512), name='n_neurons', dtype=int))

# Set default hyperparameters
default_params = [1e-3,
                  1e-5,
                  0.5,
                  312,
                  'elu',
                  3,
                  256]


# Work in progress
@use_named_args(dimensions=space)
def fitness(learning_rate, regularization, dropout_rate, batch_size, activation,
            n_layers, n_neurons):
    """Perform Bayesian Hyperparameter tuning."""

    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'leakyrelu':
        activation = nn.LeakyReLU()
    elif activation == 'elu':
        activation = nn.ELU()
    elif activation == 'selu':
        activation = nn.SELU()
    elif activation == 'prelu':
        activation = nn.PReLU()

    # print(f'Learning Rate: {learning_rate:.0e}, Regularization: {regularization:.0e}, ', end='')
    # print(f'Dropout: {dropout:.2f}')  #, Batch Size: {batch_size}')

    model = NeuralNet(activation=activation, dropout_rate=dropout_rate,
                      n_layers=n_layers, n_neurons=n_neurons)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                                weight_decay=regularization)
    # Create k-fold cross validation
    avg_best_loss, *_ = kfold_cv_train(x_train=x_train, y_train=y_train,
                                     model=model, optimizer=optimizer,
                                     batch_size=batch_size)
    # print(f'Avg. best validation loss: {sum(best_losses)/n_splits}')

    return avg_best_loss

n_calls = 200
# Hyperparemeter search using Gaussian process minimization
gp_result = gp_minimize(func=fitness,
                        x0=default_params,
                        dimensions=space,
                        n_calls=n_calls,
                        random_state=SEED,
                        verbose=True, callback=[tqdm_skopt(total=n_calls,
                                          desc='Gaussian Process')])

plot_convergence(gp_result)
plot_objective(gp_result)
plot_evaluations(gp_result)
gp_result.x


# --------------------------------------------------------------------------- #
#%%

# [0.07482420407147314, 0.025136732303555738, 0.22345284956151534, 1, 256]
lr = 0.07482420407147314
regu = 0.025136732303555738
dropout_rate = 0.22345284956151534
batch_size = 312
activation = nn.ELU()
n_layers = 1
n_neurons = 256

n_epochs = 2000
patience = 100

X_train, X_val, y_train, y_val = train_test_split(x_train, y_train,
                                            test_size=0.1, random_state=SEED)

train = Dataset(X_train, y_train)
train_loader = DataLoader(train, batch_size=batch_size,
                             shuffle=True, drop_last=False, num_workers=2)

val = Dataset(X_val, y_val)
val_loader = DataLoader(val, batch_size=batch_size,
                        shuffle=False, drop_last=False, num_workers=2)

model = NeuralNet(activation=activation, dropout_rate=dropout_rate)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=regu)
loss_fn = nn.MSELoss()

# Run training
result = train_network(model, train_data=train_loader, val_data=val_loader, 
                       optimizer=optimizer, loss_fn=loss_fn,
                       n_epochs=n_epochs, patience=patience, print_progress=True)

model.eval()
pred = model(torch.Tensor(x_test)).detach().numpy()

plt.figure()
plt.plot(result[0], linewidth=2, label='Train Loss')
plt.plot(result[1], linewidth=2, label='Val Loss')
plt.grid()
plt.yscale('log')
plt.legend()
plt.show()

# %%

# ------------------- Testing ----------------------------------------------- #


from sklearn.metrics import mean_squared_error, r2_score
# Predict on test set and reshape pred and true
y_pred = np.exp(scaler_y.inverse_transform(model(torch.Tensor(x_test)).detach().numpy().reshape(-1, 1)))
y_true = np.exp(y_test.reshape(-1, 1))

# Calculate metrics
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
me = mean_error(y_true, y_pred)
mec = model_efficiency_coefficient(y_true, y_pred)
ccc = lin_ccc(y_true, y_pred)



