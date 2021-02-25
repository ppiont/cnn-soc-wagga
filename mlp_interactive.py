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
from skopt import gp_minimize, dump, load
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
Should I shuffle train_loader in CV?
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


train_data = np.load(DATA_DIR.joinpath('train_no_outlier.npy'))
test_data = np.load(DATA_DIR.joinpath('test_no_outlier.npy'))

x_train = train_data[:, 1:]
y_train = train_data[:, 0]

input_shape=x_train.shape[-1]

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
# There is no reason to scale y_test actually
y_test = scaler_y.transform(y_test.reshape(-1, 1))

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

    def __init__(self, input_dims=input_shape, activation=nn.ELU(), dropout=0.5):
        """Initialize as subclass of nn.Module, inherit its methods."""
        super(NeuralNet, self).__init__()

        self.input_dims = input_dims
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        # Layer structure
        self.fc1 = nn.Linear(self.input_dims, 256)
        self.b1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.b2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.b3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.b4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)
        x = self.dropout(x)  # set mode?
        x = self.activation(x)
        x = self.b1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.activation(x)
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
                  n_epochs=300, patience=20, print_progress=True):
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
            if (epoch + 1) % 100 == 0:
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

model = NeuralNet(activation=nn.ELU(), dropout=0.5)
lr = 1e-3
regu = 1e-2
# SGD/RMSProp
# L1 regu
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=regu)
loss_fn = nn.MSELoss()
batch_size = 312
n_epochs = 2000
n_splits = 5
patience = 100


def kfold_cv_train(x_train=x_train, y_train=y_train, model=model, optimizer=optimizer,
                    loss_fn=nn.MSELoss(), n_splits=5, batch_size=128,
                    n_epochs=500, patience=100, shuffle=True, rng=SEED):
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
                                             print_progress=False)
        best_losses.append(min(val_loss))
        model.apply(weight_reset)

    return best_losses, train_loss, val_loss


#%%


######## above used to be before kfold definition

# Set parameters
# model = NeuralNet()
# lr = 1e-3
# regu = 1e-2
# optimizer = optim.Adam"(model.parameters(), lr=lr, weight_decay=regu)
# loss_fn = nn.MSELoss()
# batch_size = 128
# n_epochs = 500
# n_splits = 5
# patience = 100


# Run training
# den træner kun når jeg kører den uden at specificere optimizer her, selvom
# det er den samme optimizer som jeg definerer tidligere...
# kf_train = kfold_cv_train(n_epochs=2000)


# # Calc avg min val loss
# avg_loss = np.mean([np.min(x) for x in kf_train[0]])


# plt.figure()
# plt.plot(kf_train[1], linewidth=2, label='Train Loss')
# plt.plot(kf_train[2], linewidth=2, label='Val Loss')
# plt.grid()
# # plt.yscale('log')
# plt.legend()
# plt.show()


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
space.append(Real(1e-5, 1e-1, prior='log-uniform', name='learning_rate'))
space.append(Real(1e-10, 1e-1, prior='log-uniform', name='regularization'))
space.append(Real(0.0, 0.9, name='dropout'))
space.append(Integer(int(32), int(312), name='batch_size', dtype=int))

# Set default hyperparameters
default_params = [1e-3,
                  1e-5,
                  0.5,
                  312]

# Best params from first optimization
# best_params = ['elu',
#                 0.018548744510205464,
#                 1e-10,
#                 0.01698162022248246]

# best_params1 = [0.0012304697066411964,
#                 1.0421441915334575e-09,
#                 0.02539150096227823]

# best_params2 = [0.005315802234314809,
#                 0.1,
#                 0.0,
#                 312]

# best_params_colab = [0.0005877430097476587
#                      0.1,
#                      0.10470590080305037,
#                      312]

# Work in progress
@use_named_args(dimensions=space)
def fitness(learning_rate, regularization, dropout, batch_size):
    """Perform Bayesian Hyperparameter tuning."""

    # num_epochs = 2000
    # n_splits = 5
    # patience = 100
    # if activation == 'relu':
    #     activation = nn.ReLU()
    # elif activation == 'leakyrelu':
    #     activation = nn.LeakyReLU()
    # elif activation == 'elu':
    #     activation = nn.ELU()
    # print(f'Learning Rate: {learning_rate:.0e}, Regularization: {regularization:.0e}, ', end='')
    # print(f'Dropout: {dropout:.2f}')  #, Batch Size: {batch_size}')

    model = NeuralNet(activation=nn.ELU(), dropout=dropout)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=regularization)
    # Create k-fold cross validation
    best_losses, *_ = kfold_cv_train(n_epochs=2000, model=model,
                                     optimizer=optimizer, batch_size=batch_size)
    # print(f'Avg. best validation loss: {sum(best_losses)/n_splits}')

    return sum(best_losses)/n_splits

n_calls = 200
# Hyperparemeter search using Gaussian process minimization
gp_result = gp_minimize(func=fitness,
                        x0=default_params,
                        dimensions=space,
                        n_calls=n_calls,
                        verbose=True, callback=[tqdm_skopt(total=n_calls,
                                          desc='Gaussian Process')])

plot_convergence(gp_result)
plot_objective(gp_result)
plot_evaluations(gp_result)
gp_result.x


# --------------------------------------------------------------------------- #


#%%

batch_size = 312


######## WORK IN PROG #######
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train,
                                            test_size=0.2, random_state=SEED)

train = Dataset(X_train, y_train)
train_loader = DataLoader(train, batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=2)

val = Dataset(X_val, y_val)
val_loader = DataLoader(val, batch_size=batch_size,
                        shuffle=False, drop_last=False, num_workers=2)

# best_params2 = [0.005315802234314809,
#                 0.1,
#                 0.0,
#                 312]

model = NeuralNet(activation=nn.ELU(), dropout=0.0)
optimizer = optim.AdamW(model.parameters(), lr=0.005315802234314809, weight_decay=0.1)
loss_fn = nn.MSELoss()

n_epochs = 2000
n_splits = 5
patience = 100

# Run training
result = train_network(model, train_data=train_loader, val_data=val_loader, 
                       optimizer=optimizer, loss_fn=loss_fn,
                       n_epochs=n_epochs, patience=patience, print_progress=True)

# Best model
# model.load_state_dict(result[1])
model.eval()
pred = model(torch.Tensor(x_test)).detach().numpy()

# # Calc avg min val loss
# avg_loss = np.mean([np.min(x) for x in kf_train[0]])


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
y_pred = scaler_y.inverse_transform(model(torch.Tensor(x_test)).detach().numpy().reshape(-1, 1))
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
me = mean_error(y_true, y_pred)
mec = model_efficiency_coefficient(y_true, y_pred)
ccc = lin_ccc(y_true, y_pred)


# ------------------- Plotting ---------------------------------------------- #


fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_true, y_pred, c=colors[0])
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        '--', lw=2, label='1:1 line', c=colors[1])
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
# Regression line
y_true1, y_pred1 = y_true.reshape(-1, 1), y_pred.reshape(-1, 1)
ax.plot(y_true1, LinearRegression().fit(y_true1, y_pred1).predict(y_true1),
        c=colors[2], lw=2, label='Trend')
ax.legend(loc='upper left')
ax.text(-11, 370,
        f'MSE: {mse:.3f}\nME:  {me:.3f}\nMEC: {mec:.3f}\nCCC: {ccc:.3f}',
        va='top', ha='left', linespacing=1.5, snap=True,
        bbox={'facecolor': 'white', 'alpha': 0, 'pad': 5})
plt.tight_layout()
# plt.savefig('NN_x_x.svg', bbox_inches='tight',
#             pad_inches=0)
plt.show()
# %%
