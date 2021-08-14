# -*- coding: utf-8 -*-

#%%

# Standard lib imports
import os
import pathlib
import random

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_model_summary import summary
from torch.utils.data import DataLoader
import copy


# ------------------- Settings ---------------------------------------------- #

# Set matploblib style
plt.style.use("seaborn-colorblind")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.format"] = "svg"

# ------------------- Organization ------------------------------------------ #

DATA_DIR = pathlib.Path("data/")


def seed_everything(SEED=43):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(SEED)


SEED = 43
seed_everything(SEED=SEED)

#%%
# ------------------- Read and prep data ------------------------------------ #

train_data = np.load(DATA_DIR.joinpath("train_45.npy"))
test_data = np.load(DATA_DIR.joinpath("test_45.npy"))

x_train = train_data[:, 3:]
y_train = train_data[:, 0]

x_test = test_data[:, 3:]
y_test = test_data[:, 0]

input_dims = x_train.shape[-1]

# Normalize X
scaler_x = MinMaxScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)

# Normalize y
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))

# # Make tensors
# x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
# x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

#%%

# ------------------- NN setup ---------------------------------------------- #


class NeuralNet(nn.Module):
    """Neural Network class."""

    def __init__(self, input_dims=input_dims, n_layers=1, n_neurons=209, activation=nn.ReLU()):
        """Initialize as subclass of nn.Module, inherit its methods."""
        super(NeuralNet, self).__init__()

        self.input_dims = input_dims
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # Layer structure
        # First layer
        self.in_layer = nn.Linear(self.input_dims, self.n_neurons)

        # Dense, Activation and BN
        self.dense = nn.Linear(self.n_neurons, self.n_neurons)
        self.activation = activation
        self.batchnorm = nn.BatchNorm1d(self.n_neurons)

        # Output layer
        self.out_layer = nn.Linear(self.n_neurons, 1)

    def forward(self, x):
        """Forward pass."""

        x = self.batchnorm(self.activation(self.in_layer(x)))

        for _ in range(self.n_layers - 1):
            x = self.batchnorm(self.activation(self.dense(x)))

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


def train_network(model, train_data, val_data, optimizer, loss_fn, n_epochs=2000, patience=100, print_progress=True):
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
            loss, predictions = train_step(model, features, targets, optimizer, loss_fn)
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
                print(f"Stopping after {epoch} epochs due to no improvement.")
                model.load_state_dict(best_model)
                break
        # Print progress at set epoch intervals if desired
        if print_progress and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} Train Loss: {train_epoch_loss:.4}, ", end="")
            print(f"Val Loss: {val_epoch_loss:.4}")

    return train_loss, val_loss


print(summary(NeuralNet(), torch.zeros((1, 38)), show_input=False))


def weight_reset(m):
    """Reset all weights in an NN."""
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


# ------------------- Training ---------------------------------------------- #
#%%

# [0.03569811448339617, 0.03980246918558593, 1, 209]
def set_hyperparams(lr, regu, n_layers, n_neurons):
    return lr, regu, n_layers, n_neurons


# cur_best = [gp_result.x[0], gp_result.x[1], gp_result.x[2], gp_result.x[3]]
cur_best = [0.03569811448339617, 0.03980246918558593, 1, 209]
lr, regu, n_layers, n_neurons = set_hyperparams(*cur_best)

batch_size = 312
activation = nn.ReLU()
loss_fn = nn.MSELoss()
n_epochs = 2000
patience = 100

x_tr, x_v, y_tr, y_v = train_test_split(x_train, y_train, test_size=1 / 8, random_state=SEED)

train = Dataset(x_tr, y_tr)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

val = Dataset(x_v, y_v)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)


# %% TRAIN ONE MODEL

model = NeuralNet(activation=activation)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=regu)

# Run training
result = train_network(
    model,
    train_data=train_loader,
    val_data=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    n_epochs=n_epochs,
    patience=patience,
    print_progress=True,
)

model.eval()


# %%
feature_names = pd.read_csv(DATA_DIR.joinpath("feature_names_mlp_rf.csv"), index_col=0)

explainer = lime.lime_tabular.LimeTabularExplainer(
    x_train, feature_names=feature_names.feature_name.to_list(), class_names=["SOC"], verbose=True, mode="regression"
)

#%%


def wrapped_net(x):
    return model(torch.from_numpy(x).float()).detach().numpy()


ii = [50, 100, 150, 200]
exp = [explainer.explain_instance(x_test[i], wrapped_net, num_features=5) for i in ii]
# exp.show_in_notebook(show_table=True)

for i in range(len(ii)):
    with plt.style.context("ggplot"):
        exp[i].as_pyplot_figure()
# %%
