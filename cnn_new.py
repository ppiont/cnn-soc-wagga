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
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_objective, plot_convergence, plot_evaluations
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_model_summary import summary

# import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy

# import pdb  # Brug det

# Custom imports
# from feat_eng.funcs import add_min, safe_log, get_corr_feats, min_max
from custom_metrics.metrics import mean_error, lin_ccc, model_efficiency_coefficient

# ------------------- TO DO ------------------------------------------------- #


# ------------------- Settings ---------------------------------------------- #


# Set matploblib style
plt.style.use("seaborn-colorblind")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.format"] = "svg"

# Reset params if needed
# plt.rcParams.update(mpl.rcParamsDefault)


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
    # torch.backends.cudnn.benchmark = False


SEED = 43
seed_everything(SEED=SEED)

#%%
# ------------------- Read and prep data ------------------------------------ #


class Dataset(torch.utils.data.TensorDataset):
    """Characterize a PyTorch Dataset."""

    def __init__(self, features, targets):
        super().__init__()

        """Initialize with X and y."""
        self.features = torch.from_numpy(features).permute(0, 3, 1, 2)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        """Return total number of samples."""
        return len(self.targets)

    def __getitem__(self, idx):
        """Generate one data sample."""

        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.features[idx], self.targets[idx]


features = np.load("data/cnn_features.npy")
targets = np.load("data/cnn_targets.npy")[:, 0]


# Train-val-test split 7-2-1
x_train_, x_test, y_train_, y_test = train_test_split(features, targets, test_size=2 / 10)
x_train, x_val, y_train, y_val = train_test_split(x_train_, y_train_, test_size=2 / 8)

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


kfold_data = Dataset(feature_transformer.fit_transform(x_train_), target_transformer.fit_transform(y_train_))
train_data = Dataset(feature_transformer.fit_transform(x_train), target_transformer.fit_transform(y_train))
val_data = Dataset(feature_transformer.transform(x_val), target_transformer.transform(y_val))
test_data = Dataset(feature_transformer.transform(x_test), target_transformer.transform(y_test))

batch_size = 128

train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, pin_memory=False)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=2, pin_memory=False)
test_loader = DataLoader(test_data, batch_size=test_data.__len__(), num_workers=2, pin_memory=False)

# %%
# ------------------ CNN setup ---------------------------------------------- #


class CNN(nn.Module):
    """Neural Network class."""

    def __init__(self, conv1_channels=32, conv2_channels=64, linear1_neurons=64, linear2_neurons=32):
        """Initialize as subclass of nn.Module, inherit its methods."""
        super().__init__()

        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.linear1_neurons = linear1_neurons
        self.linear2_neurons = linear2_neurons

        self.conv1 = nn.Conv2d(43, self.conv1_channels, kernel_size=3, stride=3)  # stride i stedet for maxpool
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(self.conv1_channels)
        self.conv2 = nn.Conv2d(self.conv1_channels, self.conv2_channels, 3)
        self.bn2 = nn.BatchNorm2d(self.conv2_channels)
        self.flat = nn.Flatten()
        self.fc1 = nn.LazyLinear(self.linear1_neurons)
        self.bn3 = nn.BatchNorm1d(self.linear1_neurons)
        self.fc2 = nn.Linear(self.linear1_neurons, self.linear2_neurons)
        self.bn4 = nn.BatchNorm1d(self.linear2_neurons)
        self.out = nn.Linear(self.linear2_neurons, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.fc2(x)
        x = self.bn4(x)
        x = self.out(x)

        return x  # prediction


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


print(summary(CNN(), torch.zeros((1, 43, 15, 15)), show_input=False))


def weight_reset(m):
    """Reset all weights in an NN."""
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


#%%
# ------------------- Cross-validation -------------------------------------- #


# def kfold_cv_train(
#     dataset,
#     model,
#     optimizer,
#     loss_fn=nn.MSELoss(),
#     n_splits=5,
#     batch_size=128,
#     n_epochs=2000,
#     patience=100,
#     shuffle=True,
#     rng=SEED,
# ):
#     """Train a NN with K-Fold cross-validation."""
#     kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=rng)
#     best_losses = []

#     for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):

#         # Print
#         print(f"FOLD {fold}")
#         print("--------------------------------")

#         # Sample elements randomly from a given list of ids, no replacement.
#         train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#         val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

#         # Define data loaders for training and testing data in this fold
#         train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
#         val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

#         # Train
#         train_loss, val_loss = train_network(
#             model=model,
#             train_data=train_loader,
#             val_data=val_loader,
#             optimizer=optimizer,
#             loss_fn=loss_fn,
#             n_epochs=n_epochs,
#             patience=patience,
#             print_progress=False,
#         )
#         best_losses.append(min(val_loss))
#         model.apply(weight_reset)

#     return sum(best_losses) / n_splits, train_loss, val_loss


# #%%

# # ------------------- Bayesian optimization --------------------------------- #


# class tqdm_skopt(object):
#     """Progress bar object for functions with callbacks."""

#     def __init__(self, **kwargs):
#         self._bar = tqdm(**kwargs)

#     def __call__(self, res):
#         """Update bar with intermediate results."""
#         self._bar.update()


# # Set parameter search space
# # sourcery skip: merge-list-append
# space = []
# space.append(Real(1e-5, 1e-1, name="learning_rate"))
# space.append(Real(1e-10, 1e-1, name="regularization"))
# space.append(Integer(16, 128, name="conv1_channels"))
# space.append(Integer(16, 128, name="conv2_channels"))
# space.append(Integer(16, 128, name="linear1_neurons"))
# space.append(Integer(16, 128, name="linear2_neurons"))

# # Set default hyperparameters
# default_params = [1e-3, 1e-5, 32, 64, 64, 32]

# batch_size = 128
# activation = nn.ReLU()

# # Work in progress
# @use_named_args(dimensions=space)
# def fitness(learning_rate, regularization, conv1_channels, conv2_channels, linear1_neurons, linear2_neurons):
#     """Perform Bayesian Hyperparameter tuning."""

#     model = CNN(
#         conv1_channels=conv1_channels,
#         conv2_channels=conv2_channels,
#         linear1_neurons=linear1_neurons,
#         linear2_neurons=linear2_neurons,
#     )
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=regularization)
#     # Create k-fold cross validation
#     avg_best_loss, *_ = kfold_cv_train(dataset=kfold_data, model=model, optimizer=optimizer, batch_size=batch_size)
#     # print(f'Avg. best validation loss: {sum(best_losses)/n_splits}')

#     return avg_best_loss


# n_calls = 100
# # Hyperparemeter search using Gaussian process minimization
# gp_result = gp_minimize(
#     func=fitness,
#     x0=default_params,
#     dimensions=space,
#     n_calls=n_calls,
#     random_state=SEED,
#     verbose=True,
#     callback=[tqdm_skopt(total=n_calls, desc="Gaussian Process")],
# )

# #%%

# plot_convergence(gp_result)
# plot_objective(gp_result)
# plot_evaluations(gp_result)
# gp_result.x


# ------------------- Training ---------------------------------------------- #
#%%

# [0.0040766469999379135, 0.06041389403147942, 125, 128, 128, 22]
def set_hyperparams(lr, regu, conv1_channels, conv2_channels, linear1_neurons, linear2_neurons):
    return lr, regu, conv1_channels, conv2_channels, linear1_neurons, linear2_neurons


# cur_best = [gp_result.x[0], gp_result.x[1], gp_result.x[2], gp_result.x[3]]
cur_best = [0.0040766469999379135, 0.06041389403147942, 125, 128, 128, 22]
lr, regu, conv1_channels, conv2_channels, linear1_neurons, linear2_neurons = set_hyperparams(*cur_best)

batch_size = 128
loss_fn = nn.MSELoss()
n_epochs = 2000
patience = 100

# x_tr, x_v, y_tr, y_v = train_test_split(x_train, y_train, test_size=1 / 8, random_state=SEED)

# train = Dataset(x_tr, y_tr)
# train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

# val = Dataset(x_v, y_v)
# val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)


#%% TRAIN ONE MODEL

model = CNN(
    conv1_channels=conv1_channels,
    conv2_channels=conv2_channels,
    linear1_neurons=linear1_neurons,
    linear2_neurons=linear2_neurons,
)
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

# plt.figure()
# plt.plot(result[0], linewidth=2, label="Train Loss")
# plt.plot(result[1], linewidth=2, label="Val Loss")
# plt.grid()
# plt.yscale("log")
# plt.legend()
# plt.show()

# %% TRAIN THREE MODELS
def train_three():
    model1 = CNN(
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        linear1_neurons=linear1_neurons,
        linear2_neurons=linear2_neurons,
    )
    optimizer1 = optim.AdamW(model1.parameters(), lr=lr, weight_decay=regu)
    _ = train_network(
        model1,
        train_data=train_loader,
        val_data=val_loader,
        optimizer=optimizer1,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        patience=patience,
        print_progress=True,
    )
    model2 = CNN(
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        linear1_neurons=linear1_neurons,
        linear2_neurons=linear2_neurons,
    )
    optimizer2 = optim.AdamW(model2.parameters(), lr=lr, weight_decay=regu)
    _ = train_network(
        model2,
        train_data=train_loader,
        val_data=val_loader,
        optimizer=optimizer2,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        patience=patience,
        print_progress=True,
    )
    model3 = CNN(
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        linear1_neurons=linear1_neurons,
        linear2_neurons=linear2_neurons,
    )
    optimizer3 = optim.AdamW(model3.parameters(), lr=lr, weight_decay=regu)
    _ = train_network(
        model3,
        train_data=train_loader,
        val_data=val_loader,
        optimizer=optimizer3,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        patience=patience,
        print_progress=True,
    )
    return model1, model2, model3


m1, m2, m3 = train_three()

m1.eval()
m2.eval()
m3.eval()


#%%

# ------------------- Testing ----------------------------------------------- #


# # Predict on test set and reshape pred and true
# y_pred = scaler_y.inverse_transform(model(torch.Tensor(x_test)).detach().numpy().reshape(-1, 1))
# y_true = y_test.reshape(-1, 1)

# # Calculate metrics
# r2 = r2_score(y_true, y_pred)
# mse = mean_squared_error(y_true, y_pred)
# me = mean_error(y_true, y_pred)
# mec = model_efficiency_coefficient(y_true, y_pred)
# ccc = lin_ccc(y_true, y_pred)


# y_true = y_test.reshape(-1, 1)
# x_test_unpacked = next(iter(test_loader))[0]
# preds = target_transformer.inverse_transform(model(x_test_unpacked).detach().numpy())

# # Calculate metrics
# r2 = r2_score(y_true, preds)
# mse = mean_squared_error(y_true, preds)
# me = mean_error(y_true, preds)
# mec = model_efficiency_coefficient(y_true, preds)
# ccc = lin_ccc(y_true, preds)
#%%
y_true = y_test.reshape(-1, 1)
# x, y = next(iter(test_loader)), i.e. test_loader can be unpacked to run 'model(x)' etc.
y_preds = np.array([]).reshape(y_true.shape[0], 0)
# use target transformer to inverse transform^^
x_test_unpacked = next(iter(test_loader))[0]

for m in [m1, m2, m3]:
    # Predict on test set and reshape pred and true
    y_pred = target_transformer.inverse_transform(m(x_test_unpacked).detach().numpy())
    y_preds = np.hstack((y_preds, y_pred))

y_pred = np.mean(y_preds, axis=1).reshape(-1, 1)
# Calculate metrics
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
me = mean_error(y_true, y_pred)
mec = model_efficiency_coefficient(y_true, y_pred)
ccc = lin_ccc(y_true, y_pred)


# ------------------- Plotting ---------------------------------------------- #

# %%
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_true, y_pred, c=colors[0])
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--", lw=2, label="1:1 line", c=colors[1])
ax.set_xlabel("True")
ax.set_ylabel("Predicted")
# Regression line
y_true1, y_pred1 = y_true.reshape(-1, 1), y_pred.reshape(-1, 1)
ax.plot(y_true1, LinearRegression().fit(y_true1, y_pred1).predict(y_true1), c=colors[2], lw=2, label="Trend")
ax.legend(loc="upper left")
ax.text(
    -11,
    470,
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
sub_ax.plot(y_true1, LinearRegression().fit(y_true1, y_pred1).predict(y_true1), c=colors[2], lw=2)
sub_ax.set_xlim([0, 60])
sub_ax.set_ylim([0, 60])

plt.show()

# %%

# RUN FOR TRAIN DATA
y_true = y_train.reshape(-1, 1)
# x, y = next(iter(test_loader)), i.e. test_loader can be unpacked to run 'model(x)' etc.
y_preds = np.array([]).reshape(y_true.shape[0], 0)
# use target transformer to inverse transform^^
train_test_loader = DataLoader(train_data, batch_size=train_data.__len__(), num_workers=2, pin_memory=False)
x_train_unpacked = next(iter(train_test_loader))[0]

for m in [m1, m2, m3]:
    # Predict on test set and reshape pred and true
    y_pred = target_transformer.inverse_transform(m(x_train_unpacked).detach().numpy())
    y_preds = np.hstack((y_preds, y_pred))

y_pred = np.mean(y_preds, axis=1).reshape(-1, 1)
# Calculate metrics
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
me = mean_error(y_true, y_pred)
mec = model_efficiency_coefficient(y_true, y_pred)
ccc = lin_ccc(y_true, y_pred)


# %%

# %%
feature_names = pd.read_csv(DATA_DIR.joinpath("feature_names_mlp_rf.csv"), index_col=0)

explainer = lime.lime_tabular.LimeTabularExplainer(
    x_train, feature_names=feature_names.feature_name.to_list(), class_names=["SOC"], verbose=True, mode="regression"
)

# Make tensors
x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

#%%
ii = [50, 100, 150, 200]
exp = [explainer.explain_instance(x_test[i], model, num_features=5) for i in ii]
# exp.show_in_notebook(show_table=True)

for i in range(len(ii)):
    with plt.style.context("ggplot"):
        exp[i].as_pyplot_figure()




# %%
# MAP
import geopandas as gpd
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

mpl.rcParams.update(
    {
        "lines.linewidth": 1,
        "font.family": "serif",
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "xtick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.major.size": 0,
        "ytick.minor.size": 0,
        "axes.titlesize": "medium",
        "figure.titlesize": "medium",
        "figure.figsize": (5, 5),
        "figure.dpi": 450,
        "figure.autolayout": True,
        "savefig.format": "pdf",
        "savefig.transparent": True,
        "image.cmap": "seismic_r",
    }
)

#%%

targets = np.load("data/cnn_targets.npy")
full_data = Dataset(feature_transformer.transform(features), target_transformer.transform(targets[:, 0]))
full_loader = DataLoader(full_data, batch_size=full_data.__len__(), num_workers=2, pin_memory=False)
X = next(iter(full_loader))[0]
y_pred = target_transformer.inverse_transform(model(X).detach().numpy())
y_true = targets[:, 0].reshape(-1, 1)
#%%

gdf = gpd.GeoDataFrame(
    y_pred, geometry=gpd.points_from_xy(targets[:, 2], targets[:, 1]), crs="EPSG:4326"
)
#%%
gdf.insert(1, "True", y_true)
gdf.insert(2, "Difference", y_pred - y_true)
gdf.rename(columns={0: "Prediction"}, inplace=True)


# %%
# get country borders
shpfilename = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_0_countries")
# read the shp
shape = gpd.read_file(shpfilename)
# extract germany geom
poly = shape.loc[shape["ADMIN"] == "Germany"]["geometry"].values[0]
# create fig, ax
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.EuroPP()))
# add geometries and features
ax.coastlines(resolution="10m", alpha=0.3)
ax.add_feature(cfeature.BORDERS, alpha=0.3)
ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor="none", edgecolor="0.5")
# convert gpd to same proj as cartopy map
crs_proj4 = ccrs.EuroPP().proj4_init
gdf_utm32 = gdf.to_crs(crs_proj4)
# Plot
gdf_utm32.plot(
    ax=ax, marker=".", markersize=10, column="Difference", legend=True, norm=colors.CenteredNorm(), cmap="seismic_r"
)
# set extent of map
ax.set_extent([5.5, 15.5, 46.5, 55.5], crs=ccrs.PlateCarree())
# fix axes pos
map_ax = fig.axes[0]
leg_ax = fig.axes[1]
map_box = map_ax.get_position()
leg_box = leg_ax.get_position()
leg_ax.set_position([leg_box.x0, map_box.y0, leg_box.width, map_box.height])
# map_ax.set_title("Sample distribution", pad = 10)
leg_ax.set_title("Error (g/kg)", pad=10)

# save and show fig
# plt.savefig(os.path.join(fig_path, "sample_distribution_soc3.pdf"), bbox_inches = 'tight', pad_inches = 0)
plt.show()
# %%


