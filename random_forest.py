# -*- coding: utf-8 -*-

#%%

# Standard library imports
import pathlib

# Imports
import numpy as np

# import numpy.ma as ma
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from skopt.utils import use_named_args
from sklearn.preprocessing import MinMaxScaler

# from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from skopt.learning import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Integer
from skopt.plots import (
    plot_objective,
    plot_evaluations,
    plot_convergence,
    # plot_objective_2D,
)
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular


# Custom imports
from custom_metrics.metrics import mean_error, lin_ccc, model_efficiency_coefficient


# ------------------- Settings ---------------------------------------------- #


# Set matploblib style
plt.style.use("seaborn-colorblind")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
mpl.rcParams["figure.dpi"] = 450
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["savefig.format"] = "svg"


# # Reset params if needed
# mpl.rcParams.update(mpl.rcParamsDefault)


# ------------------- Organization ------------------------------------------ #


DATA_DIR = pathlib.Path("data/")
SEED = 43


# ------------------- Read and prep data ------------------------------------ #


train_data = np.load(DATA_DIR.joinpath("train_45.npy"))
test_data = np.load(DATA_DIR.joinpath("test_45.npy"))

x_train = train_data[:, 3:]
y_train = train_data[:, 0]
x_test = test_data[:, 3:]
y_test = test_data[:, 0]

# Normalize X
scaler_x = MinMaxScaler()
scaler_x.fit(x_train)
x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)



# Normalize y
scaler_y = MinMaxScaler()
scaler_y.fit(y_train.reshape(-1, 1))
y_train = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
# y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert data to float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)


# ------------------- Feature selection ------------------------------------- #


# Define progress monitoring object
class tqdm_skopt(object):
    """Progress bar object for functions with callbacks."""

    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        """Update bar with intermediate results."""
        self._bar.update()


# # Create the RFE object and compute a cross-validated score.
# rf_fs = RandomForestRegressor(n_estimators=500, max_features=15,
#                               n_jobs=-1, random_state=SEED)
# cv_fs = KFold(n_splits=5, shuffle=True, random_state=SEED)
# rfecv = RFECV(estimator=rf_fs, step=1, cv=cv_fs, min_features_to_select=30,
#               scoring='neg_mean_squared_error', verbose=1)
# rfecv.fit(x_train, y_train)

# length = x_train.shape[-1]
# plt.figure()
# plt.xlabel('Number of features selected')
# plt.ylabel('Cross validation score (Negative MSE)')
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.savefig('rfe2.svg', format='svg')
# plt.show()

# rfe_mask = rfecv.support_
# rfe_rank = rfecv.ranking_
# rfe_scores = rfecv.grid_scores_

# np.save('rfe_scores2.npy', rfe_scores)
# np.save('rfe_rank2.npy', rfe_rank)
# np.save('rfe_mask2.npy', rfe_mask)

# rfe_mask = np.load('rfe_mask2.npy')

# x_train_fs = x_train[:, rfe_mask]
# x_test_fs = x_test[:, rfe_mask]


# ------------------- RF Hyperparameter Optimization ------------------------ #
#%%

# # Define estimator
# estimator = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=SEED)

# # Define cross-validation
# cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

# # Define search space
# n_features = x_train.shape[-1]

# max_depth = 200

# space = [Integer(1, n_features, name="max_features")]
# # space.append(Integer(1, max_depth, name="max_depth"))
# space.append(Integer(2, 50, name="min_samples_split"))
# space.append(Integer(1, 50, name="min_samples_leaf"))


# @use_named_args(space)
# def objective(**params):
#     """Return objective function score for estimator."""
#     # Set hyperparameters from space decorator
#     estimator.set_params(**params)

#     return -np.mean(
#         cross_val_score(
#             estimator,
#             x_train,
#             y_train,
#             cv=cv,
#             n_jobs=-1,
#             scoring="neg_mean_squared_error",
#         )
#     )


# n_calls = 100
# res_gp = gp_minimize(
#     objective,
#     space,
#     n_calls=n_calls,
#     random_state=SEED,
#     callback=[tqdm_skopt(total=n_calls, desc="Gaussian Process")],
# )

# print(
#     f"""Best parameters:
# - max_features={res_gp.x[0]}
# - min_samples_split={res_gp.x[1]}
# - min_samples_leaf={res_gp.x[2]}"""
# )
# # - max_depth={res_gp.x[1]}

# #%%
# plot_convergence(res_gp)
# plot_evaluations(res_gp)
# plot_objective(res_gp)

# ------------------- Training ---------------------------------------------- #
#%%

rf = RandomForestRegressor(
    n_estimators=2500,
    n_jobs=-1,
    max_features=30,
    max_depth=None,
    min_samples_split=7,
    min_samples_leaf=6,
    random_state=SEED,
    criterion="mse",
    verbose=1,
    oob_score=True,
)
rf.fit(x_train, y_train)


# ------------------- Testing ----------------------------------------------- #
#%%

# Predict on test set and reshape pred and true
y_pred = scaler_y.inverse_transform(rf.predict(x_test).reshape(-1, 1))
y_true = y_test.reshape(-1, 1)
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

# Predict on train set and reshape pred and true
y_pred = scaler_y.inverse_transform(rf.predict(x_train).reshape(-1, 1))
y_true = scaler_y.inverse_transform(y_train.reshape(-1, 1))
# Calculate metrics
# r2 = np.exp(scaler_y.inverse_transform(rf.score(x_test, y_test)
#                                        .reshape(1, -1)))[0][0]
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
me = mean_error(y_true, y_pred)
mec = model_efficiency_coefficient(y_true, y_pred)
ccc = lin_ccc(y_true, y_pred)


# %%
feature_names = pd.read_csv(DATA_DIR.joinpath("feature_names_mlp_rf.csv"), index_col=0)

explainer = lime.lime_tabular.LimeTabularExplainer(
    x_train, feature_names=feature_names.feature_name.to_list(), class_names=["SOC"], verbose=True, mode="regression"
)

#%%
ii = [50, 100, 150, 200]
exp = [explainer.explain_instance(x_test[i], rf.predict, num_features=5) for i in ii]
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

mpl.rcParams.update({"lines.linewidth": 1, "font.family": "serif",
                     "xtick.labelsize": "small", "ytick.labelsize": "small",
                     "xtick.major.size": 0, "xtick.minor.size": 0,
                     "ytick.major.size": 0, "ytick.minor.size": 0,
                     "axes.titlesize": "medium", "figure.titlesize": "medium",
                     "figure.figsize": (5, 5), "figure.dpi": 450,
                     "figure.autolayout": True, "savefig.format": "pdf",
                     "savefig.transparent": True, "image.cmap": "seismic_r"})

#%%
full_data = np.vstack((train_data, test_data))
X = full_data[:, 3:]
y = full_data[:, 0].reshape(-1,1)
X = scaler_x.fit_transform(X)
y_pred = scaler_y.inverse_transform(rf.predict(X).reshape(-1, 1))
y_pred_coords = np.hstack((y_pred, full_data[:, 1:3]))
gdf = gpd.GeoDataFrame(y_pred_coords[:, 0], geometry = gpd.points_from_xy(y_pred_coords[:, 2], y_pred_coords[:, 1]), crs = "EPSG:4326")
gdf.insert(1, "True", y)
gdf.insert(2, "Difference", y_pred - y)
gdf.rename(columns={0: "Prediction"}, inplace=True)


# %%
# get country borders
shpfilename = shpreader.natural_earth(resolution = "10m", category = "cultural", name = "admin_0_countries")
# read the shp
shape = gpd.read_file(shpfilename)
# extract germany geom
poly = shape.loc[shape['ADMIN'] == 'Germany']['geometry'].values[0]
# create fig, ax
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.EuroPP()))
# add geometries and features
ax.coastlines(resolution="10m", alpha = 0.3)
ax.add_feature(cfeature.BORDERS, alpha = 0.3)
ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor = "none", edgecolor = '0.5')
# convert gpd to same proj as cartopy map
crs_proj4 = ccrs.EuroPP().proj4_init
gdf_utm32 = gdf.to_crs(crs_proj4)
# Plot
gdf_utm32.plot(ax = ax, marker = ".", markersize = 10, column = "Difference", legend = True, norm=colors.CenteredNorm(), cmap="seismic_r")
# set extent of map
ax.set_extent([5.5, 15.5, 46.5, 55.5], crs=ccrs.PlateCarree())
# fix axes pos
map_ax = fig.axes[0]
leg_ax = fig.axes[1]
map_box = map_ax.get_position()
leg_box = leg_ax.get_position()
leg_ax.set_position([leg_box.x0, map_box.y0, leg_box.width, map_box.height])
# map_ax.set_title("Sample distribution", pad = 10)
leg_ax.set_title("Error (g/kg)", pad = 10)

# save and show fig
# plt.savefig(os.path.join(fig_path, "sample_distribution_soc3.pdf"), bbox_inches = 'tight', pad_inches = 0)
plt.show()

# %%
