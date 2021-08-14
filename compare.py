#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set matploblib style
plt.style.use("seaborn-colorblind")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.transparent"] = True
plt.rcParams["savefig.format"] = "svg"


# %%
# data = {"RF": 32.27, "NN": 32.90, "CNN": 37.3}
data = {
    "RF": {"RMSE": 32.27, "MEC": 0.258, "CCC": 0.438, "ME": 1.466},
    "NN": {"RMSE": 32.90, "MEC": 0.229, "CCC": 0.452, "ME": 6.560},
    "CNN": {"RMSE": 37.3, "MEC": 0.009, "CCC": 0.396, "ME": 2.515},
}
names = list(data.keys())
rmse = [data[key]["RMSE"] for key in list(data.keys())]
mec = rmse = [data[key]["MEC"] for key in list(data.keys())]
ccc = rmse = [data[key]["CCC"] for key in list(data.keys())]
me = rmse = [data[key]["ME"] for key in list(data.keys())]

fig, axs = plt.subplots(4, 1, figsize=(5, 9))
axs[0].plot(names, rmse)
axs[1].plot(names, mec)
axs[2].plot(names, ccc)
axs[3].plot(names, me)
plt.tight_layout()
plt.show()
# %%
print(np.sqrt(928.889), np.sqrt(1555.327))
# %%
