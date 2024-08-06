import datetime
import os

import dotenv

dotenv.load_dotenv()
# can set this with the export ERA5_HDF5=/path/to/root/of/h5/files
h5_folder = os.getenv("ERA5_HDF5")

from modulus.distributed import DistributedManager

from earth2mip import registry
from earth2mip.networks import dlwp, pangu, graphcast

package = registry.get_model("e2mip://pangu")
model = pangu.load(package)

from earth2mip.initial_conditions import hdf5, cds

datasource = cds.DataSource(model.in_channel_names)

# Test to see if our datasource is working
time = datetime.datetime(2017, 5, 1, 18)
out = datasource[time]
print(out.shape)

from earth2mip.inference_medium_range import save_scores, time_average_metrics

# Use 12 initializations.
time = datetime.datetime(2017, 1, 2, 0)
initial_times = [
    time + datetime.timedelta(days=30 * i, hours=6 * i) for i in range(12)
]  # modify here to change the initializations

# Output directoy
output_dir = "outputs/03_model_scoring"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
output = save_scores(
    model,
    n=28,  # 6 hour timesteps (28*6/24 = 7-day forecast)
    initial_times=initial_times,
    data_source=datasource,
    time_mean=datasource.time_means,
    output_directory=output_dir,
)


import matplotlib.pyplot as plt
import pandas as pd

from earth2mip.forecast_metrics_io import read_metrics

series = read_metrics(output_dir)
print(series)
dataset = time_average_metrics(series)

plt.close("all")
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
channels = ["z500", "t2m", "t850"]
t = dataset.lead_time / pd.Timedelta("1 h")
for i, channel in enumerate(channels):
    y = dataset.rmse.sel(channel=channel)
    axs[i].plot(t[1:], y[1:])  # Ignore first output as that's just initial condition.
    axs[i].set_xlabel("Lead Time (hours)")
    axs[i].set_ylabel("RMSE")
    axs[i].set_title(f"DLWP {channel} RMSE 2017")

plt.savefig(f"{output_dir}/dwlp_rmse.png")