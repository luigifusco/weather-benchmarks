import sys
sys.path.append('compression-filter')
sys.path.append('compression-filter/src')

import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd

from earth2mip.forecast_metrics_io import read_metrics
from modulus.distributed import DistributedManager

from earth2mip import registry
from earth2mip.networks import dlwp, pangu, graphcast
from earth2mip.inference_medium_range import save_scores, time_average_metrics
from lib.compressed_data_source import ZarrSource, CompressedWrapper
from zarr_filter import J2KFilter
from filter_wrapper import JP2SPWV_Filter
from earth2mip.initial_conditions import hdf5, cds

package = registry.get_model("e2mip://pangu")
model = pangu.load(package)

time = datetime.datetime(2017, 1, 2, 0)
initial_times = [
    time #+ datetime.timedelta(days=30 * i, hours=6 * i) for i in range(12)
]  # modify here to change the initializations
print(model.in_channel_names)
validation_data_source = ZarrSource(model.in_channel_names, '../data/january.zarr')

compression_ratios = [0, 10, 100, 200, 500, 1000, 2000]
for ratio in compression_ratios:
    print('testing with compression ratio of', ratio)
    output_dir = f"outputs/{ratio}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    if ratio == 0:
        datasource = validation_data_source
    else:
        # create data source
        filter_opts = JP2SPWV_Filter(
            base_cr=ratio,
            height=721,
            width=1440).hdf_filter_opts
        
        datasource = CompressedWrapper(ZarrSource(model.in_channel_names, '../data/january.zarr'), J2KFilter(filter_opts, libpath='/scratch/lfusco/weather-benchmarks/compression-filter/src/build/lib/libh5z_j2k.so'))
        
    output = save_scores(
        model,
        n=28,  # 6 hour timesteps (28*6/24 = 7-day forecast)
        initial_times=initial_times,
        data_source=datasource,
        validation_data_source=validation_data_source,
        time_mean=datasource.time_means,
        output_directory=output_dir,
    )

    series = read_metrics(output_dir)
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