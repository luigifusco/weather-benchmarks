import matplotlib.pyplot as plt
import pandas as pd
import os
from earth2mip.forecast_metrics_io import read_metrics

from earth2mip.inference_medium_range import time_average_metrics

output_dir = 'outputs'
compression_ratios = [0, 10, 100, 200, 500, 1000, 2000]

for model in ['graphcast', 'pangu']:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    channels = ["z500", "t2m", "t850"]
    for i, channel in enumerate(channels):
        axs[i].set_xlabel("Lead Time (hours)")
        axs[i].set_ylabel("RMSE")
        axs[i].set_title(f"Pangu {channel} RMSE 2017")
    try:
        for r in compression_ratios:
            print(os.path.join(output_dir, model, str(r)))
            series = read_metrics(os.path.join(output_dir, model, str(r)))
            dataset = time_average_metrics(series)

            t = dataset.lead_time / pd.Timedelta("1 h")
            for i, channel in enumerate(channels):
                y = dataset.rmse.sel(channel=channel)
                axs[i].plot(t[1:], y[1:])  # Ignore first output as that's just initial condition.
    except:
        pass
    axs[0].legend(compression_ratios)
    plt.savefig(f"{model}.png")