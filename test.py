import xarray as xr
from lib.compressed_data_source import CompressedSource

import datetime
import os

import dotenv
import xarray

dotenv.load_dotenv()

from earth2mip import inference_ensemble, registry

package = registry.get_model("e2mip://pangu")
package = registry.get_model("e2mip://dlwp")

import earth2mip.networks.dlwp as dlwp
import earth2mip.networks.pangu as pangu

# Output directoy
output_dir = "outputs/02_model_comparison"
os.makedirs(output_dir, exist_ok=True)

print("Loading models into memory")
# Load DLWP model from registry
package = registry.get_model("dlwp")
dlwp_inference_model = dlwp.load(package)

# Load Pangu model(s) from registry
package = registry.get_model("pangu")
pangu_inference_model = pangu.load(package)

time = datetime.datetime(2018, 1, 1)

dlwp_data_source = CompressedSource(dlwp_inference_model.in_channel_names)
pangu_data_source = CompressedSource(pangu_inference_model.in_channel_names)

print("Running Pangu inference")
pangu_ds = inference_ensemble.run_basic_inference(
    pangu_inference_model,
    n=24,  # Note we run 24 steps here because Pangu is at 6 hour dt (6 day forecast)
    data_source=pangu_data_source,
    time=time,
)
pangu_ds.to_netcdf(f"{output_dir}/pangu_inference_out.nc")
print(pangu_ds)

print("Running DLWP inference")
dlwp_ds = inference_ensemble.run_basic_inference(
    dlwp_inference_model,
    n=24,  # Note we run 24 steps. DLWP steps at 12 hr dt, but yeilds output every 6 hrs (6 day forecast)
    data_source=dlwp_data_source,
    time=time,
)
dlwp_ds.to_netcdf(f"{output_dir}/dlwp_inference_out.nc")
print(dlwp_ds)

import matplotlib.pyplot as plt

# Open dataset from saved NetCDFs
pangu_ds = xarray.open_dataarray(f"{output_dir}/pangu_inference_out.nc")
dlwp_ds = xarray.open_dataarray(f"{output_dir}/dlwp_inference_out.nc")

# Get data-arrays at 12 hour steps
pangu_arr = pangu_ds.sel(channel="z500").values[::2]
dlwp_arr = dlwp_ds.sel(channel="z500").values[::2]
# Plot
plt.close("all")
fig, axs = plt.subplots(2, 13, figsize=(13 * 4, 5))
for i in range(13):
    axs[0, i].imshow(dlwp_arr[i, 0])
    axs[1, i].imshow(pangu_arr[i, 0])
    axs[0, i].set_title(time + datetime.timedelta(hours=12 * i))

axs[0, 0].set_ylabel("DLWP")
axs[1, 0].set_ylabel("Pangu")
plt.suptitle("z500 DLWP vs Pangu")
plt.savefig(f"{output_dir}/pangu_dlwp_z500.png")