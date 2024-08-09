import sys
sys.path.append('../compression-filter/src')
sys.path.append('../compression-filter')

import datetime
import os

import dotenv
import xarray

from earth2mip import inference_ensemble, registry
from earth2mip.initial_conditions import cds
from compressed_data_source import ZarrSource, CompressedWrapper
import numpy as np
from zarr_filter import J2KFilter
from filter_wrapper import JP2SPWV_Filter
import matplotlib.pyplot as plt

time = datetime.datetime(2017, 1, 2, 0)

filter_opts = JP2SPWV_Filter(
        base_cr=5000,
        height=721,
        width=1440).hdf_filter_opts

filt = J2KFilter(filter_opts, libpath='/scratch/lfusco/weather-benchmarks/compression-filter/src/build/lib/libh5z_j2k.so')
q1000 = np.array(ZarrSource(['q1000'])[time]).squeeze()

encoded = filt.encode(q1000.tobytes())
decoded = filt.decode(encoded)

plt.imshow(decoded.reshape((721, 1440)))
plt.savefig("saved.png")