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
from earth2mip.networks import pangu, graphcast, fcnv2_sm
from earth2mip.inference_medium_range import save_scores, time_average_metrics
from lib.compressed_data_source import ZarrSource, CompressedWrapper
from zarr_filter import J2KFilter
from filter_wrapper import JP2SPWV_Filter
from earth2mip.initial_conditions import hdf5, cds
from earth2mip.model_registry import Package


graphcast_time_loop = graphcast.load_time_loop_operational(registry.get_model(f"e2mip://graphcast"))
# pangu_time_loop = pangu.load(registry.get_model(f"e2mip://pangu"))
# fcnv2_sm_time_loop = fcnv2_sm.load(registry.get_model(f"e2mip://fcnv2_sm"))

time_loops = [(graphcast_time_loop, 'graphcast')]#, (fcnv2_sm_time_loop, 'fcnv2_sm')]#, (pangu_time_loop, 'pangu')]
compression_ratios = [0, 10, 100, 200, 500, 1000, 2000]
time = datetime.datetime(2017, 1, 2, 0)
initial_times = [
    time #+ datetime.timedelta(days=30 * i, hours=6 * i) for i in range(12)
]  # modify here to change the initializations

for model, model_name in time_loops:
    print(f'testing {model_name}:', model.in_channel_names, '->', model.out_channel_names)
    sys.stdout.flush()

    validation_data_source = ZarrSource(model.out_channel_names, '../data/january.zarr')
    for ratio in compression_ratios:
        print('testing with compression ratio of', ratio)
        sys.stdout.flush()
        output_dir = f"outputs/{model_name}/{ratio}"
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
            time_mean=validation_data_source.time_means,
            output_directory=output_dir,
        )