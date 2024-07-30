from earth2mip.initial_conditions import base
from earth2mip import grid
import datetime
import xarray as xr
from zarr_filter import J2KFilter
import numpy as np

MAPPING = {
    't': 'temperature',
    'z': 'geopotential',
    'q': 'specific_humidity',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'tcwv': 'total_column_water_vapour',
    't2m': '2m_temperature',
    'msl': 'mean_sea_level_pressure',
    'u10m': '10m_u_component_of_wind',
    'v10m': '10m_v_component_of_wind',
}

class CompressedWrapper:
    def __init__(self, base, filter):
        self.filter = filter
        self.base = base
    def __getitem__(self, time: datetime.datetime):
        arr = np.array(self.base[time], dtype=np.float32)
        encoded = self.filter.encode(arr.tobytes())
        return self.filter.decode(encoded, arr)
    def __getattr__(self, attr):
        return self.base.__getattr__(attr)


class CompressedSource(base.DataSource):

    grid: grid.LatLonGrid
    
    def __init__(self, channel_names):
        self._channel_names = channel_names
        self.d = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2')

    @property
    def channel_names(self):
        return self._channel_names
    
    @property
    def grid(self):
        return grid.equiangular_lat_lon_grid(721, 1440)

    def __getitem__(self, time: datetime.datetime):
        arrays = []
        for short_name in self.channel_names:

            # Extract the alphabetic part and numerical part of the short name
            import re
            match = re.match(r'([a-zA-Z]+)(\d*)', short_name)

            variable_type, index = match.groups()

            # Get the corresponding long name
            long_name = MAPPING[variable_type]

            # Select the correct data variable
            data_var = self.d[long_name]

            # If an index (level) is specified, select the appropriate level
            if index:
                level = int(index)
                data = data_var.sel(level=level, time=time).drop('level')
            else:
                data = data_var.sel(time=time)
            
            arrays.append(data)
        
        # return arrays
        return xr.concat(arrays, 'channel').assign_coords(channel=self.channel_names)