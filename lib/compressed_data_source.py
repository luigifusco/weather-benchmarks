from earth2mip.initial_conditions import base
from earth2mip import grid
import datetime
import xarray as xr
from zarr_filter import J2KFilter
import numpy as np
from pysz import SZ

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
    'z': 'geopotential',
    'w': 'vertical_velocity',
    'r': None,
    'u100m': None,
    'v100m': None,
    'sp': 'surface_pressure',
    'tp': 'total_precipitation',
    'tp06': 'total_precipitation',
    'tisr': 'toa_incident_solar_radiation',
    'zs': 'geopotential_at_surface',
    'lsm': 'land_sea_mask'
}

class SZWrapper:
    def __init__(self, err):
        self.SZ = SZ()
        self.err = err

    def encode(self, b):
        data, ratio = self.SZ.compress(b, 0, self.err, 0, 0)
        return data
    
    def decode(self, b, out):
        out[:] = self.SZ.decompress(b, out.shape, out.dtype)

class CompressedWrapper:
    def __init__(self, base, filter):
        self.filter = filter
        self.base = base
    def __getitem__(self, time: datetime.datetime):
        composite_array = self.base[time].values.astype(np.float32)
        for arr in composite_array:
            encoded = self.filter.encode(arr)
            self.filter.decode(encoded, arr)

        return composite_array

    def __getattr__(self, attr):
        return self.base.__getattribute__(attr)


class ZarrSource(base.DataSource):
    """
    Reads data from a zarr file, expecting the format used by weatherbench2
    """

    grid: grid.LatLonGrid
    
    def __init__(self, channel_names, source='gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2'):
        self._channel_names = channel_names
        self.d = xr.open_zarr(source)
        # self.climatology = xr.open_zarr('gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr')

    @property
    def channel_names(self):
        return self._channel_names
    
    @property
    def grid(self):
        return grid.equiangular_lat_lon_grid(721, 1440)
    
    def _get_var(self, time, short_name, dataset):
        if short_name in MAPPING:
            long_name = MAPPING[short_name]
            data = dataset[long_name].sel(time=time)
        else:
            variable_type, index = short_name[0], short_name[1:]

            long_name = MAPPING[variable_type]
            level = int(index)
            data = dataset[long_name].sel(level=level, time=time).drop('level')
        
        return data

    def __getitem__(self, time: datetime.datetime):
        arrays = []
        for short_name in self.channel_names:
            arrays.append(self._get_var(time, short_name, self.d))
        
        return xr.concat(arrays, 'channel').assign_coords(channel=self.channel_names)

    @property
    def time_means(self):
        return np.zeros((len(self.channel_names), 721, 1440))