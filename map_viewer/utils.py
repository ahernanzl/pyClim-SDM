import os
import sys
import numpy as np
import xarray as xr

# Add config paths for settings import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config')))
from settings import *
from advanced_settings import *

def load_data(base, var, climdex_name, scene, model, season):
    """ Load netCDF file as xarray"""
    fname = f"{var}_{climdex_name}_{scene}_{model}_{season}.nc"
    fpath = os.path.join(base, fname)
    return xr.open_dataset(fpath) if os.path.exists(fpath) else None


def get_variable_with_time(ds):
    """
    Return the first variable in dataset that depends on time dimension,
    or fallback to the first variable.
    """
    for var in ds.data_vars:
        if "time" in ds[var].dims:
            return var
    return list(ds.data_vars)[0]


def load_models(base, var, climdex_name, model, scene, season):
    """
    Load selected model or all models (if ensemble mean) as dictionary {modelName: data}
    """
    model_list_local = model_list if model == "ENSEMBLE MEAN" else [model, ]
    data_dict = {}
    for modelName in model_list_local:
        ds_future = load_data(base, var, climdex_name, scene, modelName, season, )
        if ds_future is None:
            continue
        varname = get_variable_with_time(ds_future)
        data = ds_future[varname]
        data_dict.update({modelName: data})
    return data_dict



def get_lat_lon(base, var, climdex_name, model, scene, season):
    """
    Load lat/lon from file as numpy array
    """
    model = model_list[0] if model == "ENSEMBLE MEAN" else model
    ds = load_data(base, var, climdex_name, scene, model, season, )
    lats = ds["lat"].values
    lons = ds["lon"].values
    return lats, lons


def get_years(base, var, climdex_name, model, scene, season):
    """
    Load years from file as numpy array
    """
    model = model_list[0] if model == "ENSEMBLE MEAN" else model
    ds_future = load_data(base, var, climdex_name, scene, model, season, )
    years = ds_future.time.dt.year.values
    return years

def generate_dynamic_periods(min_year, max_year):
    """
    Generate overlapping 30-year periods stepping every 5 years backward
    for dynamic period selection in slider marks.
    """
    periods = []
    start = max_year - 29
    while start >= min_year:
        end = start + 29
        if end > max_year:
            end = max_year
        periods.append((start, end))
        start -= 5
    periods.reverse()
    labels = [f"{start}-{end}" for start, end in periods]
    return periods, labels


def apply_change_mode(field_mean, ref_mean, bias_mode):
    """
    Apply absolute or relative change based on bias_mode.
    If 'abs' subtracts ref_mean, if 'rel' calculates relative (%) change.
    """
    if bias_mode == 'abs':
        return field_mean - ref_mean
    elif bias_mode == 'rel':
        with np.errstate(divide='ignore', invalid='ignore'):
            change = 100 * (field_mean - ref_mean) / ref_mean
            # In case ref_mean is zero or NaN, fallback to NaN
            change = change.where(np.isfinite(change), np.nan)
            return change
    else:
        # Default to absolute if bias_mode unknown
        return field_mean - ref_mean
