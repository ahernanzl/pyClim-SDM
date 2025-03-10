"""
This module contains the function compute_ccs, which computes the Climate Change
Signal (CCs) between the future and historical data (future - historical) after
aggregating each of these set of data with the reduction_function.

Author: Jose González-Abad
"""

import xarray as xr
from typing import Callable

"""
The following set of functions define basic statistical operations
to apply to the projections in certain period. These are the basis
of the computed Climate Change Signals (CCs). When defining a new
one make sure that it only takes one argument corresponding to the
projection to reduce. This reduction must be compute across the time
dimension, so the final reduction only spans the spatial coordinates.

No documentation is provided for each of these metrics as the name
of the function is self-descriptive.
"""

def P02(data: xr.Dataset) -> xr.Dataset:
    return data.quantile(0.02, 'time')

def mean(data: xr.Dataset) -> xr.Dataset:
    return data.mean('time')

def P98(data: xr.Dataset) -> xr.Dataset:
    return data.quantile(0.98, 'time')

def TNn(data: xr.Dataset) -> xr.Dataset:
    return data.groupby('time.year').min('time').mean('year')

def TXx(data: xr.Dataset) -> xr.Dataset:
    return data.groupby('time.year').max('time').mean('year')

def R01(data: xr.Dataset, threshold: int=1) -> xr.Dataset:
    return ((data >= threshold) * 1).mean('time')

def SDII(data:xr.Dataset, threshold: int=1) -> xr.Dataset:
    return data.where(x['pr'] >= 1).mean('time')

def RX1day(data: xr.Dataset) -> xr.Dataset:
    return data.groupby('time.year').max('time').mean('year')

"""
"""

def compute_ccs(hist_data: xr.Dataset, fut_data: xr.Dataset,
                reduction_function: Callable, relative: bool) -> xr.Dataset:

    """
    Compute the Climate Change Signal (CCs) between the future and historical
    data (future - historical) after reducing each of these set of data with
    the reduction_function.

    Parameters
    ----------
    hist_data : xr.Dataset
        Historical data to use as reference.

    fut_data : xr.Dataset
        Future data to compute the CCs from.

    reduction_function : Callable
        Function to reduce each of the dataset being compared
        (e.g., mean or TXx). For its proper form look above.

    relative : bool
        Whether to compute the absolute (False) or relative
        CCs.

    Note
    ----
    The CCs must be computed over variables with the same
    name. In case of multiple variables, the CCs will be
    computed across the first variable.

    Returns
    -------
    xr.Dataset
    """

    var_name_hist = list(hist_data.data_vars)[0]
    var_name_fut = list(fut_data.data_vars)[0]
    if not (var_name_hist == var_name_fut):
        raise ValueError('Data variables should have the same name.')

    hist_data_reduced = reduction_function(hist_data)
    fut_data_reduced = reduction_function(fut_data)

    if relative:
        ccs = (fut_data_reduced - hist_data_reduced) / hist_data_reduced
        ccs = ccs * 100
    else:
        ccs = fut_data_reduced - hist_data_reduced

    return ccs
