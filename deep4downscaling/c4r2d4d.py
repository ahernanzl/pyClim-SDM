"""
This module contains the function c4r2d4d, which transforms a .RData object returned
by climate4R functions into an xarray.Dataset.

Author: Oscar Mirones
"""

import numpy as np
import xarray as xr
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def c4r2d4d(c4r):

    """
    This function transforms a .RData object returned by climate4R functions into an
    xarray.Dataset for further manipulation and analysis in Python. It extracts key
    components from the input R object, such as the variable(s), data, spatial coordinates,
    temporal information, and optional metadata, and reorganizes them into an xarray.Dataset.
    The function maps spatial coordinates to longitude (lon) and latitude (lat) for gridded
    data or assigns them to a loc dimension for flattened datasets. Temporal information is
    converted into Python's datetime format for easier time-based indexing. The function
    dynamically handles multiple dimensions, such as time, lat, lon, loc, and member,
    depending on the structure of the input data. It ensures that the resulting dataset is
    compatible with Python's geospatial libraries, allowing users to efficiently work with climate
    data in Python. Additionally, any metadata from the input R object is preserved and included as
    global attributes in the output xarray.Dataset.

    Parameters
    ----------
    c4r
        The .RData object returned by climate4R functions containing the climate data and metadata.
        
    Returns
    -------
    xarray.Dataset 
        The transformed dataset in Python's xarray format, with labeled dimensions and coordinates for easy manipulation and analysis.
    """
    base = importr('base')
    
    # Extract components 
    variable = c4r.rx2('Variable')  
    data = c4r.rx2('Data')  
    xyCoords = c4r.rx2('xyCoords')  
    dates = c4r.rx2('Dates')  
    metadata = c4r.rx2('Metadata') if 'Metadata' in c4r.names else None

    dim_names = base.attr(data, 'dimensions')  
    var_names = list(variable[0])  

    # Convert data to numpy array
    data_array = np.array(data)  

    # Extract coordinates
    lon = np.array(xyCoords.rx2('x'))
    lat = np.array(xyCoords.rx2('y'))

    # Extract time information
    dates_start = np.array(dates.rx2('start') if len(variable.rx2('varName')) == 1 else dates[0].rx2('start'))
    time = pd.to_datetime(dates_start).tz_localize(None)

    # Create dynamic Dataset with the variables involved
    
    data_vars = {}
    loc_x, loc_y = np.array(xyCoords.rx2('x')), np.array(xyCoords.rx2('y'))

    def create_dataset(var_name, dims, data_slice, coords):
        data_vars[var_name] = (dims, data_slice)
        return xr.Dataset(data_vars=data_vars, coords=coords)

    if "var" in dim_names:
        for idx, var_name in enumerate(var_names):
            if "loc" in dim_names:
                if "member" in dim_names:
                    dims = ["member", "time", "loc"]
                    if "Members" in c4r.names:
                        members = seasonal.rx2('Members')
                        new_ds = create_dataset(
                            var_name, dims, data_array[ :, :, :],
                            coords={
                                "lon": ("loc", loc_x),
                                "lat": ("loc", loc_y),
                                "time": ("time", time),
                                "member": ("member", members)
                            }
                        )
                    else:
                        idx_mem = np.where(np.array(dim_names) == 'member')[0][0]
                        new_ds = create_dataset(
                            var_name, dims, data_array[idx, :, :, :],
                            coords={
                                "lon": ("loc", loc_x),
                                "lat": ("loc", loc_y),
                                "time": ("time", time),
                                "member": ("member", np.arange(data_array.shape[idx_mem]))
                            }
                        )
                else:
                    dims = ["time", "loc"]
                    new_ds = create_dataset(
                        var_name, dims, data_array[idx, :, :],
                        coords={
                            "lon": ("loc", loc_x),
                            "lat": ("loc", loc_y),
                            "time": ("time", time)
                        }
                    )
            else:
                if "member" in dim_names:
                    dims = ["member", "time", "lat", "lon"]
                    if "Members" in c4r.names:
                        members = seasonal.rx2('Members')
                        new_ds = create_dataset(
                        var_name, dims, data_array[ :, :, :, :],
                        coords={
                            "lon": ("lon", lon),
                            "lat": ("lat", lat),
                            "time": ("time", time),
                            "member": ("member", members)
                        }
                    )
                    else: 
                        idx_mem = np.where(np.array(dim_names) == 'member')[0][0]
                        new_ds = create_dataset(
                            var_name, dims, data_array[idx, :, :, :, :],
                            coords={
                                "lon": ("lon", lon),
                                "lat": ("lat", lat),
                                "time": ("time", time),
                                "member": ("member", np.arange(data_array.shape[idx_mem]))
                            }
                        )
                else:
                    dims = ["time", "lat", "lon"]
                    new_ds = create_dataset(
                        var_name, dims, data_array[idx, :, :, :],
                        coords={
                            "lon": ("lon", lon),
                            "lat": ("lat", lat),
                            "time": ("time", time)
                        }
                    )
    else:
        if "loc" in dim_names:
            if "member" in dim_names:
                dims = ["member", "time", "loc"]
                if "Members" in c4r.names:
                    members = seasonal.rx2('Members')
                    new_ds = create_dataset(
                        var_name, dims, data_array[ :, :, :, :],
                        coords={
                            "lon": ("loc", loc_x),
                            "lat": ("loc", loc_y),
                            "time": ("time", time),
                            "member": ("member", members)
                        }
                    )
                else:
                    idx_mem = np.where(np.array(dim_names) == 'member')[0][0]
                    new_ds = create_dataset(
                        var_names[0], dims, data_array[:, :, :],
                        coords={
                            "lon": ("loc", loc_x),
                            "lat": ("loc", loc_y),
                            "time": ("time", time),
                            "member": ("member", np.arange(data_array.shape[idx_mem]))
                        }
                    )
            else:
                dims = ["time", "loc"]
                new_ds = create_dataset(
                    var_names[0], dims, data_array[:, :],
                    coords={
                        "lon": ("loc", loc_x),
                        "lat": ("loc", loc_y),
                        "time": ("time", time)
                    }
                )
        else:
            if "member" in dim_names:
                dims = ["member", "time", "lat", "lon"]
                if "Members" in c4r.names:
                    members = seasonal.rx2('Members')
                    new_ds = create_dataset(
                        var_names[0], dims, data_array[:, :, :, :],
                        coords={
                            "lon": ("lon", lon),
                            "lat": ("lat", lat),
                            "time": ("time", time),
                            "member": ("member", members)
                        }
                    )
                else:
                    idx = np.where(np.array(dim_names) == 'member')[0][0]
                    new_ds = create_dataset(
                        var_names[0], dims, data_array[:, :, :, :],
                        coords={
                            "lon": ("lon", lon),
                            "lat": ("lat", lat),
                            "time": ("time", time),
                            "member": ("member", np.arange(data_array.shape[idx_mem]))
                        }
                    )
            else:
                dims = ["time", "lat", "lon"]
                new_ds = create_dataset(
                    var_names[0], dims, data_array[:, :, :],
                    coords={
                        "lon": ("lon", lon),
                        "lat": ("lat", lat),
                        "time": ("time", time)
                    }
                )
                
    # Assign metadata as attributtes if present
    if metadata is not None:
        new_ds.attrs = dict(zip(list(metadata.names), list(metadata)))

        # Assign metadata as attributes
        new_ds.attrs = dict(zip(list(metadata.names), list(metadata)))
 

    return new_ds