"""
This module contains functions for visualizing xarray.Dataset objects.
It provides tools for creating maps, plots, and visualizations of climate data.

Author: Jose González-Abad
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import math

def simple_map_plot(data: xr.Dataset, var_to_plot: str, output_path: str=None,
                    colorbar: str='coolwarm', vlimits: tuple=(None, None),
                    num_levels: int=20, central_longitude: int=0,
                    coord_names: dict={'lat': 'lat',
                                       'lon': 'lon'}) -> None:

    """
    This function generates a simple plot of a specific variable from a xr.DataArray
    or xr.Dataset.

    Parameters
    ----------
    data : xr.Dataset
        Xarray dataset to plot. It is important this it does not have a temporal
        dimensions. otherwise this function will show an error.

    var_to_plot : str
        Variable to plot from the xr.Dataset. If data is a xr.DataArray it will
        ignore this parameter.

    output_path : str
        Path inidicating where to save the resulting image (pdf). If it is not
        provided the plot will be returned interactively.

    colorbar : str, optional
        Colorbar to use in the plot (inherited from matplotlib)

    vlimits : tuple, optional
        Limits of the colorbar of the plot. If not indicated this will be computed
        by default.

    num_levels : int, optional
        The amount of levels to use in the colorbar. By default is 20.

    central_longitude : int, optional
        Central longitude for the map projection. Default is 0, which works well
        for most regions like Europe.

    coord_names : dict, optional
        Dictionary with mappings of the name of the spatial dimensions.
        By default lat and lon.

    Returns
    -------
    None
    """                   

    if isinstance(data, xr.Dataset):
        data = data[var_to_plot]

    continuous_cmap = plt.get_cmap(colorbar)
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))    

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_longitude))

    if None in vlimits:
        cs = ax.pcolormesh(data[coord_names['lon']], data[coord_names['lat']],
                        data, transform=ccrs.PlateCarree(),
                        cmap=discrete_cmap)
    else:
        cs = ax.pcolormesh(data[coord_names['lon']], data[coord_names['lat']],
                        data, transform=ccrs.PlateCarree(),
                        cmap=discrete_cmap,
                        vmin=vlimits[0], vmax=vlimits[1])

    ax.coastlines(resolution='10m')
    plt.colorbar(cs, ax=ax, orientation='horizontal')

    plt.title(var_to_plot)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def simple_map_plot_stations(data: xr.Dataset, var_to_plot: str, output_path: str=None,
                             colorbar: str='coolwarm', vlimits: tuple=(None, None),
                             num_levels: int=20, point_size: int=30,
                             point_linewidth: float=0.5,
                             coord_names: dict={'lat': 'lat',
                                                'lon': 'lon'}) -> None:             

    """
    This function generates a simple plot of a specific variable from a xr.DataArray
    or xr.Dataset. This function is tailored for station data indexed by time and 
    station coordinates.

    Parameters
    ----------
    data : xr.Dataset
        Xarray dataset to plot. It is important this it does not have a temporal
        dimensions, otherwise this function will show an error.

    var_to_plot : str
        Variable to plot from the xr.Dataset. If data is a xr.DataArray it will
        ignore this parameter.

    output_path : str
        Path inidicating where to save the resulting image (pdf). If it is not
        provided the plot will be returned interactively.

    colorbar : str, optional
        Colorbar to use in the plot (inherited from matplotlib)

    vlimits : tuple, optional
        Limits of the colorbar of the plot. If not indicated this will be computed
        by default.

    num_levels : int, optional
        The amount of levels to use in the colorbar. By default is 20.

    point_size : int, optional
        The thick of the plotted points. By default is 30.

    point_linewidth : flat, optional
        The width of the lines forming the border of the points. By default
        is 0.5.

    coord_names : dict, optional
        Dictionary with mappings of the name of the spatial dimensions.
        By default lat and lon.

    Returns
    -------
    None
    """     

    if isinstance(data, xr.Dataset):
        data = data[var_to_plot]

    continuous_cmap = plt.get_cmap(colorbar)
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))    

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if None in vlimits:
        cs = ax.scatter(data[coord_names['lon']], data[coord_names['lat']], c=data,
                        s=point_size, edgecolor='k', linewidth=point_linewidth,
                        transform=ccrs.PlateCarree(), zorder=2,
                        cmap=discrete_cmap)
    else:
        cs = ax.scatter(data[coord_names['lon']], data[coord_names['lat']], c=data,
                        s=point_size, edgecolor='k', linewidth=point_linewidth,
                        transform=ccrs.PlateCarree(), zorder=2,
                        vmin=vlimits[0], vmax=vlimits[1],
                        cmap=discrete_cmap)

    ax.coastlines(resolution='10m')
    plt.colorbar(cs, ax=ax, orientation='horizontal')

    plt.title(var_to_plot)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def multiple_map_plot(data: xr.Dataset, output_path: str=None,
                      colorbar: str='coolwarm', vlimits: tuple=(None, None),
                      num_levels: int=20, central_longitude: int=0,
                      coord_names: dict={'lat': 'lat',
                                         'lon': 'lon'}) -> None:           

    """
    This function generates a plot of all variables from a xr.Dataset.

    Parameters
    ----------
    data : xr.Dataset
        Xarray dataset to plot. It is important this it does not have a temporal
        dimensions. otherwise this function will show an error.

    output_path : str
        Path inidicating where to save the resulting image (pdf). If it is not
        provided the plot will be returned interactively.

    colorbar : str, optional
        Colorbar to use in the plot (inherited from matplotlib)

    vlimits : tuple, optional
        Limits of the colorbar of the plot. If not indicated this will be computed
        by default.

    num_levels : int, optional
        The amount of levels to use in the colorbar. By default is 20.

    central_longitude : int, optional
        Central longitude for the map projection. Default is 0, which works well
        for most regions like Europe.

    coord_names : dict, optional
        Dictionary with mappings of the name of the spatial dimensions.
        By default lat and lon.

    Returns
    -------
    None
    """       

    continuous_cmap = plt.get_cmap(colorbar)
    discrete_cmap = ListedColormap(continuous_cmap(np.linspace(0, 1, num_levels)))    

    num_variables = len(data.keys())

    if num_variables == 1:
        print('Warning: For single variable datasets, consider using simple_map_plot instead')
        num_rows, num_cols = 1,1
    elif num_variables == 3:
        num_rows, num_cols = 2,2
    else:
        num_cols = 2
        num_rows = math.ceil(num_variables/num_cols)

    fig = plt.figure(figsize=(20, 20))

    for plot_counter, var_to_plot in enumerate(data.keys(), start=1):
        
        ax = fig.add_subplot(num_rows, num_cols, plot_counter,
                             projection=ccrs.PlateCarree(central_longitude=central_longitude))

        data_to_plot = data[var_to_plot]

        if None in vlimits:
            cs = ax.pcolormesh(data_to_plot[coord_names['lon']], data_to_plot[coord_names['lat']],
                               data_to_plot, transform=ccrs.PlateCarree(),
                               cmap=discrete_cmap)
        else:
            cs = ax.pcolormesh(data_to_plot[coord_names['lon']], data_to_plot[coord_names['lat']],
                               data_to_plot, transform=ccrs.PlateCarree(),
                               cmap=discrete_cmap,
                               vmin=vlimits[0], vmax=vlimits[1])

        ax.set_title(var_to_plot)

        ax.coastlines(resolution='10m')
        
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
        fig.add_axes(ax_cb)
        plt.colorbar(cs, cax=ax_cb, orientation='vertical')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()