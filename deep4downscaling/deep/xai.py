"""
This module contains the functions for computing the saliency maps of the
deep learning models, as well as some XAI-based metrics.

Author: Jose González-Abad
"""

import sys
import xarray as xr
import captum
import torch
import numpy as np
import tqdm

import deep4downscaling.trans as trans

def get_grid_position(mask: xr.Dataset, coord: tuple) -> int:

    """
    This function returns the index in the output of the DL model
    corresponding to coord. This function is employed when the mask
    dataset has latitude and longitude dimensions (grid).

    Parameters
    ----------
    mask : xr.Dataset
        Dataset with no time dimension and latitude and longitude
        coordinates with 1 for valid gridpoints and 0 otherwise.
        In this case, the name of the data variable is irrelevant.

    coord : tuple
        A tuple (latitude, longitude) used to extract the index of the
        closest point to the specified coord.

    Returns
    -------
    int
        Index corresponding to coord.
    """

    # Compute the stacked mask
    mask_stack = mask.stack(gridpoint=('lat', 'lon'))

    # Compute the filtered stacked mask, which corresponds to the
    # output of the DL model
    mask_stack_filt = mask_stack.where(mask_stack==1, drop=True)

    # Compute the nearest latitude and longitude points in mask
    lat_value = mask.sel(lat=coord[0], method='nearest')['lat'].values.item()
    lon_value = mask.sel(lon=coord[1], method='nearest')['lon'].values.item()   

    # Get the index of this specific gridpoint in mask_stack_filt,
    # the output of the DL model. This index corresponds to the neuron
    # to interpret to get information of the gridpoint specified by coord
    coords_to_index = (lat_value, lon_value)
    nearest_point = mask_stack_filt.sel(gridpoint=coords_to_index)
    index = mask_stack_filt['gridpoint'].get_index('gridpoint').get_loc(coords_to_index)   

    return index

def get_station_position(mask: xr.Dataset, coord:tuple) -> int:

    """
    This function returns the index in the output of the DL model
    corresponding to coord. This function is employed when the mask
    dataset has only one dimension (stations).

    Parameters
    ----------
    mask : xr.Dataset
        Dataset with no time dimension. In this case, the name of
        the data variable is irrelevant.

    coord : tuple
        A tuple (latitude, longitude) used to extract the index of the
        closest point to the specified coord.

    Returns
    -------
    int
        Index corresponding to coord.
    """

    delta_lat = mask['lat'] - coord[0]
    delta_lon = mask['lon'] - coord[1]

    distance = np.sqrt(delta_lat**2 + delta_lon**2)

    idx = distance.argmin()
    idx = idx.item()

    return idx

def postprocess_saliency_torch(saliency: torch.tensor,
                               noise_threshold: float) -> torch.tensor:
    
    """
    Apply the postprocessing to the saliency maps presented in
    González-Abad et al. 2024.

    González‐Abad, J., Baño‐Medina, J., & Gutiérrez, J. M. (2023). Using explainability
    to inform statistical downscaling based on deep learning beyond standard validation
    approaches. Journal of Advances in Modeling Earth Systems, 15(11), e2023MS003641.

    Note
    ----
    For variables like precipitation, where the target may contain zeros, saliency can
    return NaNs in such cases (due to division by maximums or sums of these zeros). To
    prevent this, the function replaces NaNs with zeros.

    Parameters
    ----------
    saliency : torch.tensor
        Tensor corresponding to the saliency values to postprocess. It must have
        four dimensions (time, channel, spatial_dim1_, spatial_dim_2).

    noise_threshold : float
        Threshold to filter the noise.

    Returns
    -------
    torch.tensor    
    """

    # Absolute value
    saliency = torch.abs(saliency)

    # Divide by the spatial maximum
    saliency_spatial_max = torch.amax(saliency, dim=(1, 2, 3))
    saliency_spatial_max = saliency_spatial_max[:, None, None, None]
    saliency = saliency / saliency_spatial_max

    # Filter noise
    saliency[saliency < noise_threshold] = 0

    # Normalize
    saliency_spatial_sum = torch.sum(saliency, dim=(1, 2, 3))
    saliency_spatial_sum = saliency_spatial_sum[:, None, None, None]
    saliency = saliency / saliency_spatial_sum

    # Convert nans to zero
    saliency = torch.nan_to_num(saliency, nan=0.0)

    return saliency


def compute_ism(data: xr.Dataset, mask: xr.Dataset,
                model: torch.nn.Module, device: str,
                coord: tuple, xai_method: captum.attr,
                postprocess: bool, noise_threshold: float=0.1) -> xr.Dataset:

    """
    Compute the Individual Saliency Map as presented in González-Abad et al. 2024.
    These corresponds to the saliency maps (across the time spanned by data) for
    a specific gridpoint (coord).

    González‐Abad, J., Baño‐Medina, J., & Gutiérrez, J. M. (2023). Using explainability
    to inform statistical downscaling based on deep learning beyond standard validation
    approaches. Journal of Advances in Modeling Earth Systems, 15(11), e2023MS003641.

    Parameters
    ----------
    data : xr.Dataset
        Dataset corresponding to the input of the DL (predictors) to be used
        to compute the saliency maps. Notice that these need to be transformed
        as if they were going to be passed to the DL model (e.g., standardization).

    mask : xr.Dataset
        Dataset with no time dimension and latitude and longitude
        coordinates with 1 for valid gridpoints and 0 otherwise.
        In this case, the name of the data variable is irrelevant.

    model : torch.nn.Module
        Pytorch model to be interpreted.

    device : str
        Device used to compute the saliency maps (cuda or cpu)

    coord : tuple
        A tuple (latitude, longitude) corresponding to the gridpoint
        to compute the saliency maps from.

    xai_method : captum.attr
        Captum class corresponding to the technique to use to compute
        the saliency maps.

    postprocess : bool
        Whether to postprocess the saliency maps as presented in 
        González-Abad et al. 2024.

    noise_threshold : float, optional
        Threshold to filter the noise if the postprocessing is
        applied.

    Returns
    -------
    xr.Dataset
    """

    saliency_final = data.copy(deep=True)

    # Move model to device
    model = model.to(device)

    # Get the index of the output to interpret
    if ('lat' in mask.dims) and ('lon' in mask.dims): # Grid
        index = get_grid_position(mask=mask, coord=coord)
    elif len(mask.dims) == 1: # Stations
        index = get_station_position(mask=mask, coord=coord)
    else:
        msg_error = """Please provide a mask with either latitude (lat) and
            longitude (lon) coordinates (for station data) or a single coordinate
            corresponding to the stations (station data)."""
        raise ValueError(msg_error)

    # Move data to torch
    data_tensor = trans.xarray_to_numpy(data)
    data_tensor = torch.tensor(data_tensor).to(device)
    data_tensor = data_tensor.requires_grad_() # Set requires_grad_ for captum to work

    print('Computing ISMs...')

    # Compute saliency maps
    saliency_values = xai_method.attribute(data_tensor, target=index)

    # Postprocess the saliency maps
    if postprocess:
        saliency_values = postprocess_saliency_torch(saliency=saliency_values,
                                                     noise_threshold=noise_threshold)

    # Move to numpy
    saliency_values = saliency_values.detach().cpu().numpy()

    # Insert these values into saliency_final
    for idx, var in enumerate(saliency_final.keys()):
        saliency_final[var].values = saliency_values[:, idx, :]

    return saliency_final

def compute_asm(data: xr.Dataset, mask: xr.Dataset,
                model: torch.nn.Module, device: str,
                xai_method: captum.attr, batch_size: int,
                postprocess: bool, noise_threshold: float=0.1) -> xr.Dataset:
    
    """
    Compute the Aggregated Saliency Map (ASM) as defined in González-Abad et al. 2024.
    To do so the saliency maps are computed for all gridpoints for each time step of data.
    As a result we get a saliency map comprising information from all the saliency maps
    (both across gridpoints and time). For further detail we refer to González-Abad 2024.

    González‐Abad, J., Baño‐Medina, J., & Gutiérrez, J. M. (2023). Using explainability
    to inform statistical downscaling based on deep learning beyond standard validation
    approaches. Journal of Advances in Modeling Earth Systems, 15(11), e2023MS003641.

    González Abad, J. (2024). Towards explainable and physically-based deep learning
    statistical downscaling methods. PhD thesis, Universidad de Cantabria.

    Note
    ----
    Due to the nature of ASMs, the computation can be time consuming, especially when 
    working with a large temporal dimension.

    Parameters
    ----------
    data : xr.Dataset
        Dataset corresponding to the input of the DL (predictors) to be used
        to compute the ASM. Notice that these need to be transformed
        as if they were going to be passed to the DL model (e.g., standardization).

    mask : xr.Dataset
        Dataset with no time dimension and latitude and longitude
        coordinates with 1 for valid gridpoints and 0 otherwise.
        In this case, the name of the data variable is irrelevant.

    model : torch.nn.Module
        Pytorch model to be interpreted.

    device : str
        Device used to compute the saliency maps (cuda or cpu)

    xai_method : captum.attr
        Captum class corresponding to the technique to use to compute
        the saliency maps.

    batch_size : int
        Batch size to use for computing the saliency maps. This allows
        to take advantage of the paralleization capabilities of the GPU.

    postprocess : bool
        Whether to postprocess the saliency maps as presented in 
        González-Abad et al. 2024.

    noise_threshold : float, optional
        Threshold to filter the noise if the postprocessing is
        applied.

    Returns
    -------
    xr.Dataset
        The final ASMs spanning all the variables and spatial dimensions of
        data.
    """

    # Move model to device
    model = model.to(device)

    # Move data to torch
    data_tensor = trans.xarray_to_numpy(data)
    data_tensor = torch.tensor(data_tensor).to(device)
    data_tensor = data_tensor.requires_grad_() # Set requires_grad_ for captum to work
    time_span, num_channels, spatial_dim_1, spatial_dim_2 = data_tensor.shape # Name the dimensions

    # We get the number of output gridpoints from the mask
    if ('lat' in mask.dims) and ('lon' in mask.dims): # Grid
        mask_stack = mask.stack(gridpoint=('lat', 'lon'))
        mask_stack_filt = mask_stack.where(mask_stack==1, drop=True)
        num_gridpoints = len(mask_stack_filt['gridpoint'].values)
    elif len(mask.dims) == 1: # Stations
        idx_dim = list(mask.dims)[0]
        num_gridpoints = len(mask[idx_dim])
    else:
        msg_error = """Please provide a mask with either latitude (lat) and
            longitude (lon) coordinates (for station data) or a single coordinate
            corresponding to the stations (station data)."""
        raise ValueError(msg_error)

    # We create an empty torch.tensor to store the ASMs 
    asm_values = torch.zeros(data_tensor.shape).to(device)

    # Set number of batches (this applies to the gridpoints)
    if batch_size == 1:
        num_batches = num_gridpoints
    else:
        num_batches = (num_gridpoints + batch_size - 1) // batch_size

    print('Computing ASMs...')

    # Iterate over time
    for time_step in tqdm.tqdm(range(time_span)):

        # Subset the input data from time_step
        data_tensor_time_step = data_tensor[time_step, :]
        data_tensor_time_step = torch.unsqueeze(data_tensor_time_step, 0)

        # Iterate over gridpoints
        for batch in range(num_batches):

            # Set batch indices
            start_batch = batch * batch_size
            end_batch = min((batch + 1) * batch_size, num_gridpoints)

            # Get gridpoints in the batch
            gridpoints_batch = list(range(start_batch, end_batch))

            # Replicate the input data to match the length of the gridpoints in
            # the batch. This is a requirement of captum.
            data_tensor_time_step_rep = data_tensor_time_step.repeat(len(gridpoints_batch), 1, 1, 1)

            # Compute the saliency of the gridpoints in the batch
            xai_values = xai_method.attribute(data_tensor_time_step_rep,
                                              target=list(range(start_batch, end_batch)))

            # Postprocess the saliency maps
            if postprocess:
                xai_values = postprocess_saliency_torch(saliency=xai_values,
                                                        noise_threshold=noise_threshold)

            # Accumulate the saliency maps for the gridpoints in the batch
            xai_values = torch.sum(xai_values, dim=0) # Sum across the gridpoints on the batch
            asm_values[time_step, :] = asm_values[time_step, :] + xai_values

        # Compute the ASM for the time_step by aggregating across gridpoints
        asm_values[time_step, :] = asm_values[time_step, :] / (num_gridpoints+1)

    # Compute the final ASM by aggregating across the time_span
    asm_values = torch.mean(asm_values, dim=0)

    # Convert to numpy
    asm_values = asm_values.detach().cpu().numpy()

    # Insert the ASM values into a xarray Dataset
    asm_dataset = data.sel(time=data['time'].values[0])
    asm_dataset = asm_dataset.drop_vars('time')

    for idx, var in enumerate(asm_dataset.keys()):
        asm_dataset[var].values = asm_values[idx]

    return asm_dataset

def haversine_distance(predictor_lats: np.ndarray, predictor_lons: np.ndarray,
                       target_lat: float, target_lon: float) -> np.ndarray:

    """
    Compute the Haversine distance between the point (target_lan, target_lon)
    and each of the points (predictor_lats, predictor_lons). For more
    details see https://en.wikipedia.org/wiki/Haversine_formula

    Parameters
    ----------
    predictor_lats : np.ndarray
        Array of latitudes.

    predictor_lons : np.ndarray
        Array of longitudes.

    target_lat : float
        Latitude of the point to which to calculate the distance.

    target_lon : float
        Longtiude of the point to which to calculate the distance.

    Returns
    -------
    np.ndarray
        Array with the Haversine distance of (target_lan, target_lon)
        to each of the points (predictor_lats, predictor_lons)
    """

    # R in Haversine formula
    earth_radius = 6371

    # Transform from degrees to radians
    target_lat, target_lon = np.radians(target_lat), np.radians(target_lon)

    predictor_lons_matrix, predictor_lats_matrix = \
        np.meshgrid(predictor_lons, predictor_lats)
    predictor_coords = np.stack((predictor_lats_matrix,
                                 predictor_lons_matrix), axis=-1)

    # Iterate over coordinates in the predictor
    haversine_values = np.empty(predictor_coords.shape[:2])
    for i in range(predictor_coords.shape[0]):
        for j in range(predictor_coords.shape[1]):

            # Compute Haverstine distance
            x_lat, x_lon = predictor_coords[i, j, :]

            # Transform from degrees to radians
            x_lat, x_lon = np.radians(x_lat), np.radians(x_lon)

            part_1 = np.sin((x_lat - target_lat)/2) ** 2
            part_2 = np.cos(target_lat) * np.cos(x_lat)
            part_3 = np.sin((x_lon - target_lon)/2) ** 2
            haversine = 2 * earth_radius * np.arcsin((part_1 + part_2 * part_3) ** (1/2))

            haversine_values[i, j] = haversine

    return haversine_values  

def compute_sdm(data: xr.Dataset, mask: xr.Dataset, var_target: str,
                model: torch.nn.Module, device: str,
                xai_method: captum.attr, batch_size: int,
                postprocess: bool, noise_threshold: float=0.1) -> xr.Dataset:

    """
    Compute the Saliency Dispersion Map (SDM) as defined in González-Abad et al. 2024.
    To do so the saliency maps are computed for all gridpoints for each time step of data.
    As a result we get, for each gridpoint in the output, a dispersion value indicating how
    local (in the predictor's space) are the saliency maps/ For further detail we refer to
    González-Abad 2024.

    González‐Abad, J., Baño‐Medina, J., & Gutiérrez, J. M. (2023). Using explainability
    to inform statistical downscaling based on deep learning beyond standard validation
    approaches. Journal of Advances in Modeling Earth Systems, 15(11), e2023MS003641.

    González Abad, J. (2024). Towards explainable and physically-based deep learning
    statistical downscaling methods. PhD thesis, Universidad de Cantabria.

    Note
    ----
    Due to the nature of SDMs, the computation can be time consuming, especially when 
    working with a large temporal dimension.

    Parameters
    ----------
    data : xr.Dataset
        Dataset corresponding to the input of the DL (predictors) to be used
        to compute the SDM. Notice that these need to be transformed
        as if they were going to be passed to the DL model (e.g., standardization).

    mask : xr.Dataset
        Dataset with no time dimension and latitude and longitude
        coordinates with 1 for valid gridpoints and 0 otherwise.

    var_target : str
        Variable to interpret. Despite being required, this argument is 
        useful when working with station-based data, for which other
        variables (e.g., station id) may be included in the mask. This
        argument is only required for the SDM as we need to fill the 
        appropiate variable in the mask.

    model : torch.nn.Module
        Pytorch model to be interpreted.

    device : str
        Device used to compute the saliency maps (cuda or cpu)

    xai_method : captum.attr
        Captum class corresponding to the technique to use to compute
        the saliency maps.

    batch_size : int
        Batch size to use for computing the saliency maps. This allows
        to take advantage of the paralleization capabilities of the GPU.

    postprocess : bool
        Whether to postprocess the saliency maps as presented in 
        González-Abad et al. 2024.

    noise_threshold : float, optional
        Threshold to filter the noise if the postprocessing is
        applied.

    Returns
    -------
    xr.Dataset
        The final SDMs spanning the spatial dimension of the mask.
    """

    # Move model to device
    model = model.to(device)

    # Move data to torch
    data_tensor = trans.xarray_to_numpy(data)
    data_tensor = torch.tensor(data_tensor).to(device)
    data_tensor = data_tensor.requires_grad_() # Set requires_grad_ for captum to work
    time_span, num_channels, spatial_dim_1, spatial_dim_2 = data_tensor.shape # Name the dimensions

    # Get the number of output gridpoints from the mask
    if ('lat' in mask.dims) and ('lon' in mask.dims): # Grid
        mask_stack = mask.stack(gridpoint=('lat', 'lon'))
        mask_stack_filt = mask_stack.where(mask_stack==1, drop=True)
        num_gridpoints = len(mask_stack_filt['gridpoint'].values)
    elif len(mask.dims) == 1: # Stations
        idx_dim = list(mask.dims)[0]
        num_gridpoints = len(mask[idx_dim])
    else:
        msg_error = """Please provide a mask with either latitude (lat) and
            longitude (lon) coordinates (for station data) or a single coordinate
            corresponding to the stations (station data)."""
        raise ValueError(msg_error)

    # Create an empty torch.tensor to store the SDMs 
    sdm_values = torch.zeros(num_gridpoints).to(device)

    # Precompute the haversine distance for each gridpoint in the output
    print('Precomputing Haversine distances...')

    predictor_lats = data['lat'].values
    predictor_lons = data['lon'].values

    hv_dists = np.zeros((num_gridpoints,
                         spatial_dim_1, spatial_dim_2))

    for gp_idx in range(num_gridpoints):
        if ('lat' in mask.dims) and ('lon' in mask.dims): # Grid
            target_lat, target_lon = mask_stack_filt['gridpoint'].values[gp_idx]
        elif len(mask.dims) == 1: # Stations
            target_lat, target_lon = mask['lat'].values[gp_idx], mask['lon'].values[gp_idx]

        hv_dists[gp_idx, :] = haversine_distance(predictor_lats=predictor_lats,
                                                 predictor_lons=predictor_lons,
                                                 target_lat=target_lat,
                                                 target_lon=target_lon)

    hv_dists = torch.tensor(hv_dists, dtype=torch.float32).to(device)

    # Set number of batches (this applies to the gridpoints)
    if batch_size == 1:
        num_batches = num_gridpoints
    else:
        num_batches = (num_gridpoints + batch_size - 1) // batch_size

    print('Computing SDMs...')

    # Iterate over time
    for time_step in tqdm.tqdm(range(time_span)):

        # Subset the input data from time_step
        data_tensor_time_step = data_tensor[time_step, :]
        data_tensor_time_step = torch.unsqueeze(data_tensor_time_step, 0)

        # Iterate over gridpoints
        for batch in range(num_batches):

            # Set batch indices
            start_batch = batch * batch_size
            end_batch = min((batch + 1) * batch_size, num_gridpoints)

            # Get gridpoints in the batch
            gridpoints_batch = list(range(start_batch, end_batch))

            # Replicate the input data to match the length of the gridpoints in
            # the batch. This is a requirement of captum.
            data_tensor_time_step_rep = data_tensor_time_step.repeat(len(gridpoints_batch), 1, 1, 1)

            # Compute the saliency of the gridpoints in the batch
            xai_values = xai_method.attribute(data_tensor_time_step_rep,
                                              target=list(range(start_batch, end_batch)))

            # Postprocess the saliency maps
            if postprocess:
                xai_values = postprocess_saliency_torch(saliency=xai_values,
                                                        noise_threshold=noise_threshold)

            # Sum across variables
            xai_values = torch.sum(xai_values, dim=1)

            # Multiply by the Haversine distance
            xai_values = xai_values * hv_dists[gridpoints_batch, :]

            # Sum across features
            xai_values = torch.sum(xai_values, dim=(1, 2))

            # Accumulate the SDMs for the gridpoints in the batch
            sdm_values[gridpoints_batch] = sdm_values[gridpoints_batch] + xai_values

    # Compute the final SDM by dividing by the time_span
    sdm_values = sdm_values / time_span

    # Convert to numpy
    sdm_values = sdm_values.detach().cpu().numpy()  

    # Insert the SDM values into a xarray Dataset. For this, mask
    # is used as template
    sdm_dataset = mask.copy(deep=True)
    sdm_dataset[var_target] = sdm_dataset[var_target].astype('float32')

    # Fill the final dataset
    if ('lat' in mask.dims) and ('lon' in mask.dims): # Grid
        one_indices = (sdm_dataset[var_target].values == 1)
        sdm_dataset[var_target].values[one_indices] = sdm_values
        sdm_dataset[var_target].values[~one_indices] = np.nan
    elif len(mask.dims) == 1: # Stations
        sdm_dataset[var_target].values = sdm_values

    return sdm_dataset