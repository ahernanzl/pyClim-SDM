"""
This module contains the functions for computing the predictions of the
deep learning models.

Author: Jose González-Abad
"""

import sys
import sys
import gc
import types
import torch
import numpy as np
import xarray as xr

import deep4downscaling.trans as trans

def _predict(model: torch.nn.Module, device: str,
             batch_size: int=None, **data: np.ndarray,) -> np.ndarray:

    """
    Internal function to compute the prediction of a certain DL model given
    some input data. This function is able to handle DL models with any number
    of inputs to their forward() method.

    Parameters
    ----------
    model : torch.nn.Module
        Model used to compute the predictions.
    
    device : str
        Device used to run the inference (cuda or cpu).

    batch_size : int, optional
        If provided the predictions are computed in batches of size
        batch_size. This is useful when facing OOM errors.

    data : np.ndarray
        Input/Inputs to the model. There are no restrictions for the 
        argument name.

    Notes
    -----
    If the model predicts multiple tensors, these are returned as a
    tuple of np.ndarray(s).

    Returns
    -------
    np.ndarray
    """

    model = model.to(device)

    for key, value in data.items():
        data[key] = torch.tensor(data[key])
    num_samples = data[key].shape[0]

    model.eval()

    # No batches
    if batch_size is None:
        data_values = [x.to(device) for x in data.values()]
        with torch.no_grad():
            y_pred = model(*data_values)

    # Batches
    elif isinstance(batch_size, int):
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):

                # Compute the prediction of a specific batch
                end_idx = min(i + batch_size, num_samples)
                batch_data = [x[i:end_idx, :].to(device) for x in data.values()]
                batch_pred = model(*batch_data)

                # Initialize the torch accumulating all batches
                if i == 0:
                    # TODO: Compute predictions by batches when the model returns
                    # multiple tensors
                    if isinstance(batch_pred, tuple):
                        raise ValueError('Not implemented. Set batch_size=None')
                    y_pred = torch.zeros(num_samples, *batch_pred.shape[1:])

                y_pred[i:end_idx, :] = batch_pred

    if isinstance(y_pred, tuple): # Handle DL models returning multiple tensors
        y_pred = (x.cpu().numpy() for x in y_pred)
    else:
        y_pred = y_pred.cpu().numpy()

    return y_pred

def _pred_to_xarray(data_pred: np.ndarray, time_pred: np.ndarray,
                    var_target: str, mask: xr.Dataset,
                    spatial_dims: tuple[str, str]) -> xr.Dataset:

    """
    This internal function transforms the prediction from a DL model
    (np.ndarray) into a xr.Dataset with the corresponding temporal and
    spatial dimensions. To do so it takes as input a mask (xr.Dataset)
    defining where to introduce (spatial dimension) the predictions of
    the DL model, returning as output a xr.Dataset filled with the
    data_pred. This function requires the time dimension of the 
    output xr.Dataset, which should be easily obtained from the predictor
    of the DL model, to be computed from another xr.Dataset.

    Parameters
    ----------
    data_pred : np.ndarray
        Predictions of a DL model. Generally these are computed using the
        _predict internal function.

    time_pred : np.ndarray
        Array containing the temporal coordinated of the output xr.Dataset
        (in datetime64[ns]).

    var_target: str
        Target variable.

    mask: xr.Dataset
        Mask with no temporal dimension formed by ones/zeros for (spatial)
        positions to introduce data_pred/np.nans values.

    spatial_dims: tuple[str, str]
        Names of the spatial dimensions (e.g., ('lat', 'lon') or ('y', 'x')).

    Returns
    -------
    xr.Dataset
        The data_pred argument properly transformed to xr.Dataset.
    """

    # Expand the mask in the time dimension of the final prediction
    mask = mask.expand_dims(time=time_pred)
    mask = mask.ffill(dim='time')

    # By default xarray casts the mask to int64 but we need a 
    # float-based type
    mask[var_target] = mask[var_target].astype('float32')
    
    # Stack following the procedure perform in all these modules
    # mask = mask.stack(gridpoint=spatial_dims)
    mask = mask.stack(gridpoint=spatial_dims)

    # Assign the perdiction to the gridpoints of the mask with value 1
    # For the 0 ones we assign them np.nan
    one_indices = (mask[var_target].values == 1)
    
    # For fully-convolutional models, for which the prediction includes
    # nan and non-nans values of the y_mask
    if mask['gridpoint'].shape[0] == data_pred.shape[1]:
        mask[var_target].values = data_pred
    # For DeepESD-like models, for which the predictons only corresponds
    # to the non-nan values
    else:
        mask[var_target].values[one_indices] = data_pred.flatten()
    
    mask[var_target].values[~one_indices] = np.nan

    # Unstack and return
    mask = mask.unstack()
    return mask

def _pred_stations_to_xarray(data_pred: np.ndarray, time_pred: np.ndarray,
                             var_target: str, template: xr.Dataset) -> xr.Dataset:

    """
    This internal function transforms the prediction from a DL model
    (np.ndarray) into a xr.Dataset with the corresponding temporal and
    station dimensions. To do so it creates an empty xr.Dataset (across)
    the time dimension and fill the values with the predictiones returned
    by the DL model. This function can handle additional variables such
    as the station indices/names.

    Parameters
    ----------
    data_pred : np.ndarray
        Predictions of a DL model. Generally these are computed using the
        _predict internal function.

    time_pred : np.ndarray
        Array containing the temporal coordinated of the output xr.Dataset
        (in datetime64[ns]).

    var_target: str
        Target variable.

    template: xr.Dataset
        Template with no temporal dimension.

    Returns
    -------
    xr.Dataset
        The data_pred argument properly transformed to xr.Dataset.
    """

    # Ignore other dims (e.g., station info)
    template_var = template[var_target] 

    # Expand the template in the time dimension of the final prediction
    template_var = template_var.expand_dims(time=time_pred)
    template_var = template_var.ffill(dim='time')

    # For DeepESD-like models, for which the predictons only corresponds
    # to the non-nan values
    template_var.values = data_pred

    # Add created variable as well as ignored ones
    for var_name in template.data_vars:
        if var_name == var_target:
            template = template.assign({var_name: template_var})
        else:
            template = template.assign({var_name: template[var_name]})

    return template

def compute_preds_standard(x_data: xr.Dataset, model: torch.nn.Module, device: str,
                           var_target: str,
                           mask: xr.Dataset=None, template: xr.Dataset=None,
                           ensemble_size: int=None,
                           batch_size: int=None,
                           spatial_dims: tuple[str, str]=('lat', 'lon')) -> xr.Dataset:

    """
    Given some xr.Dataset with predictor data, this function returns the prediction
    of the DL model (in the proper format) given x_data as input. This function is
    designed to work with models computing the final prediction
    (e.g., MSE-based models).

    Notes
    -----
    This function relies on the mask or template as a key input to convert the
    raw model output into a properly formatted xr.Dataset. The mask is used for
    gridded data, while the template is used for station-based data. The argument
    provided determines the function's internal behavior. Only one of these inputs
    should be supplied; providing none/both will result in an error.

    Parameters
    ----------
    x_data : xr.Dataset
        Predictors to pass as input to the DL model. They must have a spatial
        (e.g., lat and lon) and temporal dimension.

    model : torch.nn.Module
        Pytorch model to use.

    device : str
        Device used to run the inference (cuda or cpu).

    var_target : str
        Target variable.

    mask : xr.Dataset
        Mask with no temporal dimension formed by ones/zeros for (spatial)
        positions to introduce data_pred/np.nans values.

    template : xr.Dataset
        Mask with no temporal dimension used as basis for building the 
        final xr.Dataset for stations-based data.

    ensemble_size : int, optional
        If provided, it indicates the number of samples computed by running
        the model ensemble_size times. These are saved as a new dimension in
        the xr.Dataset (member).

    batch_size : int, optional
        If provided the predictions are computed in batches of size
        batch_size. This is useful when facing OOM errors.

    spatial_dims : tuple[str, str], optional
        Names of the spatial dimensions for gridded data, defaults to ('lat', 'lon').
        Relevant only when `mask` is provided.

    Returns
    -------
    xr.Dataset
        The final prediction
    """
    
    if not ensemble_size:
        ensemble_size = 1

    x_data_arr = trans.xarray_to_numpy(x_data)

    # Check for the mask and template
    if mask and template:
        raise ValueError('Provide either a mask or a template.')
    if (not mask) and (not template):
        raise ValueError('Provide either a mask or a template, not both.')

    # Add channel dimension for one-dimensional predictors
    if len(list(x_data.keys())) <= 1:
        x_data_arr = x_data_arr[:, None, :, :] # Add empty channel dimension

    time_pred = x_data['time'].values

    # Compute and concatenate ensemble_size predictions
    data_pred = []
    for _ in range(ensemble_size):
        data_aux = _predict(model=model, device=device, x_data=x_data_arr,
                            batch_size=batch_size)
        if mask:
            data_aux = _pred_to_xarray(data_pred=data_aux, time_pred=time_pred,
                                       var_target=var_target, mask=mask,
                                       spatial_dims=spatial_dims)
        elif template:
            data_aux = _pred_stations_to_xarray(data_pred=data_aux, time_pred=time_pred,
                                                var_target=var_target, template=template)
        data_pred.append(data_aux)

    # Return the Dataset
    if ensemble_size == 1:
        data_final = data_pred[0]
    else:
        data_final = xr.concat(data_pred, dim='member')

    return data_final

def compute_preds_gaussian(x_data: xr.Dataset, model: torch.nn.Module, device: str,
                           var_target: str, mask: xr.Dataset,
                           ensemble_size: int=None,
                           batch_size: int=None,
                           spatial_dims: tuple[str, str]=('lat', 'lon')) -> xr.Dataset:

    """
    Given some xr.Dataset with predictor data, this function returns the prediction
    of the DL model (in the proper format) given x_data as input. This function
    tailors the prediction of DL models trained to minimize the NLL of a Gaussian
    distribution.

    Notes
    -----
    For this function the mask is key, as it allows to convert the raw output of
    the model to the proper xr.Dataset representation.

    Parameters
    ----------
    x_data : xr.Dataset
        Predictors to pass as input to the DL model. They must have a spatial
        (e.g., lat and lon) and temporal dimension.

    model : torch.nn.Module
        Pytorch model to use.

    device : str
        Device used to run the inference (cuda or cpu).

    var_target : str
        Target variable.

    mask : xr.Dataset
        Mask with no temporal dimension formed by ones/zeros for (spatial)
        positions to introduce data_pred/np.nans values.

    ensemble_size : int, optional
        If provided, it indicates the number of samples computed by running
        the model ensemble_size times. These are saved as a new dimension in
        the xr.Dataset (member).

    batch_size : int, optional
        If provided the predictions are computed in batches of size
        batch_size. This is useful when facing OOM errors.

    spatial_dims : tuple[str, str], optional
        Names of the spatial dimensions, defaults to ('lat', 'lon').

    Returns
    -------
    xr.Dataset
        The final prediction
    """
    
    if not ensemble_size:
        ensemble_size = 1

    x_data_arr = trans.xarray_to_numpy(x_data)
    time_pred = x_data['time'].values

    data_pred = _predict(model=model, device=device, x_data=x_data_arr,
                         batch_size=batch_size)

    # If the model return varios tensors, I assume the first one is the
    # one containing the predicted parameters (e.g., elevation case)
    if isinstance(data_pred, types.GeneratorType):
        data_pred = list(data_pred)[0]

    # Get the parameters of the Gaussian dist.
    dim_target = data_pred.shape[1] // 2
    mean = data_pred[:, :dim_target]
    log_var = data_pred[:, dim_target:]
    s_dev = np.exp(log_var) ** (1/2)

    # Compute the prediction
    data_pred = []
    for _ in range(ensemble_size):    
        data_aux = np.random.normal(loc=mean, scale=s_dev)
        data_aux = _pred_to_xarray(data_pred=data_aux, time_pred=time_pred,
                                   var_target=var_target, mask=mask,
                                   spatial_dims=spatial_dims)
        data_pred.append(data_aux)

    # Return the Dataset
    if ensemble_size == 1:
        data_final = data_pred[0]
    else:
        data_final = xr.concat(data_pred, dim='member')

    # BUG FIX: Was returning data_pred list instead of data_final
    return data_final

def compute_preds_ber_gamma(x_data: xr.Dataset, model: torch.nn.Module, threshold: float,
                            device: str, var_target: str, mask: xr.Dataset,
                            ensemble_size: int=None,
                            batch_size: int=None,
                            spatial_dims: tuple[str, str]=('lat', 'lon')) -> xr.Dataset:

    """
    Given some xr.Dataset with predictor data, this function returns the prediction
    of the DL model (in the proper format) given x_data as input. This function
    tailors the prediction of DL models trained to minimize the NLL of a Bernoulli
    and gamma distributions.

    Notes
    -----
    For this function the mask is key, as it allows to convert the raw output of
    the model to the proper xr.Dataset representation.

    Parameters
    ----------
    x_data : xr.Dataset
        Predictors to pass as input to the DL model. They must have a spatial
        (e.g., lat and lon) and temporal dimension.

    model : torch.nn.Module
        Pytorch model to use.

    threshold : float
        The value used as threshold to define the precipitation for fitting the
        gamma distribution (deep4downscaling.utils.precipitation_NLL_trans). This
        is required to correct the effect of this transformation in the final
        prediction.

    device : str
        Device used to run the inference (cuda or cpu).

    var_target : str
        Target variable.

    mask : xr.Dataset
        Mask with no temporal dimension formed by ones/zeros for (spatial)
        positions to introduce data_pred/np.nans values.

    ensemble_size : int, optional
        If provided, it indicates the number of samples computed by running
        the model ensemble_size times. These are saved as a new dimension in
        the xr.Dataset (member).

    batch_size : int, optional
        If provided the predictions are computed in batches of size
        batch_size. This is useful when facing OOM errors.

    spatial_dims : tuple[str, str], optional
        Names of the spatial dimensions, defaults to ('lat', 'lon').

    Returns
    -------
    xr.Dataset
        The final prediction
    """

    if not ensemble_size:
        ensemble_size = 1

    x_data_arr = trans.xarray_to_numpy(x_data)
    time_pred = x_data['time'].values

    data_pred = _predict(model=model, device=device, x_data=x_data_arr,
                         batch_size=batch_size)

    # Free some memory
    del x_data_arr; gc.collect()

    # If the model return varios tensors, I assume the first one is the
    # one containing the predicted parameters (e.g., elevation case)
    if isinstance(data_pred, types.GeneratorType):
        data_pred = list(data_pred)[0]

    # Get the parameters of the Bernoulli and gamma dists.
    dim_target = data_pred.shape[1] // 3
    p = data_pred[:, :dim_target]
    shape = np.exp(data_pred[:, dim_target:(dim_target*2)])
    scale = np.exp(data_pred[:, (dim_target*2):])

    # Free some memory
    del data_pred; gc.collect()

    # Iterate over ensemble members
    data_pred = []
    for _ in range(ensemble_size):

        # Compute the ocurrence
        p_random = np.random.uniform(0, 1, p.shape)
        ocurrence = (p >= p_random) * 1 

        # Free some memory
        del p_random; gc.collect()

        # Compute the amount
        amount = np.random.gamma(shape=shape, scale=scale)

        # Correct the amount
        epsilon = 1e-06
        threshold = threshold - epsilon
        amount = amount + threshold

        # Combine ocurrence and amount
        data_aux = ocurrence * amount

        # Free some memory
        del ocurrence; del amount
        gc.collect()

        data_aux = _pred_to_xarray(data_pred=data_aux, time_pred=time_pred,
                                   var_target=var_target, mask=mask,
                                   spatial_dims=spatial_dims)

        data_pred.append(data_aux)

    # Free some memory
    del p; del shape; del scale
    gc.collect()

    # Return the Dataset
    if ensemble_size == 1:
        data_final = data_pred[0]
    else:
        data_final = xr.concat(data_pred, dim='member')

    return data_final