"""
This module contains loss functions for training deep learning
downscaling models.

Authors:
    Jose González-Abad
    Alfonso Hernanz
"""

import os
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
import xarray as xr
import scipy.stats
from typing import Union

class MaeLoss(nn.Module):

    """
    Standard Mean Absolute Error (MAE). It is possible to compute
    this metric over a target dataset with nans.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output)
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(MaeLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]

        loss = torch.mean(torch.abs(target - output))
        return loss

class MseLoss(nn.Module):

    """
    Standard Mean Square Error (MSE). It is possible to compute
    this metric over a target dataset with nans.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output)
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(MseLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]

        loss = torch.mean((target - output) ** 2)
        return loss

class NLLGaussianLoss(nn.Module):

    """
    Negative Log-Likelihood of a Gaussian distribution. It is possible to compute
    this metric over a target dataset with nans.

    Notes
    -----
    This loss function needs as input two values, corresponding to the mean and
    the logarithm of the variance. THese must be provided concatenated as an
    unique vector.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output). This vector must be composed
        by the concatenation of the predicted mean and logarithm of the
        variance.
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(NLLGaussianLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        dim_target = target.shape[1]

        mean = output[:, :dim_target]
        log_var = output[:, dim_target:]
        precision = torch.exp(-log_var)

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            mean = mean[~nans_idx]
            log_var = log_var[~nans_idx]
            precision = precision[~nans_idx]
            target = target[~nans_idx]

        loss = torch.mean(0.5 * precision * (target-mean)**2 + 0.5 * log_var)
        return loss

class NLLBerGammaLoss(nn.Module):

    """
    Negative Log-Likelihood of a Bernoulli-gamma distributions. It is possible to compute
    this metric over a target dataset with nans.

    Notes
    -----
    This loss function needs as input three values, corresponding to the p, shape
    and scale parameters. THese must be provided concatenated as an unique vector.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output). This vector must be composed
        by the concatenation of the predicted p, shape and scale.
    """

    def __init__(self, ignore_nans: bool) -> None:
        super(NLLBerGammaLoss, self).__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        dim_target = target.shape[1]

        p = output[:, :dim_target]
        shape = torch.exp(output[:, dim_target:(dim_target*2)])
        scale = torch.exp(output[:, (dim_target*2):])

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            p = p[~nans_idx]
            shape = shape[~nans_idx]
            scale = scale[~nans_idx]
            target = target[~nans_idx]

        bool_rain = torch.greater(target, 0).type(torch.float32)
        epsilon = 0.000001

        noRainCase = (1 - bool_rain) * torch.log(1 - p + epsilon)
        rainCase = bool_rain * (torch.log(p + epsilon) +
                            (shape - 1) * torch.log(target + epsilon) -
                            shape * torch.log(scale + epsilon) -
                            torch.lgamma(shape + epsilon) -
                            target / (scale + epsilon))
        
        loss = -torch.mean(noRainCase + rainCase)
        return loss

class Asym(nn.Module):

    """
    Generalization of the asymmetric loss function tailored for daily precipitation developed in
    Doury et al. 2024. It is possible to compute this metric over a target dataset
    with nans.

    Doury, A., Somot, S. & Gadat, S. On the suitability of a convolutional neural
    network based RCM-emulator for fine spatio-temporal precipitation. Clim Dyn (2024).
    https://doi.org/10.1007/s00382-024-07350-8

    Notes
    -----
    This loss function relies on gamma distribution fitted for each gridpoint in the
    spatial domain. This class provides all the methods require to fit these 
    distributions to the data.

    The level of asymmetry can be adjusted by the pairs asym_weight and cdf_pow.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore nans in the
        target domain.

    asym_weight : float, optional
        Weight for the asymmetric term at the loss function relative to the MAE term.
        Default value: 1 (as in Doury et al., 2024)

    cdf_pow : float, optional
        Pow for the CDF at the asymmetric term of the loss function.
        Default value: 2 (as in Doury et al., 2024)
        Higher values make a bigger differentiation between the weight for high/low percentiles

    asym_path : str
        Path to the folder to save the fitted distributions.

    appendix : str, optional
        String to add to the files generated/loaded for this loss function.
        (e.g., appendix=test1 -> scale_test1.npy). If not provided no appendix
        will be added.

    target : torch.Tensor
        Target/ground-truth data

    output : torch.Tensor
        Predicted data (model's output). This vector must be composed
        by the concatenation of the predicted mean and logarithm of the
        variance.
    """

    def __init__(self, ignore_nans: bool, asym_path: str,
                 asym_weight: float = 1.0, cdf_pow: float = 2.0,
                 appendix: str = None) -> None:
        super(Asym, self).__init__()

        # Ensure that asym_weight and cdf_pow are numeric values
        if not isinstance(asym_weight, (int, float)):
            raise ValueError("'asym_weight' must be a numeric value.")
        if not isinstance(cdf_pow, (int, float)):
            raise ValueError("'cdf_pow' must be a numeric value.")
            
        # Convert to float if needed and check positiveness
        asym_weight = float(asym_weight)
        cdf_pow = float(cdf_pow)
        if asym_weight < 0 or cdf_pow < 0:
            raise ValueError("'asym_weight' and 'cdf_pow' must be positive.")

        self.ignore_nans = ignore_nans
        self.asym_path = asym_path
        self.asym_weight = asym_weight
        self.cdf_pow = cdf_pow
        self.appendix = appendix

    def parameters_exist(self):

        """
        Check for the existence of the gamma distributions
        """

        if self.appendix:
            shape_file_name = f'shape_{self.appendix}.npy'
            scale_file_name = f'scale_{self.appendix}.npy'
            loc_file_name = f'loc_{self.appendix}.npy'
        else:
            shape_file_name = 'shape.npy'
            scale_file_name = 'scale.npy'
            loc_file_name = 'loc.npy'

        shape_exist = os.path.exists(f'{self.asym_path}/{shape_file_name}')
        scale_exist = os.path.exists(f'{self.asym_path}/{scale_file_name}')
        loc_exist = os.path.exists(f'{self.asym_path}/{loc_file_name}')

        return (shape_exist and scale_exist and loc_exist)

    def load_parameters(self):

        """
        Load the gamma distributions from asym_path.
        """

        if self.appendix:
            shape_file_name = f'shape_{self.appendix}.npy'
            scale_file_name = f'scale_{self.appendix}.npy'
            loc_file_name = f'loc_{self.appendix}.npy'
        else:
            shape_file_name = 'shape.npy'
            scale_file_name = 'scale.npy'
            loc_file_name = 'loc.npy'

        self.shape = np.load(f'{self.asym_path}/{shape_file_name}')
        self.scale = np.load(f'{self.asym_path}/{scale_file_name}')
        self.loc = np.load(f'{self.asym_path}/{loc_file_name}')

    def _compute_gamma_parameters(self, x: np.ndarray) -> tuple:

        """
        Fit a gamma distribution to the wet days of the provided
        1D np.ndarray.

        Parameters
        ----------      
        x : np.ndarray
            1D np.ndarray containing the precipitation values across time
            for a specific gridpoint.

        Returns
        -------
        tuple
        The shape, loc and scale parameters of the fitted gamma
        distribution.
        """

        # If nan return nan
        if np.sum(np.isnan(x)) == len(x):
            return np.nan, np.nan, np.nan
        else:
            x = x[~np.isnan(x)] # Remove nans
            x = x[x >= 1] # Filter wet days
            try: # Compute dist.
                fit_shape, fit_loc, fit_scale = scipy.stats.gamma.fit(x)
            except: # If its not possible return nan
                fit_shape, fit_loc, fit_scale = np.nan, np.nan, np.nan 
            return fit_shape, fit_loc, fit_scale

    def compute_parameters(self, data: xr.Dataset, var_target: str):

        """
        Iterate over the xr.Dataset and compute for each spatial gridpoint
        the parameters of a fitted gamma distribution for the wet days.

        Parameters
        ----------      
        data : xr.Dataset
            Dataset containing the variable used as target in the model. It is
            important to provide it in the same way as it will be provided
            as target to the forward() method (e.g., nan-filtered).

        var_target : str
            Target variable.
        """

        # Get years
        gamma_params = []
        group_years = data.groupby('time.year')

        # Iterate over years
        for year, group in group_years:
            print(f'Year: {year}')
            y_year = group[var_target].values
            params_year = np.apply_along_axis(self._compute_gamma_parameters,
                                              axis=0, arr=y_year) # shape, loc, scale
            gamma_params.append(params_year)

        # Compute yearly mean
        gamma_params = np.nanmean(np.stack(gamma_params), axis=0)
        
        self.shape = gamma_params[0, :]
        self.scale = gamma_params[2, :]
        self.loc = gamma_params[1, :]

        # Save the parameters in the asym_path
        if self.appendix:
            shape_file_name = f'shape_{self.appendix}.npy'
            scale_file_name = f'scale_{self.appendix}.npy'
            loc_file_name = f'loc_{self.appendix}.npy'
        else:
            shape_file_name = 'shape.npy'
            scale_file_name = 'scale.npy'
            loc_file_name = 'loc.npy'

        np.save(file=f'{self.asym_path}/{shape_file_name}',
                arr=self.shape)
        np.save(file=f'{self.asym_path}/{scale_file_name}',
                arr=self.scale)
        np.save(file=f'{self.asym_path}/{loc_file_name}',
                arr=self.loc)

    def mask_parameters(self, mask: xr.Dataset):

        """
        Mask the shape, scale and loc parameters. This is required for
        models composed of a final fully-connected layer.

        Parameters
        ----------
        mask : xr.Dataset
            Mask without time dimension containing 1s for non-nan and
            0 for nans.
        """

        mask_var = list(mask.keys())[0]
        mask_dims = list(mask.dims.keys())

        # If gridpoint dimension does not exist, create it
        if 'gridpoint' not in mask_dims:
            mask = mask.stack(gridpoint=('lat', 'lon'))

        # We assume that 1 -> non-nan and 0 -> nan
        mask_values = mask[mask_var].values
        mask_ones = np.where(mask_values == 1)

        self.shape = self.shape[mask_ones]
        self.scale = self.scale[mask_ones]
        self.loc = self.loc[mask_ones]

    def prepare_parameters(self, device: str):

        """
        Move the gamma parameters to device and remove nans. The latter is key,
        as if the gamma has not been fitted to the data (nans), it means that
        there are not (few) rainy days. In this case, we set a weight of one
        for any underestimation (which will be uncommon).

        Parameters
        ----------
        device : str
            Device used to run the training (cuda or cpu)
        """

        self.shape = torch.tensor(self.shape).to(device)
        self.scale = torch.tensor(self.scale).to(device)
        self.loc = torch.tensor(self.loc).to(device)

        epsilon = 0.0000001
        if torch.isnan(self.shape).any():
            self.shape[torch.isnan(self.shape)] = epsilon
        if torch.isnan(self.scale).any():
            self.scale[torch.isnan(self.scale)] = epsilon
        if torch.isnan(self.loc).any():
            self.loc[torch.isnan(self.loc)] = 0

    def compute_cdf(self, data: torch.Tensor) -> torch.Tensor:
    
        """
        Compute the value of the cumulative distribution function (CDF) for
        the data.

        Parameters
        ----------      
        data : torch.Tensor
            Data (from the target dataset) to compute the CDF for.
        """

        # Compute cdfs for Torch
        if isinstance(data, torch.Tensor):
            data = data - self.loc # For scipy, loc corresponds to the mean
            data[data < 0] = 0 # Remove the negative values, which are automatically handled by scipy
            m = td.Gamma(concentration=self.shape,
                         rate=1/self.scale,
                         validate_args=False) # Deactivates the validation of the paremeters (e.g., support)
                                              # In this way the cdf method handles nans
            cdfs = m.cdf(data)

        # Compute cdfs for Numpy
        elif isinstance(data, np.ndarray):
            cdfs = np.empty_like(data)
            cdfs = scipy.stats.gamma.cdf(data,
                                         a=self.shape, scale=self.scale, loc=self.loc)

        else:
            raise ValueError('Unsupported type for the data argument.')

        return cdfs

    def forward(self, target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:

        """
        Compute the loss function for the target and output data
        """

        cdfs = self.compute_cdf(data=target)
        cdfs = torch.nan_to_num(cdfs, nan=0.0)

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            output = output[~nans_idx]
            target = target[~nans_idx]
            cdfs = cdfs[~nans_idx]

        # Compute the loss
        loss_mae = torch.mean(torch.abs(target - output))
        loss_asym = torch.mean((cdfs ** self.cdf_pow) * torch.max(torch.tensor(0.0), target - output))
        loss = loss_mae + self.asym_weight * loss_asym

        return loss
