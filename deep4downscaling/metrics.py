"""
This module contains functions for evaluating the performance of downscaling models
by calculating various metrics.

Author: Jose González-Abad
"""

import xarray as xr
import numpy as np
import xskillscore as xss

def _filter_by_season(data : xr.Dataset, season : str) -> xr.Dataset:

    """
    Internal function to filter a xr.Dataset with respect to the
    provided season.

    Parameters
    ----------
    data : xr.Dataset
        Data to filter

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    if season is None:
        pass
    elif season == 'winter':
        data = data.where(data['time.season'] == 'DJF', drop=True)
    elif season == 'summer':
        data = data.where(data['time.season'] == 'JJA', drop=True)
    elif season == 'spring':
        data = data.where(data['time.season'] == 'MAM', drop=True)
    elif season == 'autumn':
        data = data.where(data['time.season'] == 'SON', drop=True)

    return data

def bias_mean(target: xr.Dataset, pred: xr.Dataset,
              var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the bias of the mean (across time) between the target and pred
    datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = (pred.mean('time') - target.mean('time'))
    return metric

def bias_tnn(target: xr.Dataset, pred: xr.Dataset,
             var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the bias of the annual minimum of daily minimum temperature (TNn)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)

    target = target.groupby('time.year').min('time')
    pred = pred.groupby('time.year').min('time')

    metric = (pred.mean('year') - target.mean('year'))
    return metric

def bias_txx(target: xr.Dataset, pred: xr.Dataset,
             var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the bias of the annual maximum of daily maximum temperature (TXx)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)

    target = target.groupby('time.year').max('time')
    pred = pred.groupby('time.year').max('time')

    metric = (pred.mean('year') - target.mean('year'))
    return metric

def bias_quantile(target: xr.Dataset, pred: xr.Dataset, quantile: float,
                  var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the bias of the specified quantile (across time) between
    the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    quantile : float
        Quantile on which the bias is computed [0,1]

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = (pred.quantile(quantile, 'time') - target.quantile(quantile, 'time'))
    return metric

def mae(target: xr.Dataset, pred: xr.Dataset,
        var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the mean absolute error between
    the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = abs(pred - target).mean('time')
    return metric

def rmse(target: xr.Dataset, pred: xr.Dataset,
         var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the root mean square error between
    the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    metric = ((pred - target) ** 2).mean('time') ** (1/2)
    return metric

def rmse_wet(target: xr.Dataset, pred: xr.Dataset, var_target: str,
             threshold: float=1., season: str=None) -> xr.Dataset:

    """
    Compute the root mean square error between the target and
    pred datasets for the wet days (>=1 mm).

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    threshold : float
        Wet day threshold [0,+inf]

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target = target.where(target[var_target] >= threshold)
    pred = pred.where(pred[var_target] >= threshold)

    metric = ((pred - target) ** 2).mean('time') ** (1/2)
    return metric

def rmse_relative(target: xr.Dataset, pred: xr.Dataset,
                  var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the root mean square error between
    the target and pred datasets relative to the target's
    standard deviation.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_std = target.std('time')

    metric = (((pred - target) ** 2).mean('time') ** (1/2)) / target_std
    return metric

def bias_rel_mean(target: xr.Dataset, pred: xr.Dataset,
                  var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the mean (across time) between
    the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_mean = target.mean('time')

    metric = (pred.mean('time') - target_mean) / target_mean
    metric = metric * 100
    return metric

def bias_rel_quantile(target: xr.Dataset, pred: xr.Dataset, quantile: float,
                      var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the specified quantile (across time)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    quantile : float
        Quantile on which the bias is computed [0,1]

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_quantile = target.quantile(quantile, 'time')

    metric = (pred.quantile(quantile, 'time') - target_quantile) / target_quantile
    metric = metric * 100
    return metric

def bias_rel_R01(target: xr.Dataset, pred: xr.Dataset, var_target: str, 
                 threshold: float=1., season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the R01 index (across time)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    threshold : float
        Wet day threshold [0,+inf]

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    # Compute the nan_mask of the pred
    nan_mask = pred.mean('time')
    nan_mask = (nan_mask - nan_mask) + 1

    # Compute proportion of wet days
    target = (target >= threshold) * 1
    pred = (pred >= threshold) * 1

    # Apply nan_mask, otherwise we get zero
    # for nan gridpoints
    target = target * nan_mask
    pred = pred * nan_mask

    target_mean = target.mean('time')
    metric = (pred.mean('time') - target_mean) / target_mean
    metric = metric * 100
    return metric

def bias_rel_dry_days(target: xr.Dataset, pred: xr.Dataset, var_target: str, 
                      threshold: float=1., season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the proportion of dry days (across time)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    threshold : float
        Wet day threshold [0,+inf]

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    # Compute the nan_mask of the pred
    nan_mask = pred.mean('time')
    nan_mask = (nan_mask - nan_mask) + 1

    # Compute proportion of wet days
    target = (target < threshold) * 1
    pred = (pred < threshold) * 1

    # Apply nan_mask, otherwise we get zero
    # for nan gridpoints
    target = target * nan_mask
    pred = pred * nan_mask

    target_mean = target.mean('time')
    metric = (pred.mean('time') - target_mean) / target_mean
    metric = metric * 100
    return metric

def bias_rel_SDII(target: xr.Dataset, pred: xr.Dataset, var_target: str, 
                  threshold: float=1., season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the SDII index (across time)
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    threshold : float
        Wet day threshold [0,+inf]

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    # Compute the nan_mask of the pred
    nan_mask = pred.mean('time')
    nan_mask = (nan_mask - nan_mask) + 1

    # Filter wet days
    target = target.where(target[var_target] >= threshold)
    pred = pred.where(pred[var_target] >= threshold)

    # Apply nan_mask, otherwise we get zero
    # for nan gridpoints
    target = target * nan_mask
    pred = pred * nan_mask

    target_mean = target.mean('time')
    metric = (pred.mean('time') - target_mean) / target_mean
    metric = metric * 100
    return metric

def bias_rel_rx1day(target: xr.Dataset, pred: xr.Dataset,
                  var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the relative bias of the Rx1day index between
    the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target = target.groupby('time.year').max('time')
    pred = pred.groupby('time.year').max('time')

    target_mean = target.mean('year')

    metric = (pred.mean('year') - target_mean) / target_mean
    metric = metric * 100
    return metric

def ratio_std(target: xr.Dataset, pred: xr.Dataset,
              var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the ratio of standard deviations
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_std = target.std('time')
    pred_std = pred.std('time')

    metric = pred_std / target_std
    return metric

def ratio_interannual_var(target: xr.Dataset, pred: xr.Dataset,
                          var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the ratio of the interannual variatiability
    between the target and pred datasets.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)
    
    target_inter_var = target[var_target].groupby('time.year').mean(dim='time').std('year')
    pred_inter_var = pred[var_target].groupby('time.year').mean(dim='time').std('year')

    metric = pred_inter_var / target_inter_var
    metric = metric.to_dataset()
    return metric

def corr(target: xr.Dataset, pred: xr.Dataset,
         corr_type: str, deseasonal: bool,
         var_target: str, season: str=None) -> xr.Dataset:

    """
    Compute the correlation (pearson or spearman) between the target and pred
    datasets. It is possible to compute it over the deseasonalized
    data.

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data

    pred : xr.Dataset
        Predicted data

    corr_type : str
        Correlation technique to apply (pearson or spearman)

    deseasonal : bool
        Whether to compute the correlation over the 
        deseasonalized data.

    var_target : str
        Target variable.

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    if (deseasonal) and (season is not None):
        raise ValueError('It is not possible to compute the deseasonal correlation for a seasonal subset')

    if corr_type == 'spearman':
        corr_function = xskillscore.spearman_r
    elif corr_type == 'pearson':
        corr_function = xr.corr
    else:
        raise ValueError('Provide a valid value for corr_type (spearman or pearson)')

    if deseasonal:
        target = target[var_target].load()
        target = xr.apply_ufunc(
            lambda x, mean: x - mean, 
            target.groupby('time.month'),
            target.groupby('time.month').mean()
        ).drop('month')

        pred = pred[var_target].load()
        pred = xr.apply_ufunc(
            lambda x, mean: x - mean, 
            pred.groupby('time.month'),
            pred.groupby('time.month').mean()
        ).drop('month')

        metric = corr_function(target, pred, dim='time')

    else:

        metric = corr_function(target[var_target], pred[var_target], dim='time')

    metric = metric.to_dataset(name=var_target)

    return metric

def joint_quantile_exceedance(data_1: xr.Dataset, data_2: xr.Dataset,
                              var_data_1: str, var_data_2: str,
                              quantile: float, season: str=None) -> xr.Dataset:

    """
    Compute the probability of joint exceeding the quantile of
    data_1 and data_2. For instance, the exceedance probability
    of the 90th quantile of tasmax and pr.

    Parameters
    ----------
    data_1, data_2 : xr.Dataset
        Datasets to use to compute the exceedance probability

    var_data_1, var_data_2 : str
        Variable names of data_1 and data_2, respectively

    quantile : float
        Quantile on which the exceedance is computed [0,1]

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    # Compute the nan_mask
    nan_mask = data_1[var_data_1].mean('time')
    nan_mask = (nan_mask - nan_mask) + 1

    data_1 = _filter_by_season(data_1, season)
    data_2 = _filter_by_season(data_2, season)
    
    quantile_data_1 = data_1.quantile(quantile, 'time')
    quantile_data_2 = data_2.quantile(quantile, 'time')

    data_1_exceedance = data_1 > quantile_data_1
    data_2_exceedance = data_2 > quantile_data_2

    metric = data_1_exceedance[var_data_1] * data_2_exceedance[var_data_2] # Bool * Bool 
    metric = metric.to_dataset(name='joint_exceedance')

    metric = metric.mean('time') 
    metric = metric * 100
    metric = metric * nan_mask

    return metric

def bias_joint_quantile_exceedance(target_1: xr.Dataset, target_2: xr.Dataset,
                                   pred_1: xr.Dataset, pred_2: xr.Dataset,
                                   var_data_1: str, var_data_2: str,
                                   quantile: float, season: str=None) -> xr.Dataset:

    """
    Compute the bias in the probability of joint exceeding the quantile
    of the Datasets provided as input. For instance, the bias in the 
    exceedance probabilityof the 90th quantile of tasmax and pr.

    Parameters
    ----------
    target_1, target_2 : xr.Dataset
        Ground truth data

    pred_1, pred_2 : xr.Dataset
        Predicted data

    var_data_1, var_data_2 : str
        Variable names of data_1 and data_2, respectively

    quantile : float
        Quantile on which the exceedance is computed [0,1]

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target_exceedance = joint_quantile_exceedance(data_1=target_1, data_2=target_2,
                                                  var_data_1=var_data_1, var_data_2=var_data_2,
                                                  quantile=quantile, season=season)

    pred_exceedance = joint_quantile_exceedance(data_1=pred_1, data_2=pred_2,
                                                var_data_1=var_data_1, var_data_2=var_data_2,
                                                quantile=quantile, season=season)

    metric = pred_exceedance - target_exceedance

    return metric

def diurnal_temp_range(data_tasmin: xr.Dataset, data_tasmax: xr.Dataset,
                       var_tasmin: str, var_tasmax: str,
                       reduce_mean: bool, season: str=None) -> xr.Dataset:

    """
    Compute the diurnal temperature range (tasmax - tmin).

    Parameters
    ----------
    data_tasmin, data_tasmax : xr.Dataset
        Minimum and maximum temperature Datasets, respectively

    var_tasmin, var_tasmax : str
        Variable names of data_tasmin and data_tasmax, respectively

    reduce_mean : bool
        Whether to apply the mean across the temporal dimension or
        not

    season : str
        Season to filter. If passes as None, no filtering is
        applied

    Returns
    -------
    xr.Dataset
    """

    data_tasmin = _filter_by_season(data_tasmin, season)
    data_tasmax = _filter_by_season(data_tasmax, season)

    metric = data_tasmax[var_tasmax] - data_tasmin[var_tasmin]
    metric = metric.to_dataset(name='dtr')

    if reduce_mean:
        metric = metric.mean('time')

    return metric

def bias_diurnal_temp_range(target_tasmin: xr.Dataset, target_tasmax: xr.Dataset,
                            pred_tasmin: xr.Dataset, pred_tasmax: xr.Dataset,
                            var_tasmin: str, var_tasmax: str,
                            season: str=None) -> xr.Dataset:

    """
    Compute the bias of the dirunal temperature range (tasmax - tmin)
    between the target and pred Datasets.

    Parameters
    ----------
    target_tasmin, target_tasmax : xr.Dataset
        Ground truth

    pred_tasmin, pred_tasmax : xr.Dataset
        Predicted data

    var_tasmin, var_tasmax : str
        Variable names of minimum and maximum temperature
        Datasets, respectively

    season : str
        Season to filter. If passes as None, no filtering is
        applied

    Returns
    -------
    xr.Dataset
    """

    dtr_target = diurnal_temp_range(data_tasmin=target_tasmin, data_tasmax=target_tasmax,
                                    var_tasmin=var_tasmin, var_tasmax=var_tasmax,
                                    season=season, reduce_mean=True)

    dtr_pred = diurnal_temp_range(data_tasmin=pred_tasmin, data_tasmax=pred_tasmax,
                                  var_tasmin=var_tasmin, var_tasmax=var_tasmax,
                                  season=season, reduce_mean=True)

    metric = dtr_pred - dtr_target

    return metric

def corr_compound(data_1: xr.Dataset, data_2: xr.Dataset,
                  var_data_1: str, var_data_2: str,
                  corr_type: str, season: str=None) -> xr.Dataset:

    """
    Compute the correlation (pearson or spearman) between the data_1 and
    data_2 datasets. This function is similar to corr() (without
    deseasonalization) but for datasets with different variable names.

    Parameters
    ----------
    data_1, data_2 : xr.Dataset
        Datasets to compute the correlation from

    var_data_1, var_data_2 : xr.Dataset
        Name of the target variable from data_1 and data_2, respectively

    corr_type : str
        Correlation technique to apply (pearson or spearman)

    season : str
        Season to filter. If passes as None, no filtering is
        applied

    Returns
    -------
    xr.Dataset
    """

    data_1 = _filter_by_season(data_1, season)
    data_2 = _filter_by_season(data_2, season)

    if corr_type == 'spearman':
        corr_function = xskillscore.spearman_r
    elif corr_type == 'pearson':
        corr_function = xr.corr
    else:
        raise ValueError('Provide a valid value for corr_type (spearman or pearson)')

    metric = corr_function(data_1[var_data_1], 
                           data_2[var_data_2], dim='time')
    metric = metric.to_dataset(name='corr')

    return metric

def bias_corr_compound(target_1: xr.Dataset, target_2: xr.Dataset,
                       pred_1: xr.Dataset, pred_2: xr.Dataset,
                       var_data_1: str, var_data_2: str,
                       corr_type: str, season: str=None) -> xr.Dataset:

    """
    Compute the bias of the correlation (pearson or spearman) between
    the target and pred datasets. It is possible to compute this
    correlation for different variables.

    Parameters
    ----------
    target_1, target_2 : xr.Dataset
        Ground truth

    pred_1, pred_2 : xr.Dataset
        Predicted data

    var_data_1, var_data_2 : xr.Dataset
        Name of the target variables
        
    corr_type : str
        Correlation technique to apply (pearson or spearman)

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target_corr = corr_compound(data_1=target_1, data_2=target_2,
                                var_data_1=var_data_1, var_data_2=var_data_2,
                                corr_type=corr_type, season=season)

    pred_corr = corr_compound(data_1=pred_1, data_2=pred_2,
                              var_data_1=var_data_1, var_data_2=var_data_2,
                              corr_type=corr_type, season=season)

    metric = pred_corr - target_corr
    
    return metric

def crps_ensemble(target: xr.Dataset, pred: xr.Dataset,
                  var_target: str, max_pooling: int=None,
                  season: str=None) -> xr.Dataset:

    """
    Compute the Continuous Ranked Probability Score (CRPS) for a set of
    members from an ensemble (pred) with respect to an unique target
    forecast (pred).

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data.

    pred : xr.Dataset
        Predicted data. It needs to contain the member dimension
        with all the different members composing the ensemble

    var_target : str
        Target variable.

    max_pooling : int
        The maximum pooling to perform over the data before computing
        the CRPS (e.g., 4). If left empty no maximum pooling is applied.
        Max pooling is applied to compute the CRPS following
        (Harris, L. et al., 2022)

        Harris, L., McRae, A. T., Chantry, M., Dueben, P. D., & Palmer,
        T. N. (2022). A generative deep learning approach to stochastic
        downscaling of precipitation forecasts. Journal of Advances in
        Modeling Earth Systems, 14(10)

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)

    if max_pooling:
        target = target.coarsen(lat=max_pooling, lon=max_pooling,
                                boundary='trim').max()
        pred = pred.coarsen(lat=max_pooling, lon=max_pooling,
                            boundary='trim').max('member')

    metric = xss.crps_ensemble(observations=target,
                               forecasts=pred,
                               dim='time')

    return metric

def normalized_rank(target: xr.Dataset, pred: xr.Dataset,
                    var_target: str, threshold: int=None,
                    season: str=None) -> xr.Dataset:

    """
    Compute the Normalized Rank as implemented in
    (Harris, L. et al., 2022)

    Parameters
    ----------
    target : xr.Dataset
        Ground truth data. 

    pred : xr.Dataset
        Predicted data. It needs to contain the member dimension
        with all the different members composing the ensemble

    var_target : str
        Target variable.

    max_pooling : int
        The maximum pooling to perform over the data before computing
        the CRPS (e.g., 4). If left empty no maximum pooling is applied.
        Max pooling is applied to compute the CRPS following
        (Harris, L. et al., 2022)

        Harris, L., McRae, A. T., Chantry, M., Dueben, P. D., & Palmer,
        T. N. (2022). A generative deep learning approach to stochastic
        downscaling of precipitation forecasts. Journal of Advances in
        Modeling Earth Systems, 14(10)

    season : str
        Season to filter. If passes as None, no filtering is
        applied.

    Returns
    -------
    xr.Dataset
    """

    target = _filter_by_season(target, season)
    pred = _filter_by_season(pred, season)

    # Compute the nan_mask of the pred
    nan_mask = pred.mean('time').mean('member')
    nan_mask = (nan_mask - nan_mask) + 1

    if 'member' not in list(pred.dims):
        raise ValueError('Please provide a pred with a member dimension')

    # Compute metric
    metric = (pred <= target)

    if threshold:
        mask_threshold = (target > threshold)
        mask_threshold = mask_threshold.expand_dims(member=pred.dims['member'])
        metric = metric.where(mask_threshold)

    metric = metric.mean('member')
    metric = metric * nan_mask

    return metric

