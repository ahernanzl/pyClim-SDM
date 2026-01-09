import sys

import numpy as np

import deep4downscaling.trans

sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../deep4downscaling/')
import deep.loss as deep_loss
import deep.train as deep_train
import deep.models as deep_models
import deep.pred as deep_pred
import deep.utils as deep_utils

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import derived_predictors
import DL_lib
import GAN_lib
import down_scene_ANA
import down_scene_DL
import down_scene_GAN
import down_scene_MOS
import down_scene_RAW
import down_scene_TF
import down_scene_WG
import down_day
import down_point
import evaluate_methods
import grids
import launch_jobs
import launch_jobs_GPU
import MOS_lib
import plot
import postpro_lib
import postprocess
import precontrol
import preprocess
import process
import read
import transform
import TF_lib
import val_lib
import WG_lib
import write

########################################################################################################################
def get_transformation_parameters_reanalysis(targetVar, fields_and_grid):
    """
    Calculates mean and standard deviation for reanalysis and models and all predictors, using the reference period.
    A transformation using PCAs is used if selected by the user.
    For spred (predictors over the synoptic domain, used for DeepESD), the standardization is made for the whole
    domain, not pointwise, by using the mean and std value for the whole spatial domain
    """

    if fields_and_grid in ('spred-pca', 'saf'):
        perform_pca = True
    else:
        perform_pca = False


    # For pred (local predictors) and saf (synoptic analogy fields), fields and grid (spatial domain) are the same,
    # but for spred (synoptic predictors), fields are predictors and grid is synoptic
    if fields_and_grid == 'pred':
        field, grid = 'pred', 'pred'
    elif fields_and_grid == 'saf':
        field, grid = 'saf', 'saf'
    elif fields_and_grid in ('spred', 'spred-pca',):
        field, grid = 'pred', 'saf'
    else:
        print('wrong fields_and_grid')
        exit()

    pathOut = pathAux+'TRANSFORMATION/'+fields_and_grid.upper()+'/'+targetVar.upper()+'/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Read low resolution data from reanalysis
    dates = calibration_dates
    data = read.lres_data(targetVar, field=field, grid=grid)['data']

    calib_data = 1*data
    ref_data = 1*data
    del data

    # Selects standardization period
    time_first, time_last=dates.index(reference_first_date),dates.index(reference_last_date)+1
    ref_data = ref_data[time_first:time_last]

    # Fill Nans with interpolation
    if force_fillNans_for_local_predictors == True and (np.sum(np.where(np.isnan(ref_data))) != 0):
        ref_data = aux_lib.fillNans_interpolation(ref_data)

    # Calculates mean and standard deviation and saves them to files. For local predictors standardization is made
    # pointwise. Otherwise the total mean and std are used
    if fields_and_grid == 'pred':
        mean = np.nanmean(ref_data, axis=0)
        std = np.nanstd(ref_data, axis=0)
    else:
        total_mean = np.nanmean(ref_data, axis=(0, 2, 3))
        total_std = np.nanstd(ref_data, axis=(0, 2, 3))
        mean = total_mean[:, np.newaxis, np.newaxis]
        mean = np.repeat(mean, saf_nlats, axis=1)
        mean = np.repeat(mean, saf_nlons, axis=2)
        std = total_std[:, np.newaxis, np.newaxis]
        std = np.repeat(std, saf_nlats, axis=1)
        std = np.repeat(std, saf_nlons, axis=2)
        for ipred in range(ref_data.shape[1]):
            mean[ipred] = total_mean[ipred]
            std[ipred] = total_std[ipred]

    np.save(pathOut+'reanalysis_mean', mean)
    np.save(pathOut+'reanalysis_std', std)

    # Fit PCA using the reference period
    if perform_pca == True:

        # Adapt dimensions
        mean = np.expand_dims(mean, axis=0)
        mean = np.repeat(mean, ref_data.shape[0], 0)
        std = np.expand_dims(std, axis=0)
        std = np.repeat(std, ref_data.shape[0], 0)

        # Standardize
        ref_data = (ref_data - mean) / std

        # Synoptic Analogy Fields are weighted
        if fields_and_grid == 'saf':
            W = W_saf[np.newaxis, :]
            W = np.repeat(W, ref_data.shape[0], axis=0)
            W = W.reshape(ref_data.shape)
            ref_data *= W

        # Fit PCA
        if np.sum(np.isnan(ref_data)) != 0:
            ref_data = aux_lib.fillNans_interpolation(ref_data)

        pca = PCA(exp_var_ratio_th).fit(ref_data.reshape(ref_data.shape[0], -1))

        # Save trained pca object
        outfile = open(pathOut + 'reanalysis_pca', 'wb')
        pickle.dump(pca, outfile)
        outfile.close()

     # Transform predictors (standardization plus optional PCA)
    calib_data = transform(targetVar, calib_data, 'reanalysis', fields_and_grid)

    # Save transformed (standardized plus optional PCA) predictors matrix
    np.save(pathOut + 'reanalysis_transformed', calib_data)


########################################################################################################################
def get_transformation_parameters_oneModel(targetVar, fields_and_grid, model):
    """For spred (predictors over the synoptic domain, used for DeepESD), the standardization is made for the whole
    domain, not pointwise, by using the mean and std value for the whole spatial domain
    """
    print('get_transformation_parameters_oneModel', targetVar, fields_and_grid, model)

    if fields_and_grid == 'pred':
        field, grid = 'pred', 'pred'
    elif fields_and_grid == 'saf':
        field, grid = 'saf', 'saf'
    elif fields_and_grid in ('spred', 'spred-pca',):
        field, grid = 'pred', 'saf'
    else:
        print('wrong fields_and_grid')
        exit()

    # Read data and times from model
    aux = read.lres_data(targetVar, field=field, grid=grid, model=model, scene='historical')
    scene_dates = aux['times']

    reference_years = list(set([x.year for x in reference_dates]))
    idates = [i for i in range(len(scene_dates)) if scene_dates[i].year in reference_years]
    data = aux['data']
    data = data[idates]

    # Fill Nans with interpolation
    if force_fillNans_for_local_predictors == True and (np.sum(np.where(np.isnan(data))) != 0):
        data = aux_lib.fillNans_interpolation(data)

    # Calculates mean and standard deviation and saves them to files. For local predictors standardization is made
    #  pointwise. Otherwise the total mean and std are used
    if fields_and_grid == 'pred':
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
    else:
        total_mean = np.nanmean(data, axis=(0, 2, 3))
        total_std = np.nanstd(data, axis=(0, 2, 3))
        mean = total_mean[:, np.newaxis, np.newaxis]
        mean = np.repeat(mean, saf_nlats, axis=1)
        mean = np.repeat(mean, saf_nlons, axis=2)
        std = total_std[:, np.newaxis, np.newaxis]
        std = np.repeat(std, saf_nlats, axis=1)
        std = np.repeat(std, saf_nlons, axis=2)
        for ipred in range(data.shape[1]):
            mean[ipred] = total_mean[ipred]
            std[ipred] = total_std[ipred]


    return {'mean': mean, 'std': std}


########################################################################################################################
def transform(targetVar, data, scene_dates, model, fields_and_grid):
    """Provided the data array, it is standardized (and transformed to PCA, optional) and returned
    Forze_only_standardize controls that the first time the transformation is done, only the standardization is applied.
    The following times, when the PCAs have been fitted, the complete transformation is allowed.
    """

    pathIn=pathAux+'TRANSFORMATION/'+fields_and_grid.upper()+'/'+targetVar.upper()+'/'
    warnings.filterwarnings("ignore")

    # Fill Nans with interpolation
    if force_fillNans_for_local_predictors == True and (np.sum(np.where(np.isnan(data))) != 0):
        data = aux_lib.fillNans_interpolation(data)

    if fields_and_grid in ('spred-pca', 'saf'):
        perform_pca = True
    else:
        perform_pca = False

    if perform_pca == True:
        if np.sum(np.isnan(data)) != 0:
            data = aux_lib.fillNans_interpolation(data)

    # Get mean and std
    if mean_and_std_from_GCM == True and model != 'reanalysis':
        aux = get_transformation_parameters_oneModel(targetVar, fields_and_grid, model)
        mean = aux['mean']
        std = aux['std']
    else:
        mean = np.load(pathIn + 'reanalysis_mean.npy')
        std = np.load(pathIn + 'reanalysis_std.npy')

    # Adapt dimensions
    mean = np.expand_dims(mean, axis=0)
    mean = np.repeat(mean, data.shape[0], 0)
    std = np.expand_dims(std, axis=0)
    std = np.repeat(std, data.shape[0], 0)

    # Bias correct predictors from GCM month by month
    if model != 'reanalysis' and bias_correct_GCM_predictors_seasonal_cycle == True:

        # Define field and grid
        if fields_and_grid in ['pred', 'saf']:
            field, grid = fields_and_grid, None
        elif fields_and_grid in ['spred', 'spred-pca']:
            field, grid = 'pred', 'saf'

        # Load reanalysis predictors reference period
        aux = read.lres_data(targetVar, field, grid=grid, period='reference')
        rea_data = aux['data']
        rea_times = aux['times']
        del aux

        # Load model predictors reference period
        aux = read.lres_data(targetVar, field, grid=grid, model=model, scene='historical', period='reference')
        hist_data = aux['data']
        hist_times = aux['times']
        del aux

        # Create xr.Datasets
        rea_dataset = xr.Dataset(
            {var: (("time", "lat", "lon"), rea_data[:, i, :, :]) for i, var in enumerate(preds_targetVars_dict[targetVar])},
            coords={"time": rea_times, "lat": np.arange(rea_data.shape[2]), "lon": np.arange(rea_data.shape[3])})
        hist_dataset = xr.Dataset(
            {var: (("time", "lat", "lon"), hist_data[:, i, :, :]) for i, var in enumerate(preds_targetVars_dict[targetVar])},
            coords={"time": hist_times, "lat": np.arange(hist_data.shape[2]), "lon": np.arange(hist_data.shape[3])})
        data_dataset = xr.Dataset(
            {var: (("time", "lat", "lon"), data[:, i, :, :]) for i, var in enumerate(preds_targetVars_dict[targetVar])},
            coords={"time": scene_dates, "lat": np.arange(data.shape[2]), "lon": np.arange(data.shape[3])})

        # Apply the correction and convert xr.Dataset to np.array
        data_dataset = deep4downscaling.trans.scaling_delta_correction(
            data_dataset, hist_dataset, rea_dataset)
        data = data_dataset.to_array().values.swapaxes(0, 1)

    # Standardize
    data = (data - mean) / std

    # Synoptic Analogy Fields are weighted
    if fields_and_grid == 'saf':
        W = W_saf[np.newaxis, :]
        W = np.repeat(W, data.shape[0], axis=0)
        W = W.reshape(data.shape)
        data *= W

    if perform_pca == True:
        # Load pca object
        infile = open(pathIn + 'reanalysis_pca', 'rb')
        pca = pickle.load(infile)
        infile.close()
        data = data.reshape(data.shape[0], -1)
        data = pca.transform(data)
        data = data[:, :, np.newaxis, np.newaxis]

    return data

