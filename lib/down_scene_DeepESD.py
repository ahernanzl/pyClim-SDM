import sys

import numpy as np

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
import DeepESD_lib
import down_scene_ANA
import down_scene_DeepESD
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
def downscale(targetVar, methodName, family, mode, fields, scene, model):
    """
    This function goes through all days.
    It previously divides a scene in nproc chunks and processes the chunk number iproc in parallel.
    The result is saved as npy file (each chunk is one file).
    """

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Define path
    pathOut = '../results/'+experiment+'/'+targetVar.upper()+'/'+methodName+'/daily_data/'

    # Parent process reads all data, broadcasts to the other processes and creates paths for results
    print(scene, model, targetVar, methodName)
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Set scene dates and predictors
    if scene == 'TESTING':
        scene_dates = testing_dates
        scene_ndates = len(scene_dates)
        X_test = np.load(pathAux+'TRANSFORMATION/SPRED/'+targetVar+'_testing.npy')
        X_test = X_test.astype('float32')
    else:
        if scene == 'historical':
            years = historical_years
        else:
            years = ssp_years

        # Read dates (can be different for different calendars)
        aux = read.lres_data(targetVar, 'var', model=model, scene=scene)
        scene_dates = aux['times']
        idates = [i for i in range(len(scene_dates)) if scene_dates[i].year >= years[0] and scene_dates[i].year <= years[1]]
        scene_dates = list(np.array(scene_dates)[idates])
        scene_ndates = len(scene_dates)
        X_test = read.lres_data(targetVar, 'pred', grid='saf', model=model, scene=scene)['data'][idates]
        X_test = transform.transform(targetVar, X_test, model, 'spred')
        X_test = X_test.astype('float32')
        del aux

    # Remove days with Nans
    invalid_X = list(set(np.where(np.isnan(X_test))[0]))
    valid = [i for i in range(X_test.shape[0]) if i not in invalid_X]
    X_test = X_test[valid]

    # Create Dataset
    X_test_ds = xr.Dataset(
        {var: (("time", "lat", "lon"), X_test[:, i, :, :]) for i, var in enumerate(preds_targetVars_dict[targetVar])},
        coords={
            "time": np.arange(X_test.shape[0]),
            "lat": np.arange(X_test.shape[2]),
            "lon": np.arange(X_test.shape[3])
        }
    )

    y_shape = (X_test.shape[0], hres_npoints[targetVar])

    # Load trained model
    model_name = 'DeepESD-' + targetVar
    pathModel = pathAux + 'TRAINED_MODELS/' + targetVar.upper() + '/' + methodName + '/'
    if targetVar == 'pr':
        model_deep = deep_models.DeepESDpr(x_shape=X_test.shape, y_shape=y_shape, filters_last_conv=1, stochastic=False)
    else:
        model_deep = deep_models.DeepESDtas(x_shape=X_test.shape, y_shape=y_shape, filters_last_conv=1, stochastic=False)

    model_deep.load_state_dict(torch.load(f'{pathModel}/{model_name}.pt'))

    # Create template
    template = xr.Dataset(
        {targetVar: (["point"], np.ones(hres_npoints[targetVar], dtype=np.int8))},
        coords={"point": np.arange(hres_npoints[targetVar])},
    )

    # Compute predictions
    y_pred = deep_pred.compute_preds_standard(x_data=X_test_ds, model=model_deep, device=device, var_target=targetVar,
                                              template=template, batch_size=16)

    # Corvert to numpy array
    y_pred = y_pred[targetVar].values

    est = np.zeros((scene_ndates, hres_npoints[targetVar]))
    est[:] = np.nan
    est[valid] = y_pred

    # Gets scene dates
    if scene == 'TESTING':
        scene_dates = testing_dates
        calendar = reanalysis_calendar
    else:
        if scene == 'historical':
            periodFilename = historicalPeriodFilename
        else:
            periodFilename = sspPeriodFilename
        # Read dates (can be different for different calendars)
        scene_dates, calendar, datesDefined = aux_lib.retrieve_model_dates(targetVar, scene, model)
        scene_dates = np.ndarray.tolist(scene_dates)

    # Save results
    hres_lats = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper() + '_bilinear/hres_lats.npy')
    hres_lons = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper() + '_bilinear/hres_lons.npy')

    if not os.path.exists(pathOut):
        os.makedirs(pathOut)


    # Set units
    units = predictands_units[targetVar]
    if units is None:
        units = ''

    if split_mode[:4] == 'fold':
        fold_sufix = '_' + split_mode
    else:
        fold_sufix = ''

    # Special values are set to nan
    warnings.filterwarnings("ignore", message="invalid value encountered in less")
    print('-------------------------------------------------------------------------')
    print('results contain', 100*int(np.where(np.isnan(est))[0].size/est.size), '% of nans')
    print('-------------------------------------------------------------------------')
    if targetVar == 'huss':
        print('huss modification /1000...')
        est /= 1000

    # Force to theoretical range
    minAllowed, maxAllowed = predictands_range[targetVar]['min'], predictands_range[targetVar]['max']
    if  minAllowed is not None:
        est[est < minAllowed] = minAllowed
    if  maxAllowed is not None:
        est[est > maxAllowed] = maxAllowed

    # Save data to netCDF file
    write.netCDF(pathOut, targetVar+'_'+model+'_'+scene+fold_sufix+'.nc', targetVar, est, units, hres_lats, hres_lons,
                 scene_dates, calendar, regular_grid=False)

    # If using k-folds, join them
    if split_mode == 'fold5':
        aux_lib.join_kfolds(targetVar, methodName, family, mode, fields, scene, model, units, hres_lats, hres_lons)


########################################################################################################################

if __name__=="__main__":

    targetVar = sys.argv[1]
    methodName = sys.argv[2]
    family = sys.argv[3]
    mode = sys.argv[4]
    fields = sys.argv[5]
    scene = sys.argv[6]
    model = sys.argv[7]

    downscale(targetVar, methodName, family, mode, fields, scene, model)