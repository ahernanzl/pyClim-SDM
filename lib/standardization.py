import sys

import numpy as np

sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import derived_predictors
import down_scene_ANA
import down_scene_MOS
import down_scene_RAW
import down_scene_TF
import down_scene_WG
import down_day
import down_point
import evaluate_methods
import grids
import launch_jobs
import MOS_lib
import plot
import postpro_lib
import postprocess
import precontrol
import preprocess
import process
import read
import standardization
import TF_lib
import val_lib
import WG_lib
import write

def get_mean_and_std_reanalysis(targetVar, fields_and_grid):
    """
    Calculates mean and standard deviation for reanalysis and models and all predictors.
    The time period used is the one with data from both reanalysis and models historical: 1980-2005.
    """

    # For pred (local predictors) and saf (synoptic analogy fields), fields and grid (spatial domain) are the same,
    # but for spred (synoptic predictors), fields are predictors and grid is synoptic
    if fields_and_grid == 'pred':
        field, grid = 'pred', 'pred'
    elif fields_and_grid == 'saf':
        field, grid = 'saf', 'saf'
    elif fields_and_grid == 'spred':
        field, grid = 'pred', 'saf'
    else:
        print('wrong fields_and_grid')
        exit()

    pathOut = pathAux+'STANDARDIZATION/'+fields_and_grid.upper()+'/'+targetVar.upper()+'/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Read low resolution data from reanalysis
    if pseudoreality == True:
        aux = read.lres_data(targetVar, field=field, grid=grid, model=GCM_shortName, scene=scene)
        dates = aux['times']
        data = aux['data']
    else:
        dates = calibration_dates
        data = read.lres_data(targetVar, field=field, grid=grid)['data']

    calib_data = 1*data
    ref_data = 1*data
    del data

    # Selects standardization period
    time_first, time_last=dates.index(reference_first_date),dates.index(reference_last_date)+1
    ref_data = ref_data[time_first:time_last]

    # Calculates mean and standard deviation and saves them to files.
    mean = np.nanmean(ref_data, axis=0)
    std = np.nanstd(ref_data, axis=0)
    np.save(pathOut+'reanalysis_mean', mean)
    np.save(pathOut+'reanalysis_std', std)

    # Save standardized predictors matrix
    calib_data = standardize(targetVar, calib_data, 'reanalysis', fields_and_grid)
    np.save(pathAux + 'STANDARDIZATION/' + fields_and_grid.upper() + '/' + targetVar + '_reanalysis_standardized', calib_data)


########################################################################################################################
def get_mean_and_std_oneModel(targetVar, fields_and_grid, model):

    print('get_mean_and_std_oneModel', targetVar, fields_and_grid, model)

    if fields_and_grid == 'pred':
        field, grid = 'pred', 'pred'
    elif fields_and_grid == 'saf':
        field, grid = 'saf', 'saf'
    elif fields_and_grid == 'spred':
        field, grid = 'pred', 'saf'
    else:
        print('wrong fields_and_grid')
        exit()

    # Read data and times from model
    aux = read.lres_data(targetVar, field=field, grid=grid, model=model, scene='historical')
    scene_dates = aux['times']

    # Read calendar
    for pred in preds_dict[targetVar]:
        if len(pred) > 4 and pred[-4:] in all_levels:
            level = pred[-4:]
        elif len(pred) > 3 and pred[-3:] in all_levels:
            level = pred[-3:]
        else:
            level = None
        try:
            calendar = read.one_direct_predictor(targetVar, level=level, grid='ext', model=model, scene='historical')['calendar']
            break
        except:
            pass

    if calendar in ('360', '360_day'):
        time_first, time_last = scene_dates.index(reference_first_date), scene_dates.index(reference_dates[-2])
    else:
        time_first, time_last = scene_dates.index(reference_first_date), scene_dates.index(reference_last_date) + 1
    data = aux['data']
    data = data[time_first:time_last]

    # Calculates mean and standard deviation and saves them to files
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)

    return {'mean': mean, 'std': std}


########################################################################################################################
def standardize(targetVar, data, model, fields_and_grid):
    """Provided the data array, it is standardized and returned """

    pathIn=pathAux+'STANDARDIZATION/'+fields_and_grid.upper()+'/'+targetVar.upper()+'/'
    warnings.filterwarnings("ignore")

    # Get mean and std
    if mean_and_std_from_GCM == True and model != 'reanalysis':
        aux = get_mean_and_std_oneModel(targetVar, fields_and_grid, model)
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

    # Standardize
    data = (data - mean) / std
    return data
