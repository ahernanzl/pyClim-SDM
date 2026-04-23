import os
import sys

import matplotlib.pyplot as plt
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

sys.path.append('../SBCK/')
import SBCK

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
def trend_injection():
    """
    Adjusts trends to GCM RAW-BIL
    """

    # Define list for multiprocessing
    iterable = []

    if apply_ti == True:
        for method_dict in methods:
            targetVar, methodName = method_dict['var'], method_dict['methodName']

            if experiment == 'PROJECTIONS':
                aux = trend_injection_allModels(targetVar, methodName)
                for x in aux:
                    iterable.append(x)

    # Parallel processing
    if runInParallel_multiprocessing == True and experiment == 'PROJECTIONS':
        with Pool(processes=nCPUs_multiprocessing) as pool:
            pool.starmap(trend_injection_oneModel, iterable)

    return  iterable


########################################################################################################################
def trend_injection_allModels(targetVar, methodName):
    """
    Check for methods/models not yet corrected and applie bias correction
    """

    print('postprocess.trend_injection_allModels', targetVar, methodName)


    # Define and create paths
    pathIn = '../results/'+experiment+'/' + targetVar.upper() + '/' + methodName + '/daily_data/'
    pathOut = '../results/'+experiment+ti_sufix + '/' + targetVar.upper() + '/' + methodName + '/daily_data/'

    try:
        os.makedirs(pathOut)
    except:
        pass

    # Define list for multiprocessing
    iterable = []

    # Go through all models
    for model in model_list:

        # Check if all scenes by model have already been corrected
        to_be_corrected = False
        for scene in scene_list:
            if os.path.isfile(pathIn + targetVar + '_' + model + '_' + scene + '.nc') and \
                    not os.path.isfile(pathOut + targetVar + '_' + model + '_' + scene + '.nc'):
                to_be_corrected = True

        if to_be_corrected == True or force_trend_injection == True:

            print(targetVar, methodName, model, 'trend_injection')

            if running_at_HPC == False:
                if runInParallel_multiprocessing == False:
                    # Serial processing
                    trend_injection_oneModel(targetVar, methodName, model)
                else:
                    # Append combination for multiprocessing
                    iterable.append([targetVar, methodName, model])

            # Parallel processing at HPC
            elif running_at_HPC == True:
                while 1:
                    # Check for correctly finished jobs
                    for file in os.listdir('../job/'):
                        if file.endswith(".out"):
                            filename = os.path.join('../job/', file)
                            if subprocess.check_output(['tail', '-1', filename]) == b'end\n':
                                print('-----------------------')
                                print(filename, 'end')
                                os.system('mv ' + filename + ' ../job/out/')
                                os.system('mv ' + filename[:-3] + 'err ../job/out/')

                    # Check number of living jobs
                    os.system('squeue -u ' + user + ' | wc -l > ../log/nJobs.txt')
                    f = open('../log/nJobs.txt', 'r')
                    nJobs = int(f.read()) - 1
                    f.close()
                    time.sleep(1)
                    if nJobs < max_nJobs:
                        print('nJobs', nJobs)
                        break

                # Send new job
                launch_jobs.trendInjection(model, targetVar, methodName)

        else:
            print(targetVar, methodName, model, 'already corrected')

    return iterable

########################################################################################################################
def trend_injection_oneModel(targetVar, methodName, model):
    """
    Apply trend_injection for a specific model.
    """

    print('postprocess.trend_injection_oneModel', model, targetVar, methodName)

    if model != 'reanalysis':
        # Define and create paths
        pathIn = '../results/'+experiment+'/' + targetVar.upper() + '/' + methodName + '/daily_data/'
        pathInRaw = '../results/'+experiment+'/' + targetVar.upper() + '/RAW-BIL/daily_data/'
        pathOut = '../results/'+experiment+ti_sufix + '/' + targetVar.upper() + '/' + methodName + '/daily_data/'

        try:
            os.makedirs(pathOut)
        except:
            pass

        # Go through all scenes
        for scene in scene_list:

            # Check if scene/model exists
            if os.path.isfile(pathIn + targetVar + '_' + model + '_' + scene + '.nc'):
                print(scene)

                if not os.path.isfile(pathIn + targetVar + '_' + model + '_historical.nc'):
                    print('ERROR', pathIn + targetVar + '_' + model + '_historical missing (REQUIRED)')
                    exit()
                if not os.path.isfile(pathIn + targetVar + '_' + model + '_' + scene + '.nc'):
                    print('ERROR', pathIn + targetVar + '_' + model + '_' + scene + ' missing (REQUIRED)')
                    exit()
                if not os.path.isfile(pathInRaw + targetVar + '_' + model + '_historical.nc'):
                    print('ERROR', pathInRaw + targetVar + '_' + model + '_historical missing (REQUIRED)')
                    exit()
                if not os.path.isfile(pathInRaw + targetVar + '_' + model + '_' + scene + '.nc'):
                    print('ERROR', pathInRaw + targetVar + '_' + model + '_' + scene + ' missing (REQUIRED)')
                    exit()
                # Read reference data for RAW-BIL and compute mean climatology
                raw_data = read.netCDF(pathInRaw, targetVar + '_' + model + '_historical.nc', targetVar)['data']

                # Read reference data for methodName and compute mean climatology
                aux = read.netCDF(pathIn, targetVar + '_' + model + '_historical.nc', targetVar)
                dates = aux['times']
                data = aux['data']
                del aux

                ivalid = [i for i in range(len(dates)) if dates[i].year >= reference_years[0] and
                          dates[i].year <= reference_years[1]]
                mean_ref = np.nanmean(data[ivalid], axis=0)
                mean_refRaw = np.nanmean(raw_data[ivalid], axis=0)
                del data, raw_data, dates

                # Read scene data for RAW-BIL
                scene_dataRaw = read.netCDF(pathInRaw, targetVar + '_' + model + '_' + scene + '.nc', targetVar)['data']

                # Read scene data for methodName
                aux = read.netCDF(pathIn, targetVar + '_' + model + '_' + scene + '.nc', targetVar)
                scene_dates = aux['times']
                scene_data = aux['data']
                calendar = aux['calendar']
                ntimes, npoints = scene_data.shape[0], scene_data.shape[1]
                del aux

                # Compute change
                if bc_mode_dict[targetVar] == 'abs':
                    change = scene_data - mean_ref
                    changeRaw = scene_dataRaw - mean_refRaw
                elif bc_mode_dict[targetVar] == 'rel':
                    th = zero_division_th[targetVar]
                    scene_data[scene_data < th] = 0
                    mean_ref[mean_ref < th] = 0
                    change = 100 * (scene_data - mean_ref) / mean_ref
                    change[(mean_ref == 0) * (scene_data == 0)] = 0
                    change[np.isinf(change)] = np.nan

                    scene_dataRaw[scene_dataRaw < th] = 0
                    mean_refRaw[mean_refRaw < th] = 0
                    changeRaw = 100 * (scene_dataRaw - mean_refRaw) / mean_refRaw
                    changeRaw[(mean_refRaw == 0) * (scene_dataRaw == 0)] = 0
                    changeRaw[np.isinf(changeRaw)] = np.nan

                # Compute change on a moving window of 30 timesteps
                change_smoothed = uniform_filter1d(change, size=30, axis=0, mode='nearest')
                change_smoothedRaw = uniform_filter1d(changeRaw, size=30, axis=0, mode='nearest')
                del change, changeRaw

                # Compute spatial average
                change_smoothed_averaged = np.nanmean(change_smoothed, axis=1)
                change_smoothed_averaged = change_smoothed_averaged[:, np.newaxis]
                change_smoothed_averaged = np.repeat(change_smoothed_averaged, npoints, axis=1)
                change_smoothedRaw_averaged = np.nanmean(change_smoothedRaw, axis=1)
                change_smoothedRaw_averaged = change_smoothedRaw_averaged[:, np.newaxis]
                change_smoothedRaw_averaged = np.repeat(change_smoothedRaw_averaged, npoints, axis=1)
                del change_smoothed, change_smoothedRaw


                # Compute and apply delta to be forced
                if bc_mode_dict[targetVar] == 'abs':
                    diff_with_raw = change_smoothedRaw_averaged - change_smoothed_averaged
                    scene_ti = scene_data + diff_with_raw
                elif bc_mode_dict[targetVar] == 'rel':
                    th = zero_division_th[targetVar]
                    change_smoothed_averaged[change_smoothed_averaged < th] = 0
                    change_smoothedRaw_averaged[change_smoothedRaw_averaged < th] = 0
                    diff_with_raw = change_smoothedRaw_averaged  / change_smoothed_averaged
                    diff_with_raw[(change_smoothedRaw_averaged == 0) * (change_smoothed_averaged == 0)] = 0
                    diff_with_raw[np.isinf(diff_with_raw)] = np.nan
                    scene_ti = scene_data * diff_with_raw

                # Set units
                units = predictands_units[targetVar]

                # Save bias corrected scene
                hres_lats = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper()+'_bilinear/hres_lats.npy')
                hres_lons = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper()+'_bilinear/hres_lons.npy')
                write.netCDF(pathOut, targetVar + '_' + model + '_' + scene + '.nc', targetVar, scene_ti, units, hres_lats, hres_lons,
                             scene_dates, calendar, regular_grid=False)


########################################################################################################################
def bias_correction():
    """
    Bias correction of projections and get climdex from bias corrected daily data.
    """

    # Define list for multiprocessing
    iterable = []

    if apply_bc == True:
        if bc_method is None:
            print('Select a bc_method at advanced_settings.')
            exit()
        else:
            for method_dict in methods:
                targetVar, methodName = method_dict['var'], method_dict['methodName']

                if experiment == 'EVALUATION':
                    # Serial processing
                    if running_at_HPC == False:
                        bias_correction_oneModel(targetVar, methodName, 'reanalysis')

                    # Parallel processing
                    elif running_at_HPC == True:
                        launch_jobs.biasCorrection('reanalysis', targetVar, methodName)

                elif experiment == 'PROJECTIONS':
                    aux = bias_correction_allModels(targetVar, methodName)
                    for x in aux:
                        iterable.append(x)

    # Parallel processing
    if runInParallel_multiprocessing == True and experiment == 'PROJECTIONS':
        with Pool(processes=nCPUs_multiprocessing) as pool:
            pool.starmap(bias_correction_oneModel, iterable)

    return  iterable

########################################################################################################################
def bias_correction_renalysis(targetVar, methodName):
    """
    Apply bias correction for a specific model.
    """

    print('postprocess.bias_correction_renalysis', targetVar, methodName, bc_sufix)

    # Define and create paths
    pathIn = '../results/'+experiment+ti_sufix+'/' + targetVar.upper() + '/' + methodName + '/daily_data/'
    pathOut = '../results/'+experiment+ti_sufix+bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/daily_data/'

    try:
        os.makedirs(pathOut)
    except:
        pass
    pathTmp = '../tmp/'+'_'.join((targetVar, methodName, bc_sufix)) + '/'
    if not os.path.exists(pathTmp):
        os.makedirs(pathTmp)

    # Read data
    est_aux = read.netCDF(pathIn, targetVar + '_' + 'reanalysis_TESTING.nc', targetVar)
    est_times = est_aux['times']
    est_data = est_aux['data']
    obs_aux = read.hres_data(targetVar, period='calibration')
    obs_data = obs_aux['data']
    obs_times = obs_aux['times']

    # Select common dates
    all_years = list(np.unique(np.array([x.year for x in est_times])))
    nyears = len(all_years)
    idates_common = [i for i in range(len(obs_times)) if obs_times[i].year in all_years]
    obs_data = obs_data[idates_common]
    obs_times = np.array(obs_times)[idates_common]

    # Display warning if few years
    if nyears < 30:
        print('----------------------------------------')
        print('WARNING: reanalysis testing will be bias corrected, but there are only', nyears, 'yeas avalible')
        print('Bias correction with few years might not perform well')
        print('----------------------------------------')

    # print('obs', obs_data.shape, obs_times[0], obs_times[-1])
    # print('est', est_data.shape, est_times[0], est_times[-1])


    nfolds = 5
    block = nyears // nfolds
    rest = nyears % nfolds
    blocks = [block, block, block, block, block, ]
    for i in range(rest):
        blocks[i] += 1

    fold_years = []
    i = 0
    for ifold in range(nfolds):
        fold_years.append(all_years[i: i+blocks[ifold]])
        i += blocks[ifold]

    # Empty array for results
    scene_bc = np.zeros(est_data.shape)

    # Go through the n folds
    for ifold in range(nfolds):
        print('bias_correcting reanalysis', methodName, bc_sufix, testing_years, ifold+1, '/', nfolds)
        idates_sce = [i for i in range(len(obs_times)) if obs_times[i].year in fold_years[ifold]]
        idates_ref = [i for i in range(len(obs_times)) if obs_times[i].year not in fold_years[ifold]]
        # print(ifold, len(idates_ref), len(idates_sce))
        obs = obs_data[idates_ref]
        mod = est_data[idates_ref]
        sce = est_data[idates_sce]

        # Correct bias for ifold
        scene_bc[idates_sce] = MOS_lib.biasCorrect_as_postprocess(
            (100*obs).astype(predictands_codification[targetVar]['type']),
            mod, sce, targetVar, obs_times[idates_ref], est_times[idates_sce])/100.

    # Set units
    units = predictands_units[targetVar]

    # Save bias corrected scene
    hres_lats = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper()+'_bilinear/hres_lats.npy')
    hres_lons = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper()+'_bilinear/hres_lons.npy')
    write.netCDF(pathOut, targetVar + '_' + 'reanalysis_TESTING.nc', targetVar, scene_bc, units, hres_lats, hres_lons,
                 est_times, reanalysis_calendar, regular_grid=False)


########################################################################################################################
def bias_correction_allModels(targetVar, methodName):
    """
    Check for methods/models not yet corrected and applie bias correction
    """

    print('postprocess.bias_correction_allModels', targetVar, methodName, bc_sufix)


    # Define and create paths
    pathIn = '../results/'+experiment+ti_sufix+'/' + targetVar.upper() + '/' + methodName + '/daily_data/'
    pathOut = '../results/'+experiment+ti_sufix+bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/daily_data/'

    try:
        os.makedirs(pathOut)
    except:
        pass

    # Define list for multiprocessing
    iterable = []

    # Go through all models
    for model in model_list:

        # Check if model historical exists
        if os.path.isfile(pathIn + targetVar + '_' + model + '_historical.nc'):

            # Check if all scenes by model have already been corrected
            to_be_corrected = False
            for scene in scene_list:
                if os.path.isfile(pathIn + targetVar + '_' + model + '_' + scene + '.nc') and \
                        not os.path.isfile(pathOut + targetVar + '_' + model + '_' + scene + '.nc'):
                    to_be_corrected = True

            if to_be_corrected == True or force_bias_correction == True:

                print(targetVar, methodName, model, bc_sufix, 'bias_correction')

                if running_at_HPC == False:
                    if runInParallel_multiprocessing == False:
                        # Serial processing
                        bias_correction_oneModel(targetVar, methodName, model)
                    else:
                        # Append combination for multiprocessing
                        iterable.append([targetVar, methodName, model])

                # Parallel processing at HPC
                elif running_at_HPC == True:
                    while 1:
                        # Check for correctly finished jobs
                        for file in os.listdir('../job/'):
                            if file.endswith(".out"):
                                filename = os.path.join('../job/', file)
                                if subprocess.check_output(['tail', '-1', filename]) == b'end\n':
                                    print('-----------------------')
                                    print(filename, 'end')
                                    os.system('mv ' + filename + ' ../job/out/')
                                    os.system('mv ' + filename[:-3] + 'err ../job/out/')

                        # Check number of living jobs
                        os.system('squeue -u ' + user + ' | wc -l > ../log/nJobs.txt')
                        f = open('../log/nJobs.txt', 'r')
                        nJobs = int(f.read()) - 1
                        f.close()
                        time.sleep(1)
                        if nJobs < max_nJobs:
                            print('nJobs', nJobs)
                            break

                    # Send new job
                    launch_jobs.biasCorrection(model, targetVar, methodName)

            else:
                print(targetVar, methodName, model, bc_sufix, 'already corrected')


    return iterable

########################################################################################################################
def bias_correction_oneModel(targetVar, methodName, model):
    """
    Apply bias correction for a specific model.
    """

    print('postprocess.bias_correction_oneModel', model, targetVar, methodName, bc_sufix)

    if model == 'reanalysis':
        bias_correction_renalysis(targetVar, methodName)
    else:
        # Define and create paths
        pathIn = '../results/'+experiment+ti_sufix+'/' + targetVar.upper() + '/' + methodName + '/daily_data/'
        pathOut = '../results/'+experiment+ti_sufix+bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/daily_data/'

        try:
            os.makedirs(pathOut)
        except:
            pass

        # Read obs and mod in bias correction period
        obs_data = read.hres_data(targetVar, period='biasCorr')['data']
        aux = read.netCDF(pathIn, targetVar + '_' + model + '_historical.nc', targetVar)
        idates = [i for i in range(len(aux['times'])) if
                  aux['times'][i].year in range(biasCorr_years[0], biasCorr_years[1] + 1)]
        mod_data = aux['data'][idates]
        ref_dates = aux['times'][idates]

        # Go through all scenes
        for scene in scene_list:

            # Check if scene/model exists
            if os.path.isfile(pathIn + targetVar + '_' + model + '_' + scene + '.nc'):

                print(scene)

                # Read scene data
                aux = read.netCDF(pathIn, targetVar + '_' + model + '_' + scene + '.nc', targetVar)
                scene_dates = aux['times']
                scene_data = aux['data']
                calendar = aux['calendar']
                del aux

                # Correct bias for scene
                scene_bc = MOS_lib.biasCorrect_as_postprocess(100*obs_data, mod_data, scene_data, targetVar, ref_dates, scene_dates) / 100.

                # Set units
                units = predictands_units[targetVar]

                # Save bias corrected scene
                hres_lats = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper()+'_bilinear/hres_lats.npy')
                hres_lons = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper()+'_bilinear/hres_lons.npy')
                write.netCDF(pathOut, targetVar + '_' + model + '_' + scene + '.nc', targetVar, scene_bc, units, hres_lats, hres_lons,
                             scene_dates, calendar, regular_grid=False)


########################################################################################################################
def get_climdex():
    """
    Calls to get_climdex_for_evaluation (reanalysis) or get_climdex_allModels (models)
    """

    # Define list for multiprocessing
    iterable = []

    # Go through all methods
    for method_dict in methods:
        targetVar = method_dict['var']
        methodName = method_dict['methodName']
        family = method_dict['family']
        mode = method_dict['mode']
        fields = method_dict['fields']

        if experiment == 'EVALUATION':
            if runInParallel_multiprocessing == True:
                iterable.append([targetVar, methodName])
            else:
                get_climdex_for_evaluation(targetVar, methodName)
        else:
            aux = get_climdex_allModels(targetVar, methodName)
            for x in aux:
                iterable.append(x)

    # Parallel processing
    if runInParallel_multiprocessing == True:
        if experiment == 'EVALUATION':
            with Pool(processes=nCPUs_multiprocessing) as pool:
                pool.starmap(get_climdex_for_evaluation, iterable)
        else:
            with Pool(processes=nCPUs_multiprocessing) as pool:
                pool.starmap(get_climdex_oneModel, iterable)

########################################################################################################################
def get_climdex_for_evaluation(targetVar, methodName):
    """
    Calculate climdex for evaluation
    """

    print('get_climdex_for_evaluation', methodName)

    pathOut = '../results/EVALUATION' + bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/climdex/'
    pathIn = '../results/EVALUATION' + bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/daily_data/'

    if os.path.isfile(pathIn+targetVar+'_reanalysis_TESTING.nc'):

        # Read data
        d = postpro_lib.get_data_eval(targetVar, methodName)
        ref, times_ref, obs, est, times_scene = d['ref'], d['times_ref'], d['obs'], d['est'], d['times_scene']
        del d

        # Calculate all climdex
        for (data, filename) in ((obs, 'obs'), (est, 'est')):
            postpro_lib.calculate_all_climdex(pathOut, filename, targetVar, data, times_scene, ref, times_ref, reanalysis_calendar)


########################################################################################################################
def get_climdex_allModels(targetVar, methodName):
    """
    Check if climdex/scene/model already exists, and if not, calculate it,
    """
    print('postprocess.calculate_climdex', targetVar, methodName)

    # Define and create paths
    if apply_bc == False:
        path = '../results/'+experiment+ti_sufix+'/' + targetVar.upper() + '/' + methodName + '/'
    else:
        path = '../results/'+experiment+ti_sufix+bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/'
    pathIn = path + 'daily_data/'
    pathOut = path + 'climdex/'

    # Define list for multiprocessing
    iterable = []

    # Go through all models
    for model in model_list:

        # Check if model climdex has already been calculated
        filenames = []
        for climdex_name in climdex_names[targetVar]:
            for season in season_dict:
                for scene in scene_list:
                    filenames.append(pathOut + '_'.join((targetVar, climdex_name, scene, model, season)) + '.nc')

        climdex_already_calculated = True
        for filename in filenames:
            filename = filename.split('/')[-1]
            scene = filename.split('_')[2]
            model = filename.split('_')[3] + '_' + filename.split('_')[4]
            if ((os.path.isfile(pathIn + targetVar + '_' + model + '_' + scene + '.nc')) and (not os.path.isfile(pathOut+filename))):
                climdex_already_calculated = False

        if climdex_already_calculated == False or force_climdex_calculation == True:

            # Check if model historical exists
            if os.path.isfile(pathIn + targetVar + '_' + model + '_historical.nc'):
                if running_at_HPC == False:
                    if runInParallel_multiprocessing == False:
                        print(targetVar, methodName, model, 'calculating climdex')
                        # Serial processing
                        get_climdex_oneModel(targetVar, methodName, model)
                    else:
                        # Append combination for multiprocessing
                        iterable.append([targetVar, methodName, model])

                # Parallel processing at HPC
                elif running_at_HPC == True:
                    while 1:

                        # Check for correctly finished jobs
                        for file in os.listdir('../job/'):
                            if file.endswith(".out"):
                                filename = os.path.join('../job/', file)
                                if subprocess.check_output(['tail', '-1', filename]) == b'end\n':
                                    print('-----------------------')
                                    print(filename, 'end')
                                    os.system('mv ' + filename + ' ../job/out/')
                                    os.system('mv ' + filename[:-3] + 'err ../job/out/')

                        # Check number of living jobs
                        os.system('squeue -u ' + user + ' | wc -l > ../log/nJobs.txt')
                        f = open('../log/nJobs.txt', 'r')
                        nJobs = int(f.read()) - 1
                        f.close()
                        time.sleep(1)
                        if nJobs < max_nJobs:
                            print('nJobs', nJobs)
                            print(targetVar, methodName, model, 'calculating climdex')
                            break

                    # Send new job
                    launch_jobs.climdex(model, targetVar, methodName)

    return iterable


########################################################################################################################
def get_climdex_oneModel(targetVar, methodName, model):
    """
    Calculates all climdex for a specific model.
    """
    print('get_climdex_oneModel', targetVar, methodName, model)

    # Define and create paths
    if apply_bc == False:
        path = '../results/'+experiment+ti_sufix+'/' + targetVar.upper() + '/' + methodName + '/'
    else:
        path = '../results/'+experiment+ti_sufix+bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/'
    pathIn = path + 'daily_data/'
    pathOut = path + 'climdex/'
    try:
        os.makedirs(pathOut)
    except:
        pass

    # Read reference data (as a scene)
    aux = read.netCDF(pathIn, targetVar + '_' + model + '_historical.nc', targetVar)
    times_ref = aux['times']
    ref = aux['data']
    calendar = aux['calendar']
    idates = [i for i in range(len(times_ref)) if
              times_ref[i].year in range(reference_years[0], reference_years[1] + 1)]
    times_ref = [times_ref[i] for i in idates]
    ref = ref[idates]

    # Read reference climatology (for percCalendar)
    if reference_climatology_from_observations == True:
        aux = read.hres_data(targetVar, period='reference')
        ref_clim = aux['data']
        times_ref_clim = aux['times']
    else:
        ref_clim, times_ref_clim = ref, times_ref

    # Calculate climdex for reference period
    postpro_lib.calculate_all_climdex(pathOut, 'REFERENCE_' + model, targetVar, ref, times_ref, ref_clim, times_ref_clim, calendar)

    # Go through all scenes
    for scene in scene_list:

        # Check if scene/model exists
        if os.path.isfile(pathIn + targetVar + '_' +  model + '_' + scene + '.nc'):
            # Read scene data
            aux = read.netCDF(pathIn, targetVar + '_' + model + '_' + scene + '.nc', targetVar)
            times = aux['times']
            data = aux['data']
            del aux

            # Calculate climdex for scene
            postpro_lib.calculate_all_climdex(pathOut, scene + '_' + model, targetVar, data, times, ref_clim, times_ref_clim, calendar)



########################################################################################################################
def plot_results():
    """
    Divide by subregions and generate graphics for EVALUATION or for PROJECTIONS
    """

    # Establish subregions for each point
    for targetVar in targetVars:
        grids.subregions(targetVar)

    if experiment == 'EVALUATION':
        if activate_plot_annualCycle == True:
            evaluate_methods.annual_cycle()
        evaluate_methods.daily_data()
        evaluate_methods.monthly_data()
        evaluate_methods.climdex()

    if experiment == 'PROJECTIONS':
        postpro_lib.figures_projections()
        # postpro_lib.format_web_AEMET()


########################################################################################################################
def nc2ascii():
    """
    netCDFs to ASCII.
    """

    pathOutBase = '../results/'+experiment+ti_sufix+bc_sufix+'_ASCII/'

    for method_dict in methods:
        targetVar, methodName = method_dict['var'], method_dict['methodName']

        # Set units
        units = predictands_units[targetVar]

        for scene in scene_list:
            if scene == 'TESTING':
                period = str(testing_years[0]) + '-' + str(testing_years[1])
                years = [i for i in range(testing_years[0], testing_years[-1]+1)]
            elif scene == 'historical':
                period = str(historical_years[0]) + '-' + str(historical_years[1])
                years = [i for i in range(historical_years[0], historical_years[-1]+1)]
            else:
                period = str(ssp_years[0]) + '-' + str(ssp_years[1])
                years = [i for i in range(ssp_years[0], ssp_years[-1]+1)]
            for model in model_list:

                # Daily data
                pathIn = '../results/'+experiment+ti_sufix+bc_sufix+'/'+targetVar.upper()+'/'+methodName+'/daily_data/'
                fileName = targetVar+'_'+model+'_'+scene
                if os.path.isfile(pathIn + fileName +'.nc'):
                    nc = read.netCDF(pathIn, targetVar + '_' + model + '_' + scene + '.nc', targetVar)
                    times = nc['times']
                    data = nc['data']
                    data[np.isnan(data)] = fill_value
                    del nc
                    print('writing daily data to ASCCI file for', targetVar, methodName, bc_sufix, scene, model, '...')
                    # id = list(read.hres_metadata(targetVar).index.values)
                    id = list(read.hres_metadata(targetVar)['id'].values)
                    times = np.array([10000 * x.year + 100 * x.month + x.day for x in times])
                    data = np.append(times[:, np.newaxis], data, axis=1)
                    id.insert(0, 'YYYYMMDD')
                    id = [str(x) for x in id]
                    header = ' '.join((model, scene, period, targetVar, '('+units+')',
                                       methodName)) + '\n' + ';'.join(id)
                    pathOut = pathOutBase + targetVar.upper()+'/'+methodName+'/daily_data/'

                    try:
                        os.makedirs(pathOut)
                    except:
                        pass
                    np.savetxt(pathOut+fileName+'.dat', data, fmt=['%.i'] + ['%.2f'] * (len(id) - 1), delimiter=';', header=header)


                # Climdex
                pathIn = '../results/'+experiment+ti_sufix+bc_sufix+'/'+targetVar.upper()+'/'+methodName+'/climdex/'
                for climdex in climdex_names[targetVar]:
                    for season in season_dict:
                        if scene == 'TESTING':
                            fileIn = '_'.join((targetVar,climdex, 'est', season))
                        else:
                            fileIn = '_'.join((targetVar,climdex, scene, model, season))
                        fileOut = '_'.join((targetVar,climdex, scene, model, season))
                        if os.path.isfile(pathIn + fileIn +'.nc'):
                            print('writing climdex to ASCCI file for', targetVar, climdex, methodName, bc_sufix, scene, model, '...')
                            data = read.netCDF(pathIn, fileIn, targetVar+'_'+climdex)['data']
                            # id = list(read.hres_metadata(targetVar).index.values)
                            id = list(read.hres_metadata(targetVar)['id'].values)
                            times = np.array(years)
                            data = np.append(times[:, np.newaxis], data, axis=1)
                            id.insert(0, 'YYYY')
                            id = [str(x) for x in id]
                            header = ' '.join((model, scene, period, season, targetVar, climdex, '('+units+')',
                                               methodName)) + '\n' + ';'.join(id)
                            pathOut = pathOutBase + targetVar.upper()+'/'+methodName+'/climdex/'
                            if not os.path.exists(pathOut):
                                os.makedirs(pathOut)
                            np.savetxt(pathOut+fileOut+'.dat', data, fmt=['%.i'] + ['%.2f'] * (len(id) - 1), delimiter=';', header=header)


########################################################################################################################
def nc1D_to_nc2D():

    """
    netCDFs 1D to netCDFs 2D if observations correspond to a lat/lon regular grid.
    """


    for method_dict in methods:
        targetVar, methodName = method_dict['var'], method_dict['methodName']
        for scene in scene_list:
            if scene == 'TESTING':
                years = [i for i in range(testing_years[0], testing_years[-1]+1)]
            elif scene == 'historical':
                years = [i for i in range(historical_years[0], historical_years[-1]+1)]
            else:
                years = [i for i in range(ssp_years[0], ssp_years[-1]+1)]

            for model in model_list:

                # Daily data
                pathIn = '../results/'+experiment+ti_sufix+bc_sufix+'/'+targetVar.upper()+'/'+methodName+'/daily_data/'
                fileName = targetVar+'_'+model+'_'+scene+'.nc'
                pathOut = '../results/'+experiment+ti_sufix+bc_sufix+'_2D/'+targetVar.upper()+'/'+methodName+'/daily_data/'

                # Check if file exists and has not been processed yet
                if os.path.isfile(pathIn + fileName):
                    if os.path.isfile(pathOut + fileName) and force_2D == False:
                            print(pathOut + fileName, 'already converted to 2D')
                    else:
                        postpro_lib.convert_to_2D(pathIn, pathOut, fileName, targetVar, 'daily_data')
                else:
                    print(pathIn + fileName, 'does not exist')

                # Climdex
                pathIn = '../results/'+experiment+ti_sufix+bc_sufix+'/'+targetVar.upper()+'/'+methodName+'/climdex/'                # fileName = targetVar+'_'+model+'_'+scene+'.nc'
                pathOut = '../results/'+experiment+ti_sufix+bc_sufix+'_2D/'+targetVar.upper()+'/'+methodName+'/climdex/'

                for climdex in climdex_names[targetVar]:
                    for season in season_dict:
                        filenames = []
                        if scene == 'TESTING':
                            filenames.append('_'.join((targetVar,climdex, 'obs', season))+'.nc')
                            filenames.append('_'.join((targetVar,climdex, 'est', season))+'.nc')
                        else:
                            if scene == scene_list[0]:
                                filenames.append('_'.join((targetVar,climdex, 'REFERENCE', model, season))+'.nc')
                            filenames.append('_'.join((targetVar,climdex, scene, model, season))+'.nc')

                        for fileName in filenames:

                            # Check if file exists and has not been processed yet
                            if os.path.isfile(pathIn + fileName):
                                if os.path.isfile(pathOut + fileName) and force_2D == False:
                                    print(pathOut + fileName, 'already converted to 2D')
                                else:
                                    postpro_lib.convert_to_2D(pathIn, pathOut, fileName, targetVar + '_' + climdex, 'climdex', years)
                            else:
                                print(pathIn + fileName, 'does not exist')



########################################################################################################################
if __name__ == "__main__":
    nproc = MPI.COMM_WORLD.Get_size()  # Size of communicator
    iproc = MPI.COMM_WORLD.Get_rank()  # Ranks in communicator
    inode = MPI.Get_processor_name()  # Node where this MPI process runs


    task = sys.argv[1]
    model = sys.argv[2]
    targetVar = sys.argv[3]
    methodName = sys.argv[4]

    if task == 'climdex':
        get_climdex_oneModel(targetVar, methodName, model)
    elif task == 'bias_correction':
        bias_correction_oneModel(targetVar, methodName, model)
    elif task == 'trend_injection':
        trend_injection_oneModel(targetVar, methodName, model)

