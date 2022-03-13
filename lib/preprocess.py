import sys
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import BC_lib
import derived_predictors
import down_scene_ANA
import down_scene_BC
import down_scene_RAW
import down_scene_TF
import down_scene_WG
import down_day
import down_point
import evaluate_methods
import grids
import launch_jobs
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

########################################################################################################################
def preprocess():
    """
    Calls to common and to common_fold_dependent
    """

    # If using kfolds, preprocess common is done only for fold1
    if split_mode not in ['fold2', 'fold3', 'fold4', 'fold5']:
        common()
    common_fold_dependent()


########################################################################################################################
def common():
    """
    Association between high and low resolution grids
    Calculate derived predictors
    Calculate mean and std of all models
    """

    # Association between high resolution and low resolution grids, and regions labels.
    for interp_mode in ('nearest', 'bilinear', ):
        grids.association(interp_mode)

    # Calculate indirect predictors for reanalysis
    if pseudoreality == False:
        derived_predictors.reanalysis_all()

    # Calculates mean and std for all predictors both at reanalysis and models (standardization period)
    for grid in ('pred', 'saf', ):
        for var0 in ('t', 'p', ):
            print(grid, var0, 'get_mean_and_std_allModels')
            standardization.get_mean_and_std_allModels(var0, grid)
    if running_at_HPC == True:
        exit('\n-----------------------------------------------------------------------\n'
             'Wait until get_mean_and_std jobs have finished.\n'
             'Check files at jobs/err/ and solve problems with those model/scene files')


########################################################################################################################
def common_fold_dependent():
    """
    Standardize and split training/testing
    Weather types clustering
    """

    pathOut = pathAux + 'STANDARDIZATION/VAR/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Detect needed_vars
    needed_vars = []
    for var in ('tmax', 'tmin', 'pcp'):
        for method in methods:
            if method['var'] == var and 'var' in method['fields'] and var not in needed_vars:
                needed_vars.append(var)

    # Split train/test var
    for var in needed_vars:
        print('train/test split', var)

        # Reanalysis
        if pseudoreality == False:
            data_calib = read.lres_data(var, 'var')['data']

        # Model with pseudoreality
        elif pseudoreality == True:
            scene = scene_list[0]
            aux = read.lres_data(var, 'var', model=GCM_shortName, scene=scene)
            dates = aux['times']
            data = aux['data']
            time_first, time_last = dates.index(calibration_first_date), dates.index(calibration_last_date) + 1
            data_calib = data[time_first:time_last]

        years = np.array([x.year for x in calibration_dates])
        idates_test = np.array(
            [i for i in range(years.size) if ((years[i] >= testing_years[0]) * (years[i] <= testing_years[1]))])
        idates_train = np.array(
            [i for i in range(years.size) if ((years[i] < testing_years[0]) | (years[i] > testing_years[1]))])
        if idates_test.size > 0:
            testing = data_calib[idates_test]
            np.save(pathOut + var + '_testing', testing)
        else:
            print('testing period is null, testing.npy will not be generated')
        training = data_calib[idates_train]
        np.save(pathOut + var + '_training', training)

    # Standarizes ERA-Int predictors and saves them to files divided by training and testing
    # This is done for the two grids: pred and saf
    for grid in ('saf', 'pred', ):
        for var0 in ('p', 't', ):
            print(grid, var0, 'standardize and split train/test')

            # Reanalysis
            if pseudoreality == False:
                data = read.lres_data(var0, grid)['data']

            # Model with pseudoreality
            else:
                scene = scene_list[0]
                aux = read.lres_data(var0, grid, model=GCM_shortName, scene=scene)
                dates = aux['times']
                data = aux['data']
                time_first, time_last = dates.index(calibration_first_date), dates.index(calibration_last_date) + 1
                data = data[time_first:time_last]

            # Check whether predictors for calibration contain no-data
            if np.where(np.isnan(data))[0].size != 0:
                exit('Predictors for calibration contain no-data and that is not allowed by the program')

            # Standarize pred/saf and splits in training/testing
            data = standardization.standardize(var0, data, 'reanalysis', grid)
            np.save(pathAux+'STANDARDIZATION/'+grid.upper()+'/'+var0+'_reanalysis_standardized', data)
            data_calib = np.load(pathAux+'STANDARDIZATION/'+grid.upper()+'/'+var0+'_reanalysis_standardized.npy')
            years = np.array([x.year for x in calibration_dates])
            idates_test = np.array([i for i in range(years.size) if ((years[i]>=testing_years[0])*(years[i]<=testing_years[1]))])
            idates_train = np.array([i for i in range(years.size) if ((years[i]<testing_years[0])|(years[i]>testing_years[1]))])
            if idates_test.size > 0:
                testing = data_calib[idates_test]
                np.save(pathAux+'STANDARDIZATION/'+grid.upper()+'/' + var0 + '_testing', testing)
            else:
                print('testing period is null, testing.npy will not be generated')
            training = data_calib[idates_train]
            np.save(pathAux+'STANDARDIZATION/'+grid.upper()+'/' + var0 + '_training', training)

    # Fit PCA to SAFs
    ANA_lib.train_PCA()

    # Create elbow curve to help decide number of clusters. See elbow curve and set k_clusters at settings
    if k_clusters is None:
        ANA_lib.set_number_of_weather_types()

    # Get weather types (centroids and labels from clustering)
    ANA_lib.get_weather_types_centroids()

########################################################################################################################
def train_methods():
    """
    Each family of methods needs a different preprocess (training)
    """

    # Go through all methods
    for method_dict in methods:
        var = method_dict['var']
        methodName = method_dict['methodName']
        family = method_dict['family']
        mode = method_dict['mode']
        fields = method_dict['fields']

        print('preprocess train_method', var, methodName)

        if family == 'ANA':
            # Get significant predictors for each grid point and weather type. It is the same for 1/N/PDF, no need to repeat it
            if methodName.split('-')[1] == 'LOC':
                # Serial processing
                if running_at_HPC == False:
                    ANA_lib.correlations(var, methodName, mode)
                    ANA_lib.correlations_collect_chunks(var, methodName, mode)
                # Parallel processing
                elif running_at_HPC == True:
                    launch_jobs.cluster(var, methodName, mode, 'correlations')

            # Calibrate regression coefficients
            if methodName == 'WT-MLR':
                # Serial processing
                if running_at_HPC == False:
                    ANA_lib.coefficients(var, methodName, mode)
                    ANA_lib.coefficients_collect_chunks(var, methodName, mode)
                # Parallel processing
                elif running_at_HPC == True:
                    launch_jobs.cluster(var, methodName, mode, 'coefficients')

        # Calibrate classifiers and regressorss models, and save coefficients
        if family == 'TF':
            # Serial processing
            if running_at_HPC == False:
                TF_lib.train_chunk(var, methodName, family, mode, fields)
                TF_lib.collect_chunks(var, methodName, family)
            # Parallel processing
            elif running_at_HPC == True:
                launch_jobs.training(var, methodName, family, mode, fields)

        if family == 'WG':
            # Serial processing
            if running_at_HPC == False:
                WG_lib.train_chunk(var, methodName, family, mode, fields)
                WG_lib.collect_chunks(var, methodName, family)
            # Parallel processing
            elif running_at_HPC == True:
                launch_jobs.training(var, methodName, family, mode, fields)



