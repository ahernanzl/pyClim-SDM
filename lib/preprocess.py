import sys
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
import transform
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
    if experiment != 'PRECONTROL':
        common_fold_dependent()


########################################################################################################################
def common():
    """
    Association between high and low resolution grids
    Standardize reanalysis
    """

    # Association between high resolution and low resolution grids, and regions labels.
    for targetVar in targetVars:
        for interp_mode in ('nearest', 'bilinear', ):
            grids.association(interp_mode, targetVar)

    # Standardize reanalysis (pred and saf, standardization period)
    for fields_and_grid in (
            'saf',
            'pred',
            'spred',
        ):
        for targetVar in targetVars:
            print(fields_and_grid, targetVar, 'get_transformation_parameters_reanalysis')
            transform.get_transformation_parameters_reanalysis(targetVar, fields_and_grid)


########################################################################################################################
def common_fold_dependent():
    """
    Split training/testing
    Weather types clustering
    """

    # Define and create output path
    pathOut = pathAux + 'TRANSFORMATION/VAR/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Split train/test targetVar
    for targetVar in targetVars:
        print('train/test split', targetVar)

        # Reanalysis
        data_calib = read.lres_data(targetVar, 'var')['data']

        # Get days for training and testing and saves split data to files
        years = np.array([x.year for x in calibration_dates])
        idates_test = np.array(
            [i for i in range(years.size) if ((years[i] >= testing_years[0]) * (years[i] <= testing_years[1]))])
        idates_train = np.array(
            [i for i in range(years.size) if ((years[i] < testing_years[0]) | (years[i] > testing_years[1]))])
        if idates_test.size > 0:
            testing = data_calib[idates_test]
            np.save(pathOut + targetVar + '_testing', testing)
        else:
            print('testing period is null, testing.npy will not be generated')
        if idates_train.size > 0:
            training = data_calib[idates_train]
            np.save(pathOut + targetVar + '_training', training)
        else:
            print('training period is null, testing.npy will not be generated')

    # Standarizes reanalysis predictors and saves them to files divided by training and testing
    # This is done for the two grids: pred and saf
    for fields_and_grid in (
            'pred',
            'spred',
            'saf',
        ):
        for targetVar in targetVars:
            print(fields_and_grid, targetVar, ' and split train/test')

            # Load standardized data and splits in training/testing
            data_calib = np.load(pathAux+'TRANSFORMATION/'+fields_and_grid.upper()+'/'+targetVar.upper()+'/reanalysis_transformed.npy')
            if np.where(np.isnan(data_calib))[0].size != 0:
                exit('Predictors for calibration contain no-data and that is not allowed by the program')
            years = np.array([x.year for x in calibration_dates])
            idates_test = np.array([i for i in range(years.size) if ((years[i]>=testing_years[0])*(years[i]<=testing_years[1]))])
            idates_train = np.array([i for i in range(years.size) if ((years[i]<testing_years[0])|(years[i]>testing_years[1]))])
            if idates_test.size > 0:
                testing = data_calib[idates_test]
                np.save(pathAux+'TRANSFORMATION/'+fields_and_grid.upper()+'/' + targetVar + '_testing', testing)
            else:
                print('testing period is null, testing.npy will not be generated')
            if idates_train.size > 0:
                training = data_calib[idates_train]
                np.save(pathAux+'TRANSFORMATION/'+fields_and_grid.upper()+'/' + targetVar + '_training', training)
            else:
                print('training period is null, testing.npy will not be generated')

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
    
    iterable_TF = []

    # Go through all methods
    for method_dict in methods:
        targetVar = method_dict['var']
        methodName = method_dict['methodName']
        family = method_dict['family']
        mode = method_dict['mode']
        fields = method_dict['fields']

        print('preprocess train_method', targetVar, methodName)

        if family == 'ANA':
            # Get significant predictors for each grid point and weather type. It is the same for 1/N/PDF, no need to repeat it
            if methodName.split('-')[1] == 'LOC':
                # Serial processing
                if running_at_HPC == False:
                    ANA_lib.correlations(targetVar, methodName, mode)
                    ANA_lib.correlations_collect_chunks(targetVar, methodName, mode)
                # Parallel processing
                elif running_at_HPC == True:
                    launch_jobs.cluster(targetVar, methodName, mode, 'correlations')

            # Calibrate regression coefficients
            if methodName == 'MLR-WT':
                # Serial processing
                if running_at_HPC == False:
                    ANA_lib.coefficients(targetVar, methodName, mode)
                    ANA_lib.coefficients_collect_chunks(targetVar, methodName, mode)
                # Parallel processing
                elif running_at_HPC == True:
                    launch_jobs.cluster(targetVar, methodName, mode, 'coefficients')

        # Calibrate classifiers and regressorss models, and save coefficients
        if family == 'TF':
            # Serial processing
            if running_at_HPC == False:
                if runInParallel_multiprocessing == False:
                    TF_lib.train_chunk(targetVar, methodName, family, mode, fields)
                else:
                    iterable_TF.append([targetVar, methodName, family, mode, fields])

            # Parallel processing
            elif running_at_HPC == True:
                launch_jobs.training(targetVar, methodName, family, mode, fields)

        if family == 'WG':
            # Serial processing
            if running_at_HPC == False:
                WG_lib.train_chunk(targetVar, methodName, family, mode, fields)
                WG_lib.collect_chunks(targetVar, methodName, family)
            # Parallel processing
            elif running_at_HPC == True:
                launch_jobs.training(targetVar, methodName, family, mode, fields)

    # Parallel processing of TF
    if runInParallel_multiprocessing == True:
        for x in iterable_TF:
            iterable = []
            for iproc in range(nCPUs_multiprocessing):
                iterable.append([x[0], x[1], x[2], x[3], x[4], iproc, nCPUs_multiprocessing])
            with Pool(processes=nCPUs_multiprocessing) as pool:
                pool.starmap(TF_lib.train_chunk, iterable)
