import os
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
import standardization
import TF_lib
import val_lib
import WG_lib
import write

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
    pathIn = '../results/'+experiment+'/' + targetVar.upper() + '/' + methodName + '/daily_data/'
    pathOut = '../results/'+experiment+bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/daily_data/'

    try:
        os.makedirs(pathOut)
    except:
        pass
    pathTmp = '../tmp/'+'_'.join((targetVar, methodName, bc_sufix)) + '/'
    if not os.path.exists(pathTmp):
        os.makedirs(pathTmp)

    # Read data
    est_aux = read.netCDF(pathIn, 'reanalysis_TESTING.nc', targetVar)
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

    fold_years = []
    fold_years.append(all_years[0: int(nyears/2)])
    fold_years.append([x for x in all_years if x not in fold_years[0]])

    # Empty array for results
    scene_bc = np.zeros(est_data.shape)

    # Go through the two folds
    for ifold in [0, 1]:
        print('bias_correcting reanalysis', methodName, bc_sufix, testing_years, ifold+1, '/ 2')
        idates_sce = [i for i in range(len(obs_times)) if obs_times[i].year in fold_years[ifold]]
        idates_ref = [i for i in range(len(obs_times)) if obs_times[i].year not in fold_years[ifold]]
        # print(ifold, len(idates_ref), len(idates_sce))
        obs = obs_data[idates_ref]
        mod = est_data[idates_ref]
        sce = est_data[idates_sce]

        # Correct bias for ifold
        scene_bc[idates_sce] = MOS_lib.biasCorrect_as_postprocess(obs, mod, sce, targetVar, obs_times[idates_ref], est_times[idates_sce])

    # Set units
    units = predictands_units[targetVar]

    # Save bias corrected scene
    hres_lats = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper()+'_bilinear/hres_lats.npy')
    hres_lons = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper()+'_bilinear/hres_lons.npy')
    write.netCDF(pathOut, 'reanalysis_TESTING.nc', targetVar, scene_bc, units, hres_lats, hres_lons,
                 est_times, regular_grid=False)


########################################################################################################################
def bias_correction_allModels(targetVar, methodName):
    """
    Check for methods/models not yet corrected and applie bias correction
    """

    print('postprocess.bias_correction_allModels', targetVar, methodName, bc_sufix)


    # Define and create paths
    pathIn = '../results/'+experiment+'/' + targetVar.upper() + '/' + methodName + '/daily_data/'
    pathOut = '../results/'+experiment+bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/daily_data/'

    try:
        os.makedirs(pathOut)
    except:
        pass

    # Define list for multiprocessing
    iterable = []

    # Go through all models
    for model in model_list:

        # Check if model historical exists
        if os.path.isfile(pathIn + model + '_historical.nc'):

            # Check if all scenes by model have already been corrected
            to_be_corrected = False
            for scene in scene_list:
                if os.path.isfile(pathIn + model + '_' + scene + '.nc') and \
                        not os.path.isfile(pathOut + model + '_' + scene + '.nc'):
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
        pathIn = '../results/'+experiment+'/' + targetVar.upper() + '/' + methodName + '/daily_data/'
        pathOut = '../results/'+experiment+bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/daily_data/'

        try:
            os.makedirs(pathOut)
        except:
            pass

        # Read obs and mod in bias correction period
        obs_data = read.hres_data(targetVar, period='biasCorr')['data']
        aux = read.netCDF(pathIn, model + '_historical.nc', targetVar)
        idates = [i for i in range(len(aux['times'])) if
                  aux['times'][i].year in range(biasCorr_years[0], biasCorr_years[1] + 1)]
        mod_data = aux['data'][idates]

        # Go through all scenes
        for scene in scene_list:

            # Check if scene/model exists
            if os.path.isfile(pathIn + model + '_' + scene + '.nc'):

                print(scene)

                # Read scene data
                aux = read.netCDF(pathIn, model + '_' + scene + '.nc', targetVar)
                scene_dates = aux['times']
                scene_data = aux['data']
                del aux

                # Correct bias for scene
                scene_bc = MOS_lib.biasCorrect_as_postprocess(obs_data, mod_data, scene_data, targetVar, biasCorr_dates, scene_dates)

                # Set units
                units = predictands_units[targetVar]

                # Save bias corrected scene
                hres_lats = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper()+'_bilinear/hres_lats.npy')
                hres_lons = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper()+'_bilinear/hres_lons.npy')
                write.netCDF(pathOut, model + '_' + scene + '.nc', targetVar, scene_bc, units, hres_lats, hres_lons,
                             scene_dates, regular_grid=False)


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
            get_climdex_for_evaluation(targetVar, methodName)
        else:
            aux = get_climdex_allModels(targetVar, methodName)
            for x in aux:
                iterable.append(x)

    # Parallel processing
    if runInParallel_multiprocessing == True and experiment != 'EVALUATION':
        with Pool(processes=nCPUs_multiprocessing) as pool:
            pool.starmap(get_climdex_oneModel, iterable)

########################################################################################################################
def get_climdex_for_evaluation(targetVar, methodName):
    """
    Calculate climdex for evaluation
    """

    print('get_climdex_for_evaluation', methodName)

    if apply_bc == False:
        pathOut = '../results/EVALUATION/' + targetVar.upper() + '/' + methodName + '/climdex/'
        pathIn = '../results/EVALUATION/' + targetVar.upper() + '/' + methodName + '/daily_data/'
    else:
        pathOut = '../results/EVALUATION' + bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/climdex/'
        pathIn = '../results/EVALUATION' + bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/daily_data/'

    if os.path.isfile(pathIn+'reanalysis_TESTING.nc'):

        # Read data
        d = postpro_lib.get_data_eval(targetVar, methodName)
        ref, times_ref, obs, est, times_scene = d['ref'], d['times_ref'], d['obs'], d['est'], d['times_scene']
        del d

        # Calculate all climdex
        for (data, filename) in ((obs, 'obs'), (est, 'est')):
            postpro_lib.calculate_all_climdex(pathOut, filename, targetVar, data, times_scene, ref, times_ref)


########################################################################################################################
def get_climdex_allModels(targetVar, methodName):
    """
    Check if climdex/scene/model already exists, and if not, calculate it,
    """
    print('postprocess.calculate_climdex', targetVar, methodName)

    # Define and create paths
    if pseudoreality == True:
        path = '../results/'+experiment+'/pseudoreality_'+GCM_longName+'_'+RCM+'/'+targetVar.upper()+'/'+methodName+'/'
    else:
        if apply_bc == False:
            path = '../results/'+experiment+'/' + targetVar.upper() + '/' + methodName + '/'
        else:
            path = '../results/'+experiment+bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/'
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
                    filenames.append(pathOut + '_'.join((climdex_name, scene, model, season)) + '.nc')

        climdex_already_calculated = True
        for filename in filenames:
            filename = filename.split('/')[-1]
            scene = filename.split('_')[1]
            model = filename.split('_')[2] + '_' + filename.split('_')[3]
            if ((os.path.isfile(pathIn + model + '_' + scene + '.nc')) and (not os.path.isfile(pathOut+filename))):
                climdex_already_calculated = False
        if climdex_already_calculated == False or force_climdex_calculation == True:

            # Check if model historical exists
            if os.path.isfile(pathIn + model + '_historical.nc'):
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
    if pseudoreality == True:
        path = '../results/'+experiment+'/pseudoreality_'+GCM_longName+'_'+RCM+'/'+targetVar.upper()+'/'+methodName+'/'
    else:
        if apply_bc == False:
            path = '../results/'+experiment+'/' + targetVar.upper() + '/' + methodName + '/'
        else:
            path = '../results/'+experiment+bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/'
    pathIn = path + 'daily_data/'
    pathOut = path + 'climdex/'
    try:
        os.makedirs(pathOut)
    except:
        pass

    # Read reference data (as a scene)
    aux = read.netCDF(pathIn, model + '_historical.nc', targetVar)
    times_ref = aux['times']
    ref = aux['data']
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
    postpro_lib.calculate_all_climdex(pathOut, 'REFERENCE_' + model, targetVar, ref, times_ref, ref_clim, times_ref_clim)

    # Go through all scenes
    for scene in scene_list:

        # Check if scene/model exists
        if os.path.isfile(pathIn + model + '_' + scene + '.nc'):
            # Read scene data
            aux = read.netCDF(pathIn, model + '_' + scene + '.nc', targetVar)
            times = aux['times']
            data = aux['data']
            del aux

            # Calculate climdex for scene
            postpro_lib.calculate_all_climdex(pathOut, scene + '_' + model, targetVar, data, times, ref_clim, times_ref_clim)



########################################################################################################################
def plot_results():
    """
    Divide by subregions and generate graphics for EVALUATION or for PROJECTIONS
    """

    # Establish subregions for each point
    for targetVar in targetVars:
        grids.subregions(targetVar)

    if experiment == 'EVALUATION':
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

    pathOutBase = '../results/'+experiment+bc_sufix+'_ASCII/'

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
                pathIn = '../results/'+experiment+bc_sufix+'/'+targetVar.upper()+'/'+methodName+'/daily_data/'
                fileName = model+'_'+scene
                if os.path.isfile(pathIn + fileName +'.nc'):
                    nc = read.netCDF(pathIn, model + '_' + scene + '.nc', targetVar)
                    times = nc['times']
                    data = nc['data']
                    data[np.isnan(data)] = -999
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
                pathIn = '../results/'+experiment+bc_sufix+'/'+targetVar.upper()+'/'+methodName+'/climdex/'
                for climdex in climdex_names[targetVar]:
                    for season in season_dict:
                        if scene == 'TESTING':
                            fileIn = '_'.join((climdex, 'est', season))
                        else:
                            fileIn = '_'.join((climdex, scene, model, season))
                        fileOut = '_'.join((climdex, scene, model, season))
                        if os.path.isfile(pathIn + fileIn +'.nc'):
                            data = read.netCDF(pathIn, fileIn, climdex)['data']
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

