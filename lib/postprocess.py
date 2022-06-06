import os
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
def bias_correction():
    """
    Bias correction of projections and get climdex from bias corrected daily data.
    """

    if apply_bc == True:
        if bc_method == None:
            print('Select a bc_method at advanced_settings.')
            exit()
        else:
            for method_dict in methods:
                var, methodName = method_dict['var'], method_dict['methodName']
                if experiment == 'EVALUATION':
                    bias_correction_renalysis(var, methodName)
                elif experiment == 'PROJECTIONS':
                    bias_correction_allModels(var, methodName)



########################################################################################################################
def bias_correction_renalysis(var, methodName):
    """
    Apply bias correction for a specific model.
    """

    print('postprocess.bias_correction_renalysis', var, methodName, bc_method)

    # Define and create paths
    pathIn = '../results/'+experiment+'/' + var.upper() + '/' + methodName + '/daily_data/'
    pathOut = '../results/'+experiment+'_BC-' + bc_method + '/' + var.upper() + '/' + methodName + '/daily_data/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Read data
    est_aux = read.netCDF(pathIn, 'reanalysis_TESTING.nc', var)
    est_times = est_aux['times']
    est_data = est_aux['data']
    obs_aux = read.hres_data(var, period='calibration')
    obs_data = obs_aux['data']
    obs_times = obs_aux['times']

    # Select common dates
    all_years = list(np.unique(np.array([x.year for x in est_times])))
    idates_common = [i for i in range(len(obs_times)) if obs_times[i].year in all_years]
    obs_data = obs_data[idates_common]
    obs_times = np.array(obs_times)[idates_common]

    # Display warning if few years
    if len(all_years) < 30:
        print('----------------------------------------')
        print('WARNING: reanalysis testing will be bias corrected, but there are only', len(all_years), 'yeas avalible')
        print('Bias correction with few years might not perform well')
        print('----------------------------------------')

    # print('obs', obs_data.shape, obs_times[0], obs_times[-1])
    # print('est', est_data.shape, est_times[0], est_times[-1])

    # Empty array for results
    scene_bc = np.zeros(est_data.shape)

    for year in all_years:
        idates_ref = [i for i in range(len(obs_times)) if obs_times[i].year!=year]
        idates_sce = [i for i in range(len(obs_times)) if obs_times[i].year==year]
        # print(year, len(idates_ref), len(idates_sce))
        obs = obs_data[idates_ref]
        mod = est_data[idates_ref]
        sce = obs_data[idates_sce]

        # Correct bias for year
        scene_bc[idates_sce] = BC_lib.biasCorrect_as_postprocess(obs, mod, sce, var, obs_times[idates_ref],
                                                                 est_times[idates_sce])

    # Set units
    if var == 'pcp':
        units = 'mm'
    else:
        units = 'degress'

    # Save bias corrected scene
    hres_lats = np.load(pathAux + 'ASSOCIATION/' + var[0].upper()+'_bilinear/hres_lats.npy')
    hres_lons = np.load(pathAux + 'ASSOCIATION/' + var[0].upper()+'_bilinear/hres_lons.npy')
    write.netCDF(pathOut, 'reanalysis_TESTING.nc', var, scene_bc, units, hres_lats, hres_lons,
                 est_times, regular_grid=False)



########################################################################################################################
def bias_correction_allModels(var, methodName):
    """
    Check for methods/models not yet corrected and applie bias correction
    """

    print('postprocess.bias_correction_allModels', var, methodName, bc_method)


    if apply_bc == True:

        # Define and create paths
        pathIn = '../results/'+experiment+'/' + var.upper() + '/' + methodName + '/daily_data/'
        pathOut = '../results/'+experiment+'_BC-' + bc_method + '/' + var.upper() + '/' + methodName + '/daily_data/'
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

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

                if to_be_corrected == True:

                    print(var, methodName, model, bc_method, 'bias_correction')

                    # Serial processing
                    if running_at_HPC == False:
                        bias_correction_oneModel(var, methodName, model)

                    # Parallel processing
                    elif running_at_HPC == True:
                        while 1:
                            # Check for error files, save them and kill erroneous jobs
                            # for file in os.listdir('../job/'):
                            #     if file.endswith(".err"):
                            #         filename = os.path.join('../job/', file)
                            #         filesize = os.path.getsize(filename)
                            #         if filesize != 0:
                            #             jobid = filename.split('/')[-1].split('.')[0]
                            #             print('-----------------------')
                            #             print(filename, filesize)
                            #             os.system('mv ' + filename + ' ../job/err/')
                            #             os.system('mv ' + filename[:-3] + 'out ../job/err/')
                            #             os.system('scancel ' + str(jobid))

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
                        launch_jobs.biasCorrection(model, var, methodName)



########################################################################################################################
def bias_correction_oneModel(var, methodName, model):
    """
    Apply bias correction for a specific model.
    """

    print('postprocess.bias_correction_oneModel', model, var, methodName, bc_method)

    # Define and create paths
    pathIn = '../results/'+experiment+'/' + var.upper() + '/' + methodName + '/daily_data/'
    pathOut = '../results/'+experiment+'_BC-' + bc_method + '/' + var.upper() + '/' + methodName + '/daily_data/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Read obs and mod in bias correction period
    obs_data = read.hres_data(var, period='biasCorr')['data']
    aux = read.netCDF(pathIn, model + '_historical.nc', var)
    idates = [i for i in range(len(aux['times'])) if
              aux['times'][i].year in range(biasCorr_years[0], biasCorr_years[1] + 1)]
    mod_data = aux['data'][idates]

    # Go through all scenes
    for scene in scene_list:

        # Check if scene/model exists
        if os.path.isfile(pathIn + model + '_' + scene + '.nc'):

            print(scene)

            # Read scene data
            aux = read.netCDF(pathIn, model + '_' + scene + '.nc', var)
            scene_dates = aux['times']
            scene_data = aux['data']
            del aux

            # Correct bias for scene
            scene_bc = BC_lib.biasCorrect_as_postprocess(obs_data, mod_data, scene_data, var, biasCorr_dates, scene_dates)

            # Set units
            if var == 'pcp':
                units = 'mm'
            else:
                units = 'degress'

            # Save bias corrected scene
            hres_lats = np.load(pathAux + 'ASSOCIATION/' + var[0].upper()+'_bilinear/hres_lats.npy')
            hres_lons = np.load(pathAux + 'ASSOCIATION/' + var[0].upper()+'_bilinear/hres_lons.npy')
            write.netCDF(pathOut, model + '_' + scene + '.nc', var, scene_bc, units, hres_lats, hres_lons,
                         scene_dates, regular_grid=False)


########################################################################################################################
def get_climdex():
    """
    Calls to get_climdex_for_evaluation (reanalysis) or get_climdex_allModels (models)
    """


    # Go through all methods
    for method_dict in methods:
        var = method_dict['var']
        methodName = method_dict['methodName']
        family = method_dict['family']
        mode = method_dict['mode']
        fields = method_dict['fields']

        if experiment == 'EVALUATION':
            get_climdex_for_evaluation(var, methodName)
        else:
            get_climdex_allModels(var, methodName)


########################################################################################################################
def get_climdex_for_evaluation(var, methodName):
    """
    Calculate climdex for evaluation
    """
    print(methodName)
    if apply_bc == False:
        pathOut = '../results/EVALUATION/' + var.upper() + '/' + methodName + '/climdex/'
        pathIn = '../results/EVALUATION/' + var.upper() + '/' + methodName + '/daily_data/'
    else:
        pathOut = '../results/EVALUATION_BC-' + bc_method + '/' + var.upper() + '/' + methodName + '/climdex/'
        pathIn = '../results/EVALUATION_BC-' + bc_method + '/' + var.upper() + '/' + methodName + '/daily_data/'

    if os.path.isfile(pathIn+'reanalysis_TESTING.nc'):

        # Read data
        d = postpro_lib.get_data_eval(var, methodName)
        ref, times_ref, obs, est, times_scene = d['ref'], d['times_ref'], d['obs'], d['est'], d['times_scene']
        del d

        # Calculate all climdex
        for (data, filename) in ((obs, 'obs'), (est, 'est')):
            postpro_lib.calculate_all_climdex(pathOut, filename, var, data, times_scene, ref, times_ref)


########################################################################################################################
def get_climdex_allModels(var, methodName):
    """
    Check if climdex/scene/model already exists, and if not, calculate it,
    """
    print('postprocess.calculate_climdex', var, methodName)

    # Define and create paths
    if pseudoreality == True:
        path = '../results/'+experiment+'/pseudoreality_'+GCM_longName+'_'+RCM+'/'+var.upper()+'/'+methodName+'/'
    else:
        if apply_bc == False:
            path = '../results/'+experiment+'/' + var.upper() + '/' + methodName + '/'
        else:
            path = '../results/'+experiment+'_BC-' + bc_method + '/' + var.upper() + '/' + methodName + '/'
    pathIn = path + 'daily_data/'
    pathOut = path + 'climdex/'

    # Go through all models
    for model in model_list:

        # Check if model climdex has already been calculated
        filenames = []
        for climdex_name in climdex_names[var]:
            for season in season_dict.values():
                # filenames.append(pathOut + '_'.join((climdex_name, 'REFERENCE', model, season)) + '.npy')
                for scene in scene_list:
                    filenames.append(pathOut + '_'.join((climdex_name, scene, model, season)) + '.npy')
        climdex_already_calculated = True
        for filename in filenames:
            scene = filename.split('_')[1]
            model = filename.split('_')[2] + '_' + filename.split('_')[3]
            if ((os.path.isfile(pathIn + model + '_' + scene + '.nc')) and (not os.path.isfile(filename))):
                climdex_already_calculated = False
        if climdex_already_calculated == False:
            # Check if model historical exists
            if os.path.isfile(pathIn + model + '_historical.nc'):

                # Serial processing
                if running_at_HPC == False:
                    print(var, methodName, model, 'calculating climdex')
                    get_climdex_oneModel(var, methodName, model)

                # Parallel processing
                elif running_at_HPC == True:
                    while 1:
                        # Check for error files, save them and kill erroneous jobs
                        # for file in os.listdir('../job/'):
                        #     if file.endswith(".err"):
                        #         filename = os.path.join('../job/', file)
                        #         filesize = os.path.getsize(filename)
                        #         if filesize != 0:
                        #             jobid = filename.split('/')[-1].split('.')[0]
                        #             print('-----------------------')
                        #             print(filename, filesize)
                        #             os.system('mv ' + filename + ' ../job/err/')
                        #             os.system('mv ' + filename[:-3] + 'out ../job/err/')
                        #             os.system('scancel ' + str(jobid))

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
                            print(var, methodName, model, 'calculating climdex')
                            break

                    # Send new job
                    launch_jobs.climdex(model, var, methodName)


########################################################################################################################
def get_climdex_oneModel(var, methodName, model):
    """
    Calculates all climdex for a specific model.
    """
    print(var, methodName, model)

    # Define and create paths
    if pseudoreality == True:
        path = '../results/'+experiment+'/pseudoreality_'+GCM_longName+'_'+RCM+'/'+var.upper()+'/'+methodName+'/'
    else:
        if apply_bc == False:
            path = '../results/'+experiment+'/' + var.upper() + '/' + methodName + '/'
        else:
            path = '../results/'+experiment+'_BC-' + bc_method + '/' + var.upper() + '/' + methodName + '/'
    pathIn = path + 'daily_data/'
    pathOut = path + 'climdex/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Read reference data (as a scene)
    aux = read.netCDF(pathIn, model + '_historical.nc', var)
    times_ref = aux['times']
    ref = aux['data']
    idates = [i for i in range(len(times_ref)) if
              times_ref[i].year in range(reference_years[0], reference_years[1] + 1)]
    times_ref = [times_ref[i] for i in idates]
    ref = ref[idates]

    # Read reference climatology (for percCalendar)
    if reference_climatology_from_observations == True:
        aux = read.hres_data(var, period='reference')
        ref_clim = aux['data']
        times_ref_clim = aux['times']
    else:
        ref_clim, times_ref_clim = ref, times_ref


    # Calculate climdex for reference period
    postpro_lib.calculate_all_climdex(pathOut, 'REFERENCE_' + model, var, ref, times_ref, ref_clim, times_ref_clim)

    # Go through all scenes
    for scene in scene_list:

        # Check if scene/model exists
        if os.path.isfile(pathIn + model + '_' + scene + '.nc'):
            # Read scene data
            aux = read.netCDF(pathIn, model + '_' + scene + '.nc', var)
            times = aux['times']
            data = aux['data']
            del aux

            # Calculate climdex for scene
            postpro_lib.calculate_all_climdex(pathOut, scene + '_' + model, var, data, times, ref_clim, times_ref_clim)



########################################################################################################################
def plot_results():
    """
    Divide by subregions and generate graphics for EVALUATION or for PROJECTIONS
    """

    # Establish subregions for each point
    for var0 in target_vars0:
        grids.subregions(var0)

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

    if apply_bc == True:
        sufix = '_BC-'+bc_method
    else:
        sufix = ''

    pathOutBase = '../results/'+experiment+sufix+'_ASCII/'

    for method_dict in methods:
        var, methodName = method_dict['var'], method_dict['methodName']

        if var == 'pcp':
            units = 'mm'
        else:
            units = degree_sign

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
                pathIn = '../results/'+experiment+sufix+'/'+var.upper()+'/'+methodName+'/daily_data/'
                fileName = model+'_'+scene
                if os.path.isfile(pathIn + fileName +'.nc'):
                    nc = read.netCDF(pathIn, model + '_' + scene + '.nc', var)
                    times = nc['times']
                    data = nc['data']
                    data[np.isnan(data)] = -999
                    del nc
                    print('writing daily data to ASCCI file for', var, methodName, bc_method, scene, model, '...')
                    id = list(read.hres_metadata(var[0]).index.values)
                    times = np.array([10000 * x.year + 100 * x.month + x.day for x in times])
                    data = np.append(times[:, np.newaxis], data, axis=1)
                    id.insert(0, 'YYYYMMDD')
                    id = [str(x) for x in id]
                    header = ' '.join((model, scene, period, var, '('+units+')',
                                       methodName)) + '\n' + ';'.join(id)
                    pathOut = pathOutBase + var.upper()+'/'+methodName+'/daily_data/'
                    if not os.path.exists(pathOut):
                        os.makedirs(pathOut)
                    np.savetxt(pathOut+fileName+'.dat', data, fmt=['%.i'] + ['%.2f'] * (len(id) - 1), delimiter=';', header=header)


                # Climdex
                pathIn = '../results/'+experiment+sufix+'/'+var.upper()+'/'+methodName+'/climdex/'
                for climdex in climdex_names[var]:
                    for season in season_dict.values():
                        if scene == 'TESTING':
                            fileIn = '_'.join((climdex, 'est', season))
                        else:
                            fileIn = '_'.join((climdex, scene, model, season))
                        fileOut = '_'.join((climdex, scene, model, season))
                        if os.path.isfile(pathIn + fileIn +'.npy'):
                            data = np.load(pathIn + fileIn +'.npy')
                            id = list(read.hres_metadata(var[0]).index.values)
                            times = np.array(years)
                            data = np.append(times[:, np.newaxis], data, axis=1)
                            id.insert(0, 'YYYY')
                            id = [str(x) for x in id]
                            header = ' '.join((model, scene, period, season, var, climdex, '('+units+')',
                                               methodName)) + '\n' + ';'.join(id)
                            pathOut = pathOutBase + var.upper()+'/'+methodName+'/climdex/'
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
    var = sys.argv[3]
    methodName = sys.argv[4]

    if task == 'climdex':
        get_climdex_oneModel(var, methodName, model)
    elif task == 'bias_correction':
        bc_method = sys.argv[5]
        if bc_method == 'None':
            bc_method = None
        bias_correction_oneModel(var, methodName, model, bc_method)

