import sys
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import BC_lib
import down_scene_ANA
import down_scene_BC
import down_scene_RAW
import down_scene_TF
import down_scene_WG
import down_day
import down_point
import evaluate_GCMs
import evaluate_methods
import grids
import launch_jobs
import plot
import postpro_lib
import postprocess
import derived_predictors
import preprocess
import process
import read
import standardization
import TF_lib
import val_lib
import WG_lib
import write

########################################################################################################################
def reanalisys(method_dict, scene, model):
    """
    This function calls the down_scene in two different ways depending on the server it is run.
    """

    var = method_dict['var']
    methodName = method_dict['methodName']
    family = method_dict['family']
    mode = method_dict['mode']
    fields = method_dict['fields']

    # Serial processing
    if running_at_HPC == False:
        if family == 'ANA':
            down_scene_ANA.downscale_chunk(var, methodName, family, mode, fields, scene, model)
            down_scene_ANA.collect_chunks(var, methodName, family, mode, fields, scene, model)
        elif family == 'TF':
            down_scene_TF.downscale_chunk(var, methodName, family, mode, fields, scene, model)
            down_scene_TF.collect_chunks(var, methodName, family, mode, fields, scene, model)
        elif family == 'RAW':
            down_scene_RAW.downscale_chunk(var, methodName, family, mode, fields, scene, model)
            down_scene_RAW.collect_chunks(var, methodName, family, mode, fields, scene, model)
        elif family == 'BC':
            down_scene_BC.downscale_chunk(var, methodName, family, mode, fields, scene, model)
            down_scene_BC.collect_chunks(var, methodName, family, mode, fields, scene, model)
        elif family == 'WG':
            down_scene_WG.downscale_chunk(var, methodName, family, mode, fields, scene, model)
            down_scene_WG.collect_chunks(var, methodName, family, mode, fields, scene, model)

    # Parallel processing
    elif running_at_HPC == True:
        launch_jobs.process(var, methodName, family, mode, fields, scene, model)

########################################################################################################################
def models(method_dict, scene, model):
    """
    This function checks if method/scene/model has already been downscaled, and if not, it calls the down_scene in two
    different ways depending on the server it is run.
    """

    var = method_dict['var']
    methodName = method_dict['methodName']
    family = method_dict['family']
    mode = method_dict['mode']
    fields = method_dict['fields']

    pathOut = '../results/'+experiment+'/'

    if experiment == 'PSEUDOREALITY':
        pathOut += 'pseudoreality_' + GCM_longName + '_' + RCM + '/'

    if scene == 'historical':
        periodFilename = historicalPeriodFilename
    else:
        periodFilename = rcpPeriodFilename

    # check if scene/model exists
    if not os.path.isfile('../input_data/models/psl_' + model + '_' + scene +'_'+ modelRealizationFilename + '_'+periodFilename + '.nc'):
        print(scene, model, 'Does not exist')
    else:
        # Check if scene/model has already been processed
        if os.path.isfile(pathOut + var.upper() + '/' + methodName + '/daily_data/' + model + '_' + scene + '.nc'):
            print('-------------------------------')
            print(scene, model, methodName, 'Already processed')
        else:
            # Serial processing
            if running_at_HPC == False:
                print('-------------------------------')
                print(scene, model, methodName, 'Processing')
                if family == 'ANA':
                    down_scene_ANA.downscale_chunk(var, methodName, family, mode, fields, scene, model)
                    down_scene_ANA.collect_chunks(var, methodName, family, mode, fields, scene, model)
                elif family == 'TF':
                    down_scene_TF.downscale_chunk(var, methodName, family, mode, fields, scene, model)
                    down_scene_TF.collect_chunks(var, methodName, family, mode, fields, scene, model)
                elif family == 'RAW':
                    down_scene_RAW.downscale_chunk(var, methodName, family, mode, fields, scene, model)
                    down_scene_RAW.collect_chunks(var, methodName, family, mode, fields, scene, model)
                elif family == 'BC':
                    down_scene_BC.downscale_chunk(var, methodName, family, mode, fields, scene, model)
                    down_scene_BC.collect_chunks(var, methodName, family, mode, fields, scene, model)
                elif family == 'WG':
                    down_scene_WG.downscale_chunk(var, methodName, family, mode, fields, scene, model)
                    down_scene_WG.collect_chunks(var, methodName, family, mode, fields, scene, model)

            # Parallel processing
            elif running_at_HPC == True:
                while 1:
                    # Check for error files, save them and kill erroneous jobs
                    for file in os.listdir('../job/'):
                        if file.endswith(".err"):
                            filename = os.path.join('../job/', file)
                            filesize = os.path.getsize(filename)
                            if filesize != 0:
                                jobid = filename.split('/')[-1].split('.')[0]
                                print('-----------------------')
                                print(filename, filesize)
                                os.system('mv ' + filename + ' ../job/err/')
                                os.system('mv ' + filename[:-3]+'out ../job/err/')
                                os.system('scancel ' + str(jobid))

                    # Check for correctly finished jobs
                    for file in os.listdir('../job/'):
                        if file.endswith(".out"):
                            filename=os.path.join('../job/', file)
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
                        print('-------------------------------')
                        print(scene, model, methodName, 'Processing')
                        break

                # Send new job
                launch_jobs.process(var, methodName, family, mode, fields, scene, model)

########################################################################################################################
def downscale():
    """
    Calls to downscale reanalysis or downscale models/scenes.
    """

    # Go through all methods
    for method_dict in methods:

        # Go through all scenes
        for scene in scene_list:

            # Go through all models
            for model in model_list:

                # Call process.reanalysis or process.models
                if model == 'reanalysis':
                    reanalisys(method_dict, scene, model)
                else:
                    models(method_dict, scene, model)

