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
def reanalisys(method_dict, scene, model):
    """
    This function calls the down_scene in two different ways depending on the server it is run.
    """

    targetVar = method_dict['var']
    methodName = method_dict['methodName']
    family = method_dict['family']
    mode = method_dict['mode']
    fields = method_dict['fields']

    # Serial processing
    if running_at_HPC == False:
        if family == 'ANA':
            down_scene_ANA.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
            down_scene_ANA.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)
        elif family == 'TF':
            down_scene_TF.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
            down_scene_TF.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)
        elif family == 'RAW':
            down_scene_RAW.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
            down_scene_RAW.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)
        elif family == 'MOS':
            down_scene_MOS.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
            down_scene_MOS.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)
        elif family == 'WG':
            down_scene_WG.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
            down_scene_WG.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)
        elif family == 'DEEP':
            down_scene_DEEP.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
            down_scene_DEEP.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)

    # Parallel processing
    elif running_at_HPC == True:
        launch_jobs.process(targetVar, methodName, family, mode, fields, scene, model)
        print('Downscaling', targetVar, methodName, 'reanalysis')


########################################################################################################################
def models(method_dict, scene, model):
    """
    This function checks if method/scene/model has already been downscaled, and if not, it calls the down_scene in two
    different ways depending on the server it is run.
    """

    targetVar = method_dict['var']
    methodName = method_dict['methodName']
    family = method_dict['family']
    mode = method_dict['mode']
    fields = method_dict['fields']

    pathOut = '../results/'+experiment+'/'

    if experiment == 'PSEUDOREALITY':
        pathOut += 'pseudoreality_' + GCM_longName + '_' + RCM + '/'

    # Check if scene/model has already been processed
    if os.path.isfile(pathOut + targetVar.upper() + '/' + methodName + '/daily_data/' + model + '_' + scene + '.nc'):
        print('-------------------------------')
        print(targetVar, scene, model, methodName, 'Already processed')
    else:
        # Serial processing
        if running_at_HPC == False:
            print('-------------------------------')
            print(targetVar, scene, model, methodName, 'Processing')
            if family == 'ANA':
                down_scene_ANA.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
                down_scene_ANA.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)
            elif family == 'TF':
                down_scene_TF.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
                down_scene_TF.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)
            elif family == 'RAW':
                down_scene_RAW.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
                down_scene_RAW.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)
            elif family == 'MOS':
                down_scene_MOS.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
                down_scene_MOS.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)
            elif family == 'WG':
                down_scene_WG.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
                down_scene_WG.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)
            elif family == 'DEEP':
                down_scene_DEEP.downscale_chunk(targetVar, methodName, family, mode, fields, scene, model)
                down_scene_DEEP.collect_chunks(targetVar, methodName, family, mode, fields, scene, model)

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
                    break

            # Send new job
            launch_jobs.process(targetVar, methodName, family, mode, fields, scene, model)
            print('nJobs', nJobs)
            print('-------------------------------')
            print(targetVar, scene, model, methodName, 'Processing')

########################################################################################################################
def downscale():
    """
    Calls to downscale reanalysis or downscale models/scenes.
    """

    # Go through all methods
    for method_dict in methods:

        # Go through all models
        for model in model_list:

            # Go through all scenes
            for scene in scene_list:

                # Call process.reanalysis or process.models
                if model == 'reanalysis':
                    reanalisys(method_dict, scene, model)
                else:
                    models(method_dict, scene, model)

