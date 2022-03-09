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
import evaluate_methods
import grids
import launch_jobs
import plot
import postpro_lib
import postprocess
import derived_predictors
import precontrol
import preprocess
import process
import read
import standardization
import TF_lib
import val_lib
import WG_lib
import write


def get_mean_and_std_allModels(var0, grid):
    """
    Calculates mean and standard deviation for reanalysis and models and all predictors.
    The time period used is the one with data from both reanalysis and models historical: 1980-2005.
    """

    scene = 'historical'
    periodFilename = historicalPeriodFilename


    pathOut = pathAux+'STANDARDIZATION/'+grid.upper()+'/'+var0.upper()+'/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Read low resolution data from reanalysis
    if pseudoreality == True:
        aux = read.lres_data(var0, grid, model=GCM_shortName, scene=scene)
        dates = aux['times']
        data = aux['data']
    else:
        dates = calibration_dates
        data = read.lres_data(var0, grid)['data']

    # Selects standardization period
    time_first, time_last=dates.index(reference_first_date),dates.index(reference_last_date)+1
    data=data[time_first:time_last]

    # Calculates mean and standard deviation and saves them to files
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    np.save(pathOut+'reanalysis_mean', mean)
    np.save(pathOut+'reanalysis_std', std)

    # Read historical series of all models
    expected_models = []
    for model in model_list:
        if model != 'reanalysis':
            if os.path.isfile('../input_data/models/psl_' + model + '_' + scene + '_' + modelRealizationFilename + '_' +
                              periodFilename + '.nc'):
                print('get_mean_and_std', var0, grid, scene, model)

                # Serial processing
                if running_at_HPC == False:
                    get_mean_and_std_oneModel(var0, grid, model, scene)

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
                                    os.system('mv ' + filename[:-3] + 'out ../job/err/')
                                    os.system('scancel ' + str(jobid))

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
                    launch_jobs.standardize(var0, grid, model, scene)

    # Check if all jobs have ended successfully
    if running_at_HPC == True:
        while 1:
            # Wait for all jobs to end
            os.system('squeue -u ' + user + ' | wc -l > ../log/nJobs.txt')
            f = open('../log/nJobs.txt', 'r')
            nJobs = int(f.read()) - 1
            f.close()
            time.sleep(1)
            if nJobs == 0:
                break

    # Check if expected files have been created
    for model in expected_models:
        if (
        not os.path.isfile(pathAux + 'STANDARDIZATION/' + grid.upper() + '/' + var0.upper() + '/' + model + '_mean.npy')) or \
                (not os.path.isfile(
                    pathAux + 'STANDARDIZATION/' + grid.upper() + '/' + var0.upper() + '/' + model + '_std.npy')):
            print('Error in get_mean_and_std', model)
            exit()

########################################################################################################################
def get_mean_and_std_oneModel(var0, grid, model, scene):

    pathOut = pathAux+'STANDARDIZATION/'+grid.upper()+'/'+var0.upper()+'/'

    # Read data and times from model/scene
    aux = read.lres_data(var0, grid, model=model, scene=scene)
    scene_dates = aux['times']
    calendar = read.netCDF('../input_data/models/', 'psl_' + model + '_' + scene +'_'+ modelRealizationFilename + '_'+
               historicalPeriodFilename+ '.nc', 'psl')['calendar']

    if calendar == '360':
        time_first, time_last = scene_dates.index(reference_first_date), -1
    else:
        time_first, time_last = scene_dates.index(reference_first_date), scene_dates.index(reference_last_date) + 1
    data = aux['data']
    data = data[time_first:time_last]

    # Calculates mean and standard deviation and saves them to files
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    np.save(pathOut+model+'_mean', mean)
    np.save(pathOut+model+'_std', std)
    del data

########################################################################################################################
def standardize(var0, A, model, grid):
    pathIn=pathAux+'STANDARDIZATION/'+grid.upper()+'/'+var0.upper()+'/'

    # Get mean and std
    if mean_and_std_from_GCM == True:
        mean = np.load(pathIn + model + '_mean.npy')
        std = np.load(pathIn + model + '_std.npy')
    elif mean_and_std_from_GCM == False:
        mean = np.load(pathIn + 'reanalysis_mean.npy')
        std = np.load(pathIn + 'reanalysis_std.npy')

    # Adapt dimensions
    mean = np.expand_dims(mean, axis=0)
    mean = np.repeat(mean, A.shape[0], 0)
    std = np.expand_dims(std, axis=0)
    std = np.repeat(std, A.shape[0], 0)

    # Standardize
    A = (A - mean) / std

    return A

########################################################################################################################
if __name__ == "__main__":
    nproc = MPI.COMM_WORLD.Get_size()  # Size of communicator
    iproc = MPI.COMM_WORLD.Get_rank()  # Ranks in communicator
    inode = MPI.Get_processor_name()  # Node where this MPI process runs
    var0 = sys.argv[1]
    grid = sys.argv[2]
    model = sys.argv[3]
    scene = sys.argv[4]

    get_mean_and_std_oneModel(var0, grid, model, scene)