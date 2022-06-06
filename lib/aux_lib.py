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
def initial_checks():
    """
    Check whether aux, tmp and results folders exist.
    Check for living jobs and kill them (optional)
    """

    # Create needed paths
    if not os.path.exists('../job/'):
        os.makedirs('../job/')
        os.makedirs('../job/out/')
        os.makedirs('../job/err/')
    if not os.path.exists('../log/'):
        os.makedirs('../log/')
    needed_paths = ['aux', 'tmp', 'results']
    nonExisting_paths = []
    for pathName in needed_paths:
        if not os.path.exists('../'+pathName+'/'):
            nonExisting_paths.append(pathName)
    # os.makedirs('../'+pathName+'/')
    if nonExisting_paths != []:
        print('--------------------------------------------------------------------')
        print('The following directories will be created unless they already exist:')
        for path in needed_paths:
            print('  - ' + path)
        print('and they will storage a great volume of data.')
        print('It is recommended to stop the program and create them as links to a masive storage disk.')
        a = input('Enter "c" so they will be automatically created, or press "Enter" to stop the program and create them yourself: ')
        if a == 'c':
            for pathName in nonExisting_paths:
                os.makedirs('../'+pathName+'/')
        else:
            exit()

    # Kill living old jobs
    if running_at_HPC == True:
        a = input('\nPress "k" to kill running jobs and delete logs or any other key to preserve them:')
        # a = 'k'
        # a = ''
        if a.lower() == 'k':
            os.system('rm -r ../tmp/*')
            os.system('rm -r ../job/out/*')
            os.system('rm -r ../job/err/*')
            os.system('squeue -u ' + user + ' > ../job/jobs.txt')
            with open('../job/jobs.txt', 'r') as f:
                for line in f:
                    jobid = str([x for x in line.split(' ') if x != ''][0])
                    if jobid != 'JOBID':
                        os.system('scancel ' + jobid)
            # empty log folder
            files = glob.glob('../job/*.*')
            for f in files:
                os.remove(f)

########################################################################################################################
def check_var_units():
    """
    Check units for the different variables by the reanalysis and by a GCM
    """

    # for var in ('tmax', 'tmin', 'pcp'):
    for var in reaNames:

        try:
            aux = read.one_direct_predictor(var)
            reaMin = np.nanmin(aux['data'])
            reaMin = str(np.round(reaMin, 1))
            reaMax = np.nanmax(aux['data'])
            reaMax = str(np.round(reaMax, 1))
            reaUnits=aux['units']
        except:
            reaUnits, reaMin, reaMax = '-', '-', '-'

        model_found = False
        for model in model_names_list:
            for scene in scene_names_list:
                scene = scene.lower().replace('-', '').replace('.', '')
                try:
                    aux=read.one_direct_predictor(var, model=model, scene=scene)
                    modMin = np.nanmin(aux['data'])
                    modMin = str(np.round(modMin, 1))
                    modMax = np.nanmax(aux['data'])
                    modMax = str(np.round(modMax, 1))
                    modUnits=aux['units']
                    model_found = True
                    break
                except:
                    modUnits, modMin, modMax = '-', '-', '-'
                if model_found == True:
                    break
            if model_found == True:
                break

        print('-------------------------------------------------------------------------------------------------------')
        print('Variable:', var)
        print('Reanalysis:'.ljust(15), 'Min:', reaMin.ljust(10), 'Max:', reaMax.ljust(10), 'Units:', reaUnits.ljust(15))
        print('Models:'.ljust(15),      'Min:', modMin.ljust(10), 'Max:', modMax.ljust(10), 'Units:', modUnits.ljust(15))


########################################################################################################################
def join_kfolds(var, methodName, family, mode, fields, scene, model, units, hres_lats, hres_lons):
    """
    Join 5 kfolds and into a single file and delete the folds.
    Caution. The folds are joint in the specific order 1-2-3-4-5.
    """

    # Path
    path = '../results/'+experiment+'/'+var.upper()+'/'+methodName+'/daily_data/'

    # Join folds
    data_list = []
    times = []
    for ifold in range(5):
        aux = read.netCDF(path, '_'.join((model, scene, 'fold')) + str(ifold+1) + '.nc', var)
        data_list.append(aux['data'])
        times += list(aux['times'])
    data = np.concatenate((data_list[0], data_list[1], data_list[2], data_list[3], data_list[4]))

    # Save results and delete folds
    write.netCDF(path, '_'.join((model, scene)) + '.nc', var, data, units, hres_lats, hres_lons, times, regular_grid=False)
    # os.system('rm ' + path + '*fold*')
    for ifold in range(5):
        os.remove(path + model + '_' + scene + '_fold' + str(ifold) + '.nc')


########################################################################################################################
def prepare_hres_data_ascii2npy(var):
    """
    This function prepares hres_data: ascci to npy
    When run for the first time, reads data from large txt file, which is too slow.
    When run next times it read data from npy file.
    In order turn ascii file to npy file, the ascii file is read line by line, and each line is appended to an array.
    This array gets too big for memory.
    To prevent swapping the array is cut into chuncks which later are read.

    In the ascii file missing data correspond to fill_value_txt
    In the npy file missing data correspond to np.nan
    This function convert the fill_value_txt to np.nan, and also the np.nan to a special value so the rest of the
    program understand. This special value is the maximum value for int16 or uint16 (temp and pcp respectively).
    The reason for this design is that predictands are used as numpy arrays of integers, and they do not allow np.nan

    When the original data contain years out of both calibration and reference period they are remove at first_run.

    """

    filename = pathHres + var + '_' + hresPeriodFilename[var[0]]
    fill_value_txt = -999 # This is the value in the txt file, and this function convert it to np.nan for the .npy file

    minYear = int(hresPeriodFilename[var[0]].split('-')[0][:4])
    maxYear = int(hresPeriodFilename[var[0]].split('-')[1][:4])

    tmp='../tmp/'
    if not os.path.exists(tmp):
        os.makedirs(tmp)

    # ------------------------
    # read data
    # ------------------------
    if pseudoreality == True:
        exit('Do not run read.hres_data first time True for pseudoreality')
    num_lines = sum(1 for line in open(filename + '.txt'))
    lon_line=hres_npoints[var[0]]+1
    data=np.zeros((0))
    chunk=0
    CHUNK=0

    f = open(filename + '.txt', 'r')
    for i in range(num_lines):
        line = f.readline()
        line = line.split(' ')
        line = [item for item in line if item!='']

        # Checks all lines has the same lengh
        if len(line) != lon_line:
            exit(line)

        year = int(line[0][:4])
        if year in range(minYear, maxYear+1):
            # Appends line to array
            data = np.append(data, np.asanyarray(line, dtype=float), axis=0)
            # Cut in chunks
            if (i % 100 == 0) or (i == (num_lines - 1)):
                print('read.hres_data', filename, i, '/', num_lines, 100 * i / num_lines, '%')
                np.save(tmp + 'chunk' + str(chunk), data)
                chunk += 1
                data = np.zeros((0))
    f.close()


    for ichunk in range(chunk):
        print('ichunk', ichunk, '/' , chunk)
        data = np.append(data, np.load(tmp + 'chunk' + str(ichunk) + '.npy'))
        os.system('rm ' + tmp + 'chunk' + str(ichunk) + '.npy')

        # Cut in CHUNKS
        if (ichunk % 10 == 0) or (ichunk == (chunk - 1)):
            np.save(tmp + 'CHUNK' + str(CHUNK), data)
            CHUNK += 1
            data = np.zeros((0))

    for ICHUNK in range(CHUNK):
        print('ICHUNK', ICHUNK, '/', CHUNK)
        data = np.append(data, np.load(tmp + 'CHUNK' + str(ICHUNK) + '.npy'))
        os.system('rm ' + tmp + 'CHUNK' + str(ICHUNK) + '.npy')

    data = data.reshape(-1, lon_line)
    data[data==fill_value_txt] = np.nan
    data = data[np.argsort(data[:, 0])]
    np.save(filename, data)
