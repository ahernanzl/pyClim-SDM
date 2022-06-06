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

def downscale_chunk(var, methodName, family, mode, fields, scene, model, iproc=0, nproc=1):
    """
    This function goes through all points (regression).
    The result is saved as npy file (each chunk is one file).
    """

    # create chunks
    n_chunks = nproc
    len_chunk = int(math.ceil(float(hres_npoints[var[0]]) / n_chunks))
    points_chunk = []
    for ichunk in range(n_chunks):
        points_chunk.append(list(range(hres_npoints[var[0]]))[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    ichunk = iproc
    npoints_ichunk = len(points_chunk[ichunk])

    # Define paths
    pathOut = '../tmp/ESTIMATED_' + '_'.join((var, methodName, scene, model)) + '/'

    # Parent process reads all data, broadcasts to the other processes and creates paths for results
    if iproc == 0:
        print(var, methodName, scene, model)
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Read data and converts obs to uint16 or int16 to save memory
        obs = read.hres_data(var, period='training')['data']
        i_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_mode+'/i_4nn.npy')
        j_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_mode+'/j_4nn.npy')
        w_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_mode+'/w_4nn.npy')

        # Set scene dates and predictors
        if scene == 'TESTING':
            var_calib = np.load(pathAux+'STANDARDIZATION/VAR/'+var+'_training.npy')
            scene_dates = testing_dates
            var_scene = np.load(pathAux+'STANDARDIZATION/VAR/'+var+'_testing.npy')
        else:
            if scene == 'historical':
                years = historical_years
            else:
                years = ssp_years

            # Read var_calib (from model)
            aux = read.lres_data(var, 'var', model=model, scene='historical')
            dates = aux['times']
            var_calib = aux['data']
            if var[0] == 'p':
                preds = preds_dict['p']
            elif var[0] == 't':
                preds = preds_dict['t']
            ncName = list(preds.keys())[0]
            calendar = read.one_direct_predictor(ncName, model=model, scene=scene)['calendar']

            # Auxiliary dates "avail" are defined in case historical projections contain years out of calibration_years
            avail_first_date = max(calibration_first_date, dates[0])
            avail_last_date = min(calibration_last_date, dates[-1])
            avail_ndates = (avail_last_date - avail_first_date).days + 1
            avail_dates = [avail_first_date + datetime.timedelta(days=i) for i in range(avail_ndates)]

            time_first, time_last = dates.index(avail_first_date), dates.index(avail_last_date) + 1
            var_calib = var_calib[time_first:time_last]
            if calendar == '365':
                dates = [x for x in avail_dates if not ((x.month == 2) and (x.day == 29))]
            elif calendar == '360':
                dates = [x for x in avail_dates if not (((x.month == 2) and (x.day >= 29)) | (x.day > 30))]
            else:
                dates = avail_dates
            years_aux = np.array([x.year for x in dates])
            idates_calib = np.array(
                [i for i in range(years_aux.size) if ((years_aux[i] < testing_years[0]) | (years_aux[i] > testing_years[1]))])
            var_calib = var_calib[idates_calib]

            # Read var_scene (from model)
            aux = read.lres_data(var, 'var', model=model, scene=scene)
            scene_dates = aux['times']
            idates = [i for i in range(len(scene_dates)) if scene_dates[i].year >= years[0] and scene_dates[i].year <= years[1]]
            scene_dates = list(np.array(scene_dates)[idates])
            var_scene = aux['data'][idates]

    # Declares variables for the other processes
    else:
        obs = None
        scene_dates = None
        i_4nn = None
        j_4nn = None
        w_4nn = None
        var_calib = None
        var_scene = None

    # Share data with all subprocesses
    if nproc>1:
        obs = MPI.COMM_WORLD.bcast(obs, root=0)
        scene_dates = MPI.COMM_WORLD.bcast(scene_dates, root=0)
        i_4nn = MPI.COMM_WORLD.bcast(i_4nn, root=0)
        j_4nn = MPI.COMM_WORLD.bcast(j_4nn, root=0)
        w_4nn = MPI.COMM_WORLD.bcast(w_4nn, root=0)
        var_calib = MPI.COMM_WORLD.bcast(var_calib, root=0)
        var_scene = MPI.COMM_WORLD.bcast(var_scene, root=0)
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step

    if nproc > 1:
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step

    # Create empty array for results
    est = np.zeros((len(scene_dates), npoints_ichunk))
    special_value = int(predictands_codification[var]['special_value'])

    # Goes through all points of the chunk
    for ipoint in range(len(points_chunk[ichunk])):
        ipoint_local_index = ipoint
        ipoint = points_chunk[ichunk][ipoint_local_index]

        # Prints for monitoring
        if ipoint_local_index % 10==0:
            print('--------------------')
            print('ichunk:	', ichunk, '/', n_chunks)
            print('downscaling', var, methodName, scene, model, round(100*ipoint_local_index/npoints_ichunk, 2), '%')

        # Interpolate to ipoint
        X_test = grids.interpolate_predictors(var_scene, i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_mode)
        X_train = grids.interpolate_predictors(var_calib, i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_mode)
        Y_train = obs[:, ipoint]
        Y_train = Y_train[:, np.newaxis]


        # Apply downscaling
        if methodName == 'QM':
            est[:, ipoint_local_index] = BC_lib.quantile_mapping(Y_train, X_train, X_test, var)[:, 0]
        if methodName == 'DQM':
            est[:, ipoint_local_index] = BC_lib.detrended_quantile_mapping(Y_train, X_train, X_test, var)[:, 0]
        if methodName == 'QDM':
            est[:, ipoint_local_index] = BC_lib.quantile_delta_mapping(Y_train, X_train, X_test, var)[:, 0]
        elif methodName == 'PSDM':
            est[:, ipoint_local_index] = BC_lib.scaled_distribution_mapping(Y_train, X_train, X_test, var)[:, 0]

    # Undo converssion
    est = est.astype('float64')

    # Saves results
    np.save(pathOut + 'ichunk_' + str(ichunk) + '.npy', est)


########################################################################################################################
def collect_chunks(var, methodName, family, mode, fields, scene, model, n_chunks=1):
    """
    This function collects the results of downscale_chunk() and saves them into a final single file.
    """
    print('--------------------------------------')
    print(scene, model, 'collect chunks', n_chunks)

    # Gets scene dates
    if scene == 'TESTING':
        scene_dates = testing_dates
        model_dates = testing_dates
    else:
        if scene == 'historical':
            periodFilename = historicalPeriodFilename
            scene_dates = historical_dates
        else:
            periodFilename = sspPeriodFilename
            scene_dates = ssp_dates
        # Read dates (can be different for different calendars)
        path = '../input_data/models/'
        ncVar = modNames[var]
        modelName, modelRun = model.split('_')[0], model.split('_')[1]
        filename = ncVar + '_' + modelName + '_' + scene +'_'+ modelRun + '_'+periodFilename + '.nc'
        model_dates = np.ndarray.tolist(read.netCDF(path, filename, ncVar)['times'])
        model_dates = [x for x in model_dates if x.year >= scene_dates[0].year and x.year <= scene_dates[-1].year]

    # Create empty array and accumulate results
    est = np.zeros((len(model_dates), 0))
    for ichunk in range(n_chunks):
        path = '../tmp/ESTIMATED_'+ '_'.join((var, methodName, scene, model)) + '/'
        filename = path + '/ichunk_' + str(ichunk) + '.npy'
        est = np.append(est, np.load(filename), axis=1)
    shutil.rmtree(path)

    # Save to file
    if experiment == 'EVALUATION':
        pathOut = '../results/'+experiment+'/'+var.upper()+'/'+methodName+'/daily_data/'
    elif experiment == 'PSEUDOREALITY':
        aux = np.zeros((len(scene_dates), hres_npoints[var[0]]))
        aux[:] = np.nan
        idates = [i for i in range(len(scene_dates)) if scene_dates[i] in model_dates]
        aux[idates] = est
        est = aux
        del aux
        pathOut = '../results/'+experiment+'/'+ GCM_longName + '_' + RCM + '/'+var.upper()+'/'+methodName+'/daily_data/'
    else:
        aux = np.zeros((len(scene_dates), hres_npoints[var[0]]))
        aux[:] = np.nan
        idates = [i for i in range(len(scene_dates)) if scene_dates[i] in model_dates]
        aux[idates] = est
        est = aux
        del aux
        pathOut = '../results/'+experiment+'/'+var.upper()+'/'+methodName+'/daily_data/'

    # Save results
    hres_lats = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_mode+'/hres_lats.npy')
    hres_lons = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_mode+'/hres_lons.npy')

    # Set units
    if var == 'pcp':
        units = 'mm'
    else:
        units = 'degress'

    if split_mode[:4] == 'fold':
        sufix = '_' + split_mode
    else:
        sufix = ''

    # Special values are set to nan
    warnings.filterwarnings("ignore", message="invalid value encountered in less")
    est[np.abs(est-predictands_codification[var]['special_value']) < 0.01] = np.nan
    print('-------------------------------------------------------------------------')
    print('results contain', np.where(np.isnan(est))[0].size, 'nans out of', est.size)
    print('-------------------------------------------------------------------------')

    # Save data to netCDF file
    write.netCDF(pathOut, model+'_'+scene+sufix+'.nc', var, est, units, hres_lats, hres_lons, scene_dates, regular_grid=False)
    # print(est[0, :10], est.shape)

    # If using k-folds, join them
    if split_mode == 'fold5':
        aux_lib.join_kfolds(var, methodName, family, mode, fields, scene, model, units, hres_lats, hres_lons)

########################################################################################################################

if __name__=="__main__":

    nproc = MPI.COMM_WORLD.Get_size()         # Size of communicator
    iproc = MPI.COMM_WORLD.Get_rank()         # Ranks in communicator
    inode = MPI.Get_processor_name()          # Node where this MPI process runs
    var = sys.argv[1]
    methodName = sys.argv[2]
    family = sys.argv[3]
    mode = sys.argv[4]
    fields = sys.argv[5]
    scene = sys.argv[6]
    model = sys.argv[7]

    downscale_chunk(var, methodName, family, mode, fields, scene, model, iproc, nproc)
    MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step
    if iproc==0:
        collect_chunks(var, methodName, family, mode, fields, scene, model, nproc)
