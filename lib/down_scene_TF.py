import sys
sys.path.append(('../config/'))
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
        y_train = read.hres_data(var, period='training')['data']
        y_train = (100 * y_train).astype(predictands_codification[var]['type'])
        i_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_mode+'/i_4nn.npy')
        j_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_mode+'/j_4nn.npy')
        w_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_mode+'/w_4nn.npy')

        # Read X_train
        if 'pred' in fields:
            pred_calib = np.load(pathAux+'STANDARDIZATION/PRED/'+var[0]+'_training.npy')
            pred_calib = pred_calib.astype('float32')
            X_train = pred_calib
        if 'saf' in fields:
            saf_calib = np.load(pathAux+'STANDARDIZATION/SAF/'+var[0]+'_training.npy')
            saf_calib = saf_calib.astype('float32')
            X_train = saf_calib
        if 'var' in fields:
            var_calib = np.load(pathAux+'STANDARDIZATION/VAR/'+var+'_training.npy')
            if 'pred' not in fields:
                X_train = var_calib
            else:
                # For Radom Forest and Extreme Gradient Boost mixing pred (standardized) and var (pcp) is allowed
                X_train = np.concatenate((X_train, var_calib), axis=1)

        # Set scene dates and predictors
        if scene == 'TESTING':
            scene_dates = testing_dates
            if 'pred' in fields:
                pred_scene = np.load(pathAux+'STANDARDIZATION/PRED/'+var[0]+'_testing.npy')
                pred_scene = pred_scene.astype('float32')
                X_test = pred_scene
            if 'saf' in fields:
                saf_scene = np.load(pathAux+'STANDARDIZATION/SAF/'+var[0]+'_testing.npy')
                saf_scene = saf_scene.astype('float32')
                X_test = saf_scene
            if 'var' in fields:
                var_scene = np.load(pathAux+'STANDARDIZATION/VAR/'+var+'_testing.npy')
                if 'pred' not in fields:
                    X_test = var_scene
                else:
                    # For Radom Forest and Extreme Gradient Boost mixing pred (standardized) and var (pcp) is allowed
                    X_test = np.concatenate((X_test, var_scene), axis=1)
        else:
            if scene == 'historical':
                years = historical_years
                periodFilename = historicalPeriodFilename
            else:
                years = ssp_years
                periodFilename= sspPeriodFilename

            # Read dates (can be different for different calendars)
            ncVar = modNames[var]
            modelName, modelRun = model.split('_')[0], model.split('_')[1]
            scene_dates = read.netCDF('../input_data/models/', ncVar + '_' + modelName + '_' + scene +'_'+ modelRun + '_'+periodFilename+ '.nc',
                            ncVar)['times']
            idates = [i for i in range(len(scene_dates)) if scene_dates[i].year >= years[0] and scene_dates[i].year <= years[1]]
            scene_dates = list(np.array(scene_dates)[idates])
            if 'pred' in fields:
                pred_scene = read.lres_data(var, 'pred', model=model, scene=scene)['data'][idates]
                pred_scene = standardization.standardize(var[0], pred_scene, model, 'pred')
                pred_scene = pred_scene.astype('float32')
                X_test = pred_scene
            if 'saf' in fields:
                saf_scene = read.lres_data(var, 'saf', model=model, scene=scene)['data'][idates]
                saf_scene = standardization.standardize(var[0], saf_scene, model, 'saf')
                saf_scene = saf_scene.astype('float32')
                X_test = saf_scene
            if 'var' in fields:
                var_scene = read.lres_data(var, 'var', model=model, scene=scene)['data'][idates]
                if 'pred' not in fields:
                    X_test = var_scene
                else:
                    # For Radom Forest and Extreme Gradient Boost mixing pred (standardized) and var (pcp) is allowed
                    X_test = np.concatenate((X_test, var_scene), axis=1)

    # Declares variables for the other processes
    else:
        y_train = None
        scene_dates = None
        i_4nn = None
        j_4nn = None
        w_4nn = None
        X_train = None
        X_test = None

    # Share data with all subprocesses
    if nproc>1:
        y_train = MPI.COMM_WORLD.bcast(y_train, root=0)
        scene_dates = MPI.COMM_WORLD.bcast(scene_dates, root=0)
        i_4nn = MPI.COMM_WORLD.bcast(i_4nn, root=0)
        j_4nn = MPI.COMM_WORLD.bcast(j_4nn, root=0)
        w_4nn = MPI.COMM_WORLD.bcast(w_4nn, root=0)
        X_train = MPI.COMM_WORLD.bcast(X_train, root=0)
        X_test = MPI.COMM_WORLD.bcast(X_test, root=0)
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step

    if nproc > 1:
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step

    # Create empty array for results
    est = np.zeros((len(scene_dates), npoints_ichunk))
    est = est.astype(predictands_codification[var]['type'])
    special_value = int(100 * predictands_codification[var]['special_value'])
    npreds = X_test.shape[1]
    clf = None

    # Goes through all points of the chunk
    for ipoint in points_chunk[ichunk]:
        ipoint_local_index = points_chunk[ichunk].index(ipoint)

        # Load regressor and classifier of ipoint
        try:
            reg = pickle.load(open(pathAux + 'TRAINED_MODELS/' + var.upper() + '/' + methodName +
                                   '/reg_' + str(ipoint), 'rb'))
        except:
            reg = keras.models.load_model(pathAux + 'TRAINED_MODELS/' + var.upper() + '/' + methodName +
                                          '/reg_' + str(ipoint) + '.h5', compile=False)

        if var == 'pcp':
            try:
                clf = pickle.load(open(pathAux + 'TRAINED_MODELS/' + var.upper() + '/' + methodName +
                                       '/clf_' + str(ipoint), 'rb'))
            except:
                clf = keras.models.load_model(pathAux + 'TRAINED_MODELS/' + var.upper() + '/' + methodName +
                                              '/clf_' + str(ipoint) + '.h5', compile=False)

        # Prints for monitoring
        if ipoint_local_index % 1==0:
            print('--------------------')
            print('ichunk:	', ichunk, '/', n_chunks)
            print('downscaling', var, methodName, scene, model, round(100*ipoint_local_index/npoints_ichunk, 2), '%')


        # Prepare X_test shape
        if var+'_'+methodName not in methods_using_preds_from_whole_grid:
            X_test_ipoint = grids.interpolate_predictors(X_test, i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_mode)
        elif methodName not in ['CNN', 'CNN-SYN']:
            X_test_ipoint = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
        else:
            X_test_ipoint = X_test

        # Check missing predictors, remove them and recalibrate
        if recalibrating_when_missing_preds == True:
            missing_preds = np.unique(np.where(np.isnan(X_test_ipoint))[1])
            if len(missing_preds) != 0:
                print('Recalibrating because of missing predictors:', missing_preds%npreds)
                y_train_ipoint = y_train[:, ipoint]

                # Interpolate valid_preds
                valid_preds = [x for x in range(X_test_ipoint.shape[1]) if x not in missing_preds]
                if var + '_' + methodName not in methods_using_preds_from_whole_grid:
                    X_train_ipoint = grids.interpolate_predictors(X_train[:, valid_preds, :, :], i_4nn[ipoint],
                                                                  j_4nn[ipoint], w_4nn[ipoint], interp_mode)
                elif methodName not in ['CNN', 'CNN-SYN']:
                    X_train_ipoint = X_train[:, valid_preds, :, :].reshape(X_train.shape[0], len(valid_preds), -1)
                else:
                    X_train_ipoint = X_train[:, valid_preds, :, :]

                # Check for missing predictands and remove them (if no missing predictors there is no need to check on
                # predictands, because classifier/regressor are already trained
                valid = np.where(y_train_ipoint < special_value)[0]
                if valid.size < 30:
                    exit('Not enough valid predictands to train')
                if valid.size != y_train_ipoint.size:
                    X_train_ipoint = X_train_ipoint[valid]
                    y_train_ipoint = y_train_ipoint[valid]

                # Train regressors and classifiers without missing predictors or predictands
                reg, clf = TF_lib.train_point(var, methodName, X_train_ipoint, y_train_ipoint, ipoint)

                # Remove missing predictors from X_test
                X_test_ipoint = X_test_ipoint[:, valid_preds]

        # Check for days with missing predictors to set them to np.nan later
        elif recalibrating_when_missing_preds == False:
            idays_with_missing_preds = np.unique(np.where(np.isnan(X_test_ipoint))[0])
            if len(idays_with_missing_preds) != 0:
                X_test_ipoint[np.isnan(X_test_ipoint)] = special_value
                # if idays_with_missing_preds.shape == est[:, ipoint_local_index].shape:
                #     exit('there is at least one predictor which is missing in all days. You should set '
                #           'recalibrating_when_missing_preds to True at settings')

        # Apply downscaling
        if methodName in ['CNN', 'CNN-SYN']:
            X_test_ipoint = np.swapaxes(np.swapaxes(X_test_ipoint, 1, 2), 2, 3)
        if var == 'pcp':
            est[:, ipoint_local_index] = down_point.pcp_TF(methodName, X_test_ipoint, clf, reg)
        else:
            est[:, ipoint_local_index] = down_point.t_TF(X_test_ipoint, reg)

        # Set to np.nan days with missing predictors for TF methods
        if (recalibrating_when_missing_preds == False) and (len(idays_with_missing_preds) != 0):
            est[idays_with_missing_preds, ipoint_local_index] = special_value

    # Undo converssion
    est = est.astype('float64') / 100.

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
            periodFilename= historicalPeriodFilename
            scene_dates = historical_dates
        else:
            periodFilename= sspPeriodFilename
            scene_dates = ssp_dates
        # Read dates (can be different for different calendars)
        path = '../input_data/models/'
        ncVar = modNames[var]
        modelName, modelRun = model.split('_')[0], model.split('_')[1]
        filename = ncVar + '_' + modelName + '_' + scene +'_'+ modelRun + '_'+periodFilename+ '.nc'
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
