import sys
sys.path.append(('../config/'))
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


def downscale_chunk(var, methodName, family, mode, fields, scene, model, iproc=0, nproc=1):
    """
    This function goes through all points (regression).
    The result is saved as npy file (each chunk is one file).
    """

    # create chunks
    n_chunks = nproc
    len_chunk = int(math.ceil(float(hres_npoints) / n_chunks))
    points_chunk = []
    for ichunk in range(n_chunks):
        points_chunk.append(list(range(hres_npoints))[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    ichunk = iproc
    npoints_ichunk = len(points_chunk[ichunk])

    # Define paths
    pathTmp = '../tmp/TRAINED_'+ '_'.join((var, methodName, scene, model)) + '/'
    pathOut = '../tmp/ESTIMATED_' + '_'.join((var, methodName, scene, model)) + '/'

    # Parent process reads all data, broadcasts to the other processes and creates paths for results
    if iproc == 0:
        print(var, methodName, scene, model)
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Read data and converts obs to uint16 or int16 to save memory
        obs = read.hres_data(var, period='training')['data']
        obs = (100 * obs).astype(predictands_codification[var]['type'])
        i_4nn = np.load(pathAux+'ASSOCIATION/'+interp_dict[mode]+'/i_4nn.npy')
        j_4nn = np.load(pathAux+'ASSOCIATION/'+interp_dict[mode]+'/j_4nn.npy')
        w_4nn = np.load(pathAux+'ASSOCIATION/'+interp_dict[mode]+'/w_4nn.npy')
        pred_calib = None
        saf_calib = None
        var_calib = None
        pred_scene = None
        saf_scene = None
        var_scene = None

        if 'pred' in fields:
            pred_calib = np.load(pathAux+'STANDARDIZATION/PRED/'+var[0]+'_training.npy')
            pred_calib = pred_calib.astype('float32')
        if 'saf' in fields:
            saf_calib = np.load(pathAux+'STANDARDIZATION/SAF/'+var[0]+'_training.npy')
            saf_calib = saf_calib.astype('float32')
        if 'var' in fields:
            var_calib = np.load(pathAux+'STANDARDIZATION/VAR/'+var+'_training.npy')

        # Split clf/reg in chunks
        if var == 'pcp':
            trained_model_names = ['clf', 'reg', ]
        else:
            trained_model_names = ['reg', ]

        for trained_model_name in trained_model_names:

            # Load trained model
            infile = open(pathAux + 'TRAINED_MODELS/' + var.upper() + '/' + methodName + '_' +
                          trained_model_name, 'rb')
            trained_model = pickle.load(infile)
            infile.close()

            # Save trained_model chunks so each iproc reads its own chunck
            for i in range(nproc):
                outfile = open(pathTmp + 'trained_' + trained_model_name + '_' + str(i), 'wb')
                pickle.dump(trained_model[points_chunk[i]], outfile)
                outfile.close()

        # Set scene dates and predictors
        if scene == 'TESTING':
            scene_dates = testing_dates

            if 'pred' in fields:
                pred_scene = np.load(pathAux+'STANDARDIZATION/PRED/'+var[0]+'_testing.npy')
                pred_scene = pred_scene.astype('float32')
            if 'saf' in fields:
                saf_scene = np.load(pathAux+'STANDARDIZATION/SAF/'+var[0]+'_testing.npy')
                saf_scene = saf_scene.astype('float32')
            if 'var' in fields:
                var_scene = np.load(pathAux+'STANDARDIZATION/VAR/'+var+'_testing.npy')
        else:
            if scene == 'historical':
                years = historical_years
                periodFilename = historicalPeriodFilename
            else:
                years = ssp_years
                periodFilename= rcpPeriodFilename

            # Read dates (can be different for different calendars)
            scene_dates = read.netCDF('../input_data/models/', 'psl_' + model + '_' + scene +'_'+ modelRealizationFilename + '_'+periodFilename+ '.nc',
                            'psl')['times']
            idates = [i for i in range(len(scene_dates)) if scene_dates[i].year >= years[0] and scene_dates[i].year <= years[1]]
            scene_dates = list(np.array(scene_dates)[idates])
            if 'pred' in fields:
                pred_scene = read.lres_data(var, 'pred', model=model, scene=scene)['data'][idates]
                pred_scene = standardization.standardize(var[0], pred_scene, model, 'pred')
                pred_scene = pred_scene.astype('float32')
            if 'saf' in fields:
                saf_scene = read.lres_data(var, 'saf', model=model, scene=scene)['data'][idates]
                saf_scene = standardization.standardize(var[0], saf_scene, model, 'saf')
                saf_scene = saf_scene.astype('float32')
            if 'var' in fields:
                var_scene = read.lres_data(var, 'var', model=model, scene=scene)['data'][idates]

    # Declares variables for the other processes
    else:
        obs = None
        scene_dates = None
        i_4nn = None
        j_4nn = None
        w_4nn = None
        pred_calib = None
        saf_calib = None
        var_calib = None
        pred_scene = None
        saf_scene = None
        var_scene = None

    # Share data with all subprocesses
    if nproc>1:
        obs = MPI.COMM_WORLD.bcast(obs, root=0)
        scene_dates = MPI.COMM_WORLD.bcast(scene_dates, root=0)
        i_4nn = MPI.COMM_WORLD.bcast(i_4nn, root=0)
        j_4nn = MPI.COMM_WORLD.bcast(j_4nn, root=0)
        w_4nn = MPI.COMM_WORLD.bcast(w_4nn, root=0)
        pred_calib = MPI.COMM_WORLD.bcast(pred_calib, root=0)
        saf_calib = MPI.COMM_WORLD.bcast(saf_calib, root=0)
        var_calib = MPI.COMM_WORLD.bcast(var_calib, root=0)
        pred_scene = MPI.COMM_WORLD.bcast(pred_scene, root=0)
        saf_scene = MPI.COMM_WORLD.bcast(saf_scene, root=0)
        var_scene = MPI.COMM_WORLD.bcast(var_scene, root=0)
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step

    # Load regressors and classifiers of ichunk
    infile = open(pathTmp + 'trained_reg_' + str(ichunk), 'rb')
    trained_reg = pickle.load(infile)
    infile.close()
    if var == 'pcp':
        infile = open(pathTmp + 'trained_clf_' + str(ichunk), 'rb')
        trained_clf = pickle.load(infile)
        infile.close()
    else:
        trained_clf = trained_reg.size * [None]

    if nproc > 1:
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step
    if iproc == 0:
        shutil.rmtree(pathTmp)

    # Create empty array for results
    est = np.zeros((len(scene_dates), npoints_ichunk))
    est = est.astype(predictands_codification[var]['type'])
    special_value = int(100 * predictands_codification[var]['special_value'])

    # Goes through all points of the chunk
    for ipoint in points_chunk[ichunk]:
        ipoint_local_index = points_chunk[ichunk].index(ipoint)

        # Selects trained models for ipoint
        reg_ipoint, clf_ipoint = trained_reg[ipoint_local_index], trained_clf[ipoint_local_index]

        # Prints for monitoring
        if ipoint_local_index % 1==0:
            print('--------------------')
            print('ichunk:	', ichunk, '/', n_chunks)
            print('downscaling', var, methodName, scene, model, round(100*ipoint_local_index/npoints_ichunk, 2), '%')

        # Interpolate to ipoint
        X_test = grids.interpolate_predictors(pred_scene, i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_dict[mode])

        # Check missing predictors, remove them and recalibrate
        if recalibrating_when_missing_preds == True:
            missing_preds = np.unique(np.where(np.isnan(X_test))[1])
            if len(missing_preds) != 0:
                print('Recalibrating because of missing predictors:', missing_preds)
                Y_train = obs[:, ipoint]

                # Interpolate valid_preds
                valid_preds = [x for x in range(X_test.shape[1]) if x not in missing_preds]
                X_train = grids.interpolate_predictors(pred_calib[:, valid_preds, :, :], i_4nn[ipoint],
                                                       j_4nn[ipoint], w_4nn[ipoint], interp_dict[mode])

                # Check for missing predictands and remove them (if no missing predictors there is no need to check on
                # predictands, because classifier/regressor are already trained
                valid = np.where(Y_train < special_value)[0]
                if valid.size < 30:
                    exit('Not enough valid predictands to train')
                if valid.size != Y_train.size:
                    X_train = X_train[valid]
                    Y_train = Y_train[valid]

                # Train regressors and classifiers without missing predictors or predictands
                reg_ipoint, clf_ipoint = TF_lib.train_ipoint(var, methodName, X_train, Y_train, ipoint)

                # Remove missing predictors from X_test
                X_test = X_test[:, valid_preds]

        # Check for days with missing predictors to set them to np.nan later
        elif recalibrating_when_missing_preds == False:
            idays_with_missing_preds = np.unique(np.where(np.isnan(X_test))[0])
            if len(idays_with_missing_preds) != 0:
                X_test[np.isnan(X_test)] = special_value
                # if idays_with_missing_preds.shape == est[:, ipoint_local_index].shape:
                #     exit('there is at least one predictor which is missing in all days. You should set '
                #           'recalibrating_when_missing_preds to True at settings')

        # Apply downscaling
        if var == 'pcp':
            est[:, ipoint_local_index] = down_point.pcp_TF(methodName, X_test, clf_ipoint, reg_ipoint)
        else:
            est[:, ipoint_local_index] = down_point.t_TF(X_test, reg_ipoint)

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
            periodFilename= rcpPeriodFilename
            scene_dates = ssp_dates
        # Read dates (can be different for different calendars)
        path = '../input_data/models/'
        filename = 'psl_' + model + '_' + scene +'_'+ modelRealizationFilename + '_'+periodFilename+ '.nc'
        model_dates = np.ndarray.tolist(read.netCDF(path, filename, 'psl')['times'])
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
        aux = np.zeros((len(scene_dates), hres_npoints))
        aux[:] = np.nan
        idates = [i for i in range(len(scene_dates)) if scene_dates[i] in model_dates]
        aux[idates] = est
        est = aux
        del aux
        pathOut = '../results/'+experiment+'/'+ GCM_longName + '_' + RCM + '/'+var.upper()+'/'+methodName+'/daily_data/'
    else:
        aux = np.zeros((len(scene_dates), hres_npoints))
        aux[:] = np.nan
        idates = [i for i in range(len(scene_dates)) if scene_dates[i] in model_dates]
        aux[idates] = est
        est = aux
        del aux
        pathOut = '../results/'+experiment+'/'+var.upper()+'/'+methodName+'/daily_data/'

    # Save results
    hres_lats = np.load(pathAux+'ASSOCIATION/'+interp_dict[mode]+'/hres_lats.npy')
    hres_lons = np.load(pathAux+'ASSOCIATION/'+interp_dict[mode]+'/hres_lons.npy')

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
