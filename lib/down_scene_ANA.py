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

def downscale_chunk(targetVar, methodName, family, mode, fields, scene, model, iproc=0, nproc=1):
    """
    This function goes through all days.
    It previously divides a scene in nproc chunks and processes the chunk number iproc in parallel.
    The result is saved as npy file (each chunk is one file).
    """


    # Define path
    pathOut='../tmp/'+targetVar+'_'+methodName+'_'+ model + '_' + scene + '/'

    # Define analogy_mode
    analogy_mode = methodName.split('-')[1]

    # Parent process reads all data, broadcasts to the other processes and creates paths for results
    if iproc == 0:
        print(scene, model, targetVar, methodName)
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)
        # Read data and converts obs to uint16 or int16 to save memory
        obs = read.hres_data(targetVar, period='training')['data']
        obs = (100 * obs).astype(predictands_codification[targetVar]['type'])
        i_4nn = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/i_4nn.npy')
        j_4nn = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/j_4nn.npy')
        w_4nn = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/w_4nn.npy')
        pred_calib = None
        saf_calib = None
        var_calib = None
        pred_scene = None
        saf_scene = None
        var_scene = None
        centroids = None
        coef = None
        intercept = None
        corr = None

        if 'pred' in fields:
            pred_calib = np.load(pathAux+'STANDARDIZATION/PRED/'+targetVar+'_training.npy')
            pred_calib = pred_calib.astype('float32')
        if 'saf' in fields:
            saf_calib = np.load(pathAux+'STANDARDIZATION/SAF/'+targetVar+'_training.npy')
            saf_calib = saf_calib.astype('float32')
            saf_calib = saf_calib.reshape(saf_calib.shape[0], -1)
            W = W_saf[np.newaxis, :]
            W = np.repeat(W, saf_calib.shape[0], axis=0)
            saf_calib *= W
            infile = open(pathAux + 'PCA/pca', 'rb')
            pca = pickle.load(infile)
            infile.close()
            saf_calib = pca.transform(saf_calib)
        if 'var' in fields:
            var_calib = np.load(pathAux+'STANDARDIZATION/VAR/'+targetVar+'_training.npy')

        if methodName == 'MLR-WT':
            centroids = np.load(pathAux+'WEATHER_TYPES/centroids.npy')
            coef = np.load(pathAux+'COEFFICIENTS/'+targetVar+'_'+methodName+'_coefficients.npy')
            intercept = np.load(pathAux+'COEFFICIENTS/'+targetVar+'_'+methodName+'_intercept.npy')
        elif analogy_mode == 'LOC':
            centroids = np.load(pathAux+'WEATHER_TYPES/centroids.npy')
            corr = np.load(pathAux+'COEFFICIENTS/'+targetVar+'_'+methodName+'_correlations.npy')
            corr[np.isnan(corr)]=0
            corr = abs(corr) >= anal_corr_th_dict[targetVar]
            print('corr_th', anal_corr_th_dict[targetVar], 100.*np.count_nonzero(corr)/corr.size,'%')

        # Set scene dates and predictors
        if scene == 'TESTING':
            scene_dates = testing_dates

            if 'pred' in fields:
                pred_scene = np.load(pathAux+'STANDARDIZATION/PRED/'+targetVar+'_testing.npy')
                pred_scene = pred_scene.astype('float32')
            if 'saf' in fields:
                saf_scene = np.load(pathAux+'STANDARDIZATION/SAF/'+targetVar+'_testing.npy')
                saf_scene = saf_scene.astype('float32')
                saf_scene = saf_scene.reshape(saf_scene.shape[0], -1)
                W = W_saf[np.newaxis, :]
                W = np.repeat(W, saf_scene.shape[0], axis=0)
                saf_scene *= W
                infile = open(pathAux + 'PCA/pca', 'rb')
                pca = pickle.load(infile)
                infile.close()
                saf_scene = pca.transform(saf_scene)
            if 'var' in fields:
                var_scene = np.load(pathAux+'STANDARDIZATION/VAR/'+targetVar+'_testing.npy')
        else:
            if scene == 'historical':
                years = historical_years
            else:
                years = ssp_years

            # Read dates (can be different for different calendars)
            aux = read.lres_data(targetVar, 'var', model=model, scene=scene)
            scene_dates = aux['times']
            idates = [i for i in range(len(scene_dates)) if scene_dates[i].year >= years[0] and scene_dates[i].year <= years[1]]
            scene_dates = list(np.array(scene_dates)[idates])
            if 'pred' in fields:
                pred_scene = read.lres_data(targetVar, 'pred', model=model, scene=scene)['data'][idates]
                pred_scene = standardization.standardize(targetVar, pred_scene, model, 'pred')
                pred_scene = pred_scene.astype('float32')
                del aux
            if 'saf' in fields:
                saf_scene = read.lres_data(targetVar, 'saf', model=model, scene=scene)['data'][idates]
                saf_scene = standardization.standardize(targetVar, saf_scene, model, 'saf')
                saf_scene = saf_scene.astype('float32')
                saf_scene = saf_scene.reshape(saf_scene.shape[0], -1)
                W = W_saf[np.newaxis, :]
                W = np.repeat(W, saf_scene.shape[0], axis=0)
                saf_scene *= W
                infile = open(pathAux + 'PCA/pca', 'rb')
                pca = pickle.load(infile)
                infile.close()
                global_days_with_nan = np.where(np.isnan(saf_scene))[0]
                global_days_with_nan = list(dict.fromkeys(global_days_with_nan))
                saf_scene[global_days_with_nan] = predictands_codification[targetVar]['special_value']
                saf_scene = pca.transform(saf_scene)
                saf_scene[global_days_with_nan] = predictands_codification[targetVar]['special_value']
            if 'var' in fields:
                var_scene = read.lres_data(targetVar, 'var', model=model, scene=scene)['data'][idates]

        # Trick the program so it uses 'var' (VAR) as 'saf'
        if analogy_mode == 'VAR':
            saf_calib, saf_scene = var_calib, var_scene

        # # Check for missing data at SAFs
        # if np.where(np.isnan(saf_scene))[0].size != 0:
        #     exit('\n-------------------------------------------------------\n'
        #          'There are missing data at Synoptic Analogy Fields (SAFs)\n'
        #          'The program is not prepared to handle np.nan at SAFs (synoptic distances to all days would be infinite):'
        #          '  - If the problem exists only in one model, maybe that model should not be used.'
        #          '  - If the problem exists in many models, a different set of SAFs must be used.')

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
        centroids = None
        coef = None
        intercept = None
        corr = None

    # Share data with all subprocesses
    if nproc > 1:
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
        centroids = MPI.COMM_WORLD.bcast(centroids, root=0)
        coef = MPI.COMM_WORLD.bcast(coef, root=0)
        intercept = MPI.COMM_WORLD.bcast(intercept, root=0)
        corr = MPI.COMM_WORLD.bcast(corr, root=0)

    # Create chunks
    scene_ndates = len(scene_dates)
    global_scene_dates = scene_dates
    n_chunks = nproc
    len_chunk = int(math.ceil(float(scene_ndates) / n_chunks))
    dates_chunk = []
    for ichunk in range(n_chunks):
        dates_chunk.append(scene_dates[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    len_chunk = []
    for ichunk in range(n_chunks):
        len_chunk.append(len(dates_chunk[ichunk]))

    if 'saf' in fields:
        saf_scene_chunk = []
    if 'pred' in fields:
        pred_scene_chunk = []
    if 'var' in fields:
        var_scene_chunk = []
    idates = []
    for ichunk in range(n_chunks):
        aux_idates = []
        for date in dates_chunk[ichunk]:
            aux_idates.append(scene_dates.index(date))
        idates.append(aux_idates)
        if 'saf' in fields:
            saf_scene_chunk.append(saf_scene[aux_idates])
        if 'pred' in fields:
            pred_scene_chunk.append(pred_scene[aux_idates])
        if 'var' in fields:
            var_scene_chunk.append(var_scene[aux_idates])
    ichunk = iproc
    scene_dates = dates_chunk[ichunk]
    scene_ndates = len_chunk[ichunk]
    if 'saf' in fields:
        saf_scene = saf_scene_chunk[ichunk]
    if 'pred' in fields:
        pred_scene = pred_scene_chunk[ichunk]
    if 'var' in fields:
        var_scene = var_scene_chunk[ichunk]
    est = np.zeros((scene_ndates, hres_npoints[targetVar]))
    est = est.astype(predictands_codification[targetVar]['type'])

    # Goes throuch all dates
    for idate in range(scene_ndates):
        date = scene_dates[idate]
        if idate % 1==0:
            print('----------------------------------------')
            print(scene, model, targetVar, methodName)
            print('ichunk:	', ichunk, '/', n_chunks)
            print('day:', idate, '/', scene_ndates, '(', int(100*idate/scene_ndates), '% )')
            # print(date.date())

        # Selects fields for idates
        pred_scene_idate, saf_scene_idate, var_scene_idate = None, None, None
        if 'saf' in fields:
            saf_scene_idate = saf_scene[idate]
        if 'pred' in fields:
            pred_scene_idate = pred_scene[idate]
        if 'var' in fields:
            var_scene_idate = var_scene[idate]

        # Trick the program so it uses 'var' (VAR) as 'saf'
        if analogy_mode == 'VAR':
            saf_scene_idate = var_scene_idate

        # If saf_scene does not contain nans, process. Otherwise, fill est with nans
        if np.count_nonzero(np.isnan(saf_scene[idate])) == 0:
            est[idate] = down_day.down_day(targetVar, pred_scene_idate, saf_scene_idate, var_scene_idate, pred_calib, saf_calib,
                                    var_calib, obs, corr, coef, intercept, centroids, i_4nn, j_4nn, w_4nn, methodName)
        else:
            est[idate] = 100 * predictands_codification[targetVar]['special_value']

    # Undo converssion
    est = est.astype('float64') / 100.
    # print(est[0:10], est.shape)

    # Saves results
    np.save(pathOut + 'ichunk_'+ str(ichunk) + '.npy', est)

########################################################################################################################
def collect_chunks(targetVar, methodName, family, mode, fields, scene, model, n_chunks=1):
    """
    This function collects the results of downscale_chunk() and saves them into a final single file.
    """
    print('--------------------------------------')
    print(scene, model, 'collect chunks', n_chunks)

    # Create empty array and accumulate
    est = np.zeros((0, hres_npoints[targetVar]))
    for ichunk in range(n_chunks):
        path ='../tmp/'+targetVar+'_'+methodName+'_'+ model + '_' + scene + '/'
        filename = path + 'ichunk_' + str(ichunk) + '.npy'
        est = np.append(est, np.load(filename), axis=0)
    shutil.rmtree(path)

    # Gets scene dates
    if scene == 'TESTING':
        scene_dates = testing_dates
    else:
        if scene == 'historical':
            periodFilename = historicalPeriodFilename
            scene_dates = historical_dates
        else:
            periodFilename = sspPeriodFilename
            scene_dates = ssp_dates
        # Read dates (can be different for different calendars)
        path = '../input_data/models/'
        ncVar = modNames[targetVar]
        modelName, modelRun = model.split('_')[0], model.split('_')[1]
        filename = ncVar + '_' + modelName + '_' + scene +'_'+ modelRun + '_'+periodFilename + '.nc'
        model_dates = np.ndarray.tolist(read.netCDF(path, filename, ncVar)['times'])
        aux = np.zeros((len(scene_dates), hres_npoints[targetVar]))
        aux[:] = np.nan
        idates = [i for i in range(len(scene_dates)) if scene_dates[i] in model_dates]
        aux[idates] = est
        est = aux
        del aux


    # Save to file
    if experiment == 'PSEUDOREALITY':
        pathOut = '../results/'+experiment+'/'+ GCM_longName + '_' + RCM + '/'+targetVar.upper()+'/'+methodName+'/daily_data/'
    else:
        pathOut = '../results/'+experiment+'/'+targetVar.upper()+'/'+methodName+'/daily_data/'

    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Save results
    hres_lats = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/hres_lats.npy')
    hres_lons = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/hres_lons.npy')

    # Set units
    units = predictands_units[targetVar]
    if units == None:
        units = ''

    if split_mode[:4] == 'fold':
        fold_sufix = '_' + split_mode
    else:
        fold_sufix = ''

    # Special values are set to nan
    warnings.filterwarnings("ignore", message="invalid value encountered in less")
    est[np.abs(est-predictands_codification[targetVar]['special_value']) < 0.01] = np.nan
    print('-------------------------------------------------------------------------')
    print('results contain', 100*int(np.where(np.isnan(est))[0].size/est.size), '% of nans')
    print('-------------------------------------------------------------------------')

    # Force to theoretical range
    minAllowed, maxAllowed = predictands_range[targetVar]['min'], predictands_range[targetVar]['max']
    if  minAllowed != None:
        est[est < 100*minAllowed] == 100*minAllowed
    if  maxAllowed != None:
        est[est > 100*maxAllowed] == 100*maxAllowed

    # Save data to netCDF file
    write.netCDF(pathOut, model+'_'+scene+fold_sufix+'.nc', targetVar, est, units, hres_lats, hres_lons, scene_dates, regular_grid=False)

    # If using k-folds, join them
    if split_mode == 'fold5':
        aux_lib.join_kfolds(targetVar, methodName, family, mode, fields, scene, model, units, hres_lats, hres_lons)


########################################################################################################################

if __name__=="__main__":

    nproc = MPI.COMM_WORLD.Get_size()         # Size of communicator
    iproc = MPI.COMM_WORLD.Get_rank()         # Ranks in communicator
    inode = MPI.Get_processor_name()          # Node where this MPI process runs
    targetVar = sys.argv[1]
    methodName = sys.argv[2]
    family = sys.argv[3]
    mode = sys.argv[4]
    fields = sys.argv[5]
    scene = sys.argv[6]
    model = sys.argv[7]

    downscale_chunk(targetVar, methodName, family, mode, fields, scene, model, iproc, nproc)
    MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step
    if iproc==0:
        collect_chunks(targetVar, methodName, family, mode, fields, scene, model, nproc)