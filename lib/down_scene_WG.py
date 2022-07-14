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

def downscale_chunk_WG_PDF(targetVar, methodName, family, mode, fields, scene, model, iproc, nproc):
    """
    This function goes through all points (regression).
    The result is saved as npy file (each chunk is one file).
    """

    targetGroup = targetGroups_dict[targetVar]
    aggMonths = aggregation_pcp_WG_PDF

    # create chunks
    n_chunks = nproc
    len_chunk = int(math.ceil(float(hres_npoints[targetVar]) / n_chunks))
    points_chunk = []
    for ichunk in range(n_chunks):
        points_chunk.append(list(range(hres_npoints[targetVar]))[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    ichunk = iproc
    npoints_ichunk = len(points_chunk[ichunk])

    # Define paths
    pathTmp = '../tmp/TRAINED_'+ '_'.join((targetVar, methodName, scene, model)) + '/'
    pathOut = '../tmp/ESTIMATED_' + '_'.join((targetVar, methodName, scene, model)) + '/'

    # Parent process reads all data, broadcasts to the other processes and creates paths for results
    if iproc == 0:
        print(targetVar, methodName, scene, model)
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Read data
        i_4nn = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/i_4nn.npy')
        j_4nn = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/j_4nn.npy')
        w_4nn = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/w_4nn.npy')

        # Split eg in chunks
        trained_model_names = ['PARAM1_reg', 'PARAM2_reg', ]

        for trained_model_name in trained_model_names:

            # Load trained model
            infile = open(pathAux + 'TRAINED_'+methodName+'/' + targetVar.upper() + '/' + methodName + '_' +
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
            var_scene = np.load(pathAux+'STANDARDIZATION/VAR/'+targetVar+'_testing.npy')
        else:
            if scene == 'historical':
                years = historical_years
                periodFilename = historicalPeriodFilename
            else:
                years = ssp_years
                periodFilename = sspPeriodFilename

            # Read dates (can be different for different calendars)
            ncVar = modNames[targetVar]
            modelName, modelRun = model.split('_')[0], model.split('_')[1]
            scene_dates = read.netCDF('../input_data/models/', ncVar + '_' + modelName + '_' + scene +'_'+ modelRun + '_'+periodFilename + '.nc',
                            ncVar)['times']
            idates = [i for i in range(len(scene_dates)) if scene_dates[i].year >= years[0] and scene_dates[i].year <= years[1]]
            scene_dates = list(np.array(scene_dates)[idates])
            var_scene = read.lres_data(targetVar, 'var', model=model, scene=scene)['data'][idates]

    # Declares variables for the other processes
    else:
        scene_dates = None
        i_4nn = None
        j_4nn = None
        w_4nn = None
        var_scene = None

    # Share data with all subprocesses
    if nproc>1:
        scene_dates = MPI.COMM_WORLD.bcast(scene_dates, root=0)
        i_4nn = MPI.COMM_WORLD.bcast(i_4nn, root=0)
        j_4nn = MPI.COMM_WORLD.bcast(j_4nn, root=0)
        w_4nn = MPI.COMM_WORLD.bcast(w_4nn, root=0)
        var_scene = MPI.COMM_WORLD.bcast(var_scene, root=0)
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step


    # Interpolate var_scene
    var_scene_interp = np.zeros((var_scene.shape[0], npoints_ichunk))
    for ipoint in range(npoints_ichunk):
        ipoint_global = points_chunk[ichunk][ipoint]
        if ipoint % 1000 == 0:
            print('interpolating', ipoint)
        var_scene_interp[:, ipoint] = grids.interpolate_predictors(var_scene,
                                  i_4nn[ipoint_global], j_4nn[ipoint_global], w_4nn[ipoint_global], interp_mode)[:, 0]
    # np.save(pathTmp+'var_scene_interp', var_scene_interp)
    # var_scene_interp = np.load(pathTmp+'var_scene_interp.npy')
    del var_scene

    # Load trained models of ichunk
    infile = open(pathTmp + 'trained_PARAM1_reg_' + str(ichunk), 'rb')
    PARAM1_reg = pickle.load(infile)
    infile.close()
    infile = open(pathTmp + 'trained_PARAM2_reg_' + str(ichunk), 'rb')
    PARAM2_reg = pickle.load(infile)
    infile.close()

    if nproc > 1:
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step
    if iproc == 0:
        shutil.rmtree(pathTmp)

    # Create empty array for results
    scene_ndates = len(scene_dates)
    est = np.zeros((scene_ndates, npoints_ichunk))

    # Define times
    times = scene_dates
    years = set([x.year for x in times])
    nYears = len(years)
    n_blocks = 12*nYears

    # Select points from chunk
    X = var_scene_interp
    del var_scene_interp

    # Create empty arrays for results
    PARAM1_X = np.zeros((scene_ndates, npoints_ichunk))
    PARAM2_X = np.zeros((scene_ndates, npoints_ichunk))

    # Calculate parameters (X, low_res) for each month (with the previous and following months)
    i_block = 0
    for year in years:
        for month in range(1, 13):
            print('i_block', i_block, '/', n_blocks)

            if aggMonths == 1:
                # For monthly aggregations
                idates = [i for i in range(len(times)) if (times[i].year == year) and (times[i].month == month)]
            else:       
                # For 3-months aggregations
                if month == 1:
                    idates = [i for i in range(len(times)) if
                              ((times[i].year == year - 1) and (times[i].month == 12)) or
                              ((times[i].year == year) and (times[i].month == month)) or
                              ((times[i].year == year) and (times[i].month == month + 1))]
                elif month == 12:
                    idates = [i for i in range(len(times)) if
                              ((times[i].year == year) and (times[i].month == month - 1)) or
                              ((times[i].year == year) and (times[i].month == month)) or
                              ((times[i].year == year + 1) and (times[i].month == 1))]
                else:
                    idates = [i for i in range(len(times)) if
                              ((times[i].year == year) and (times[i].month == month - 1)) or
                              ((times[i].year == year) and (times[i].month == month)) or
                              ((times[i].year == year) and (times[i].month == month + 1))]

            # Select dates and calculate PARAM1 and PARAM2
            X_month = X[idates]

            if targetVar != 'pr':
                # mean and std
                PARAM1_X[idates] = np.nanmean(X_month, axis=0)
                PARAM2_X[idates] = np.nanstd(X_month, axis=0)

            else:
                # R01 and mean
                PARAM1_X[idates] = np.nanmean(X_month>=1, axis=0)
                PARAM2_X[idates] = np.nanmean(X_month, axis=0)

            i_block +=1

    del X

    # Goes through all points of the chunk
    for ipoint in range(npoints_ichunk):

        # Prints for monitoring
        if ipoint % 100 == 0:
            print('--------------------')
            print('ichunk:	', ichunk, '/', n_chunks)
            print('downscaling', targetVar, methodName, scene, model, round(100*ipoint/npoints_ichunk, 2), '%')

        # Select ipoint
        PARAM1_X_ipoint = PARAM1_X[:, ipoint]
        PARAM2_X_ipoint = PARAM2_X[:, ipoint]

        # Selects trained models for ipoint
        PARAM1_reg_ipoint = PARAM1_reg[ipoint]
        PARAM2_reg_ipoint = PARAM2_reg[ipoint]

        # Calculate estimated parameters with the regressors
        PARAM1_est_ipoint = PARAM1_reg_ipoint.predict(PARAM1_X_ipoint.reshape(-1, 1))
        PARAM2_est_ipoint = PARAM2_reg_ipoint.predict(PARAM2_X_ipoint.reshape(-1, 1))

        # Estimate daily data
        if targetVar != 'pr':
            # Normal distribution
            PARAM2_est_ipoint[PARAM2_est_ipoint <= 0] = 0.001
            est[:, ipoint] = np.random.normal(PARAM1_est_ipoint, PARAM2_est_ipoint)[:, 0]

        else:

            # Exponential distribution
            PARAM1_est_ipoint[PARAM1_est_ipoint < 0] = 0
            PARAM2_est_ipoint[PARAM2_est_ipoint < 0] = 0
            nDays = est[:, ipoint].size
            r01 = PARAM1_est_ipoint[:, 0]
            mean = PARAM2_est_ipoint[:, 0]

            # Get sdii
            r01[r01 == 0] = 99999
            sdii = mean / r01
            r01[r01 == 99999] = 0
            sdii[mean == 0] = 0.0001

            # Exponential distribution shifted
            sdii[sdii <= 1] = 1.001
            est_ipoint = 1 + np.random.exponential(sdii-1, size=nDays)

            # Force dry days
            isdry = (r01 < np.random.uniform(size=nDays))
            est_ipoint[isdry == True] = 0

            est[:, ipoint] = est_ipoint

    # Saves results to file
    np.save(pathOut + 'ichunk_' + str(ichunk) + '.npy', est)

########################################################################################################################
def downscale_chunk_WG_NMM(targetVar, methodName, family, mode, fields, scene, model, iproc, nproc):
    """This function calculates, for each point in the chunk, a first order markov model with transition probabilities
    conditioned by low resolution precipitation. Then, for wet days, intensity is taken from an ECDF also conditioned"""

    targetGroup = targetGroups_dict[targetVar]
    thresholds = thresholds_WG_NMM
    nthresholds = len(thresholds)

    # create chunks
    n_chunks = nproc
    len_chunk = int(math.ceil(float(hres_npoints[targetVar]) / n_chunks))
    points_chunk = []
    for ichunk in range(n_chunks):
        points_chunk.append(list(range(hres_npoints[targetVar]))[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    ichunk = iproc
    npoints_ichunk = len(points_chunk[ichunk])

    # Define paths
    pathTmp = '../tmp/TRAINED_'+ '_'.join((targetVar, methodName, scene, model)) + '/'
    pathOut = '../tmp/ESTIMATED_' + '_'.join((targetVar, methodName, scene, model)) + '/'

    # Parent process reads all data, broadcasts to the other processes and creates paths for results
    if iproc == 0:
        print(targetVar, methodName, scene, model)
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Read data
        i_4nn = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/i_4nn.npy')
        j_4nn = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/j_4nn.npy')
        w_4nn = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/w_4nn.npy')

        # Split eg in chunks
        trained_model_names = ['P00', 'P01', 'P10', 'P11', 'ECDF_pcp', ]

        for trained_model_name in trained_model_names:

            # Load trained model
            trained_model = np.load(pathAux + 'TRAINED_'+methodName+'/'+targetVar.upper()+'/'+trained_model_name+'.npy')

            # Save trained_model chunks so each iproc reads its own chunck
            for i in range(nproc):
                np.save(pathTmp + 'trained_' + trained_model_name + '_' + str(i), trained_model[points_chunk[i]])

        # Set scene dates and predictors
        if scene == 'TESTING':
            scene_dates = testing_dates
            var_scene = np.load(pathAux+'STANDARDIZATION/VAR/'+targetVar+'_testing.npy')
        else:
            if scene == 'historical':
                years = historical_years
                periodFilename = historicalPeriodFilename
            else:
                years = ssp_years
                periodFilename = sspPeriodFilename

            # Read dates (can be different for different calendars)
            ncVar = modNames[targetVar]
            modelName, modelRun = model.split('_')[0], model.split('_')[1]
            scene_dates = read.netCDF('../input_data/models/', ncVar + '_' + modelName + '_' + scene +'_'+ modelRun + '_'+periodFilename + '.nc',
                            ncVar)['times']
            idates = [i for i in range(len(scene_dates)) if scene_dates[i].year >= years[0] and scene_dates[i].year <= years[1]]
            scene_dates = list(np.array(scene_dates)[idates])
            var_scene = read.lres_data(targetVar, 'var', model=model, scene=scene)['data'][idates]

    # Declares variables for the other processes
    else:
        scene_dates = None
        i_4nn = None
        j_4nn = None
        w_4nn = None
        var_scene = None

    # Share data with all subprocesses
    if nproc>1:
        scene_dates = MPI.COMM_WORLD.bcast(scene_dates, root=0)
        i_4nn = MPI.COMM_WORLD.bcast(i_4nn, root=0)
        j_4nn = MPI.COMM_WORLD.bcast(j_4nn, root=0)
        w_4nn = MPI.COMM_WORLD.bcast(w_4nn, root=0)
        var_scene = MPI.COMM_WORLD.bcast(var_scene, root=0)
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step

    # Interpolate var_scene
    var_scene_interp = np.zeros((var_scene.shape[0], npoints_ichunk))
    for ipoint in range(npoints_ichunk):
        ipoint_global = points_chunk[ichunk][ipoint]
        if ipoint % 1000 == 0:
            print('interpolating', ipoint)
        var_scene_interp[:, ipoint] = grids.interpolate_predictors(var_scene,
                                  i_4nn[ipoint_global], j_4nn[ipoint_global], w_4nn[ipoint_global], interp_mode)[:, 0]
    del var_scene
    # np.save(pathTmp+'var_scene_interp', var_scene_interp)
    # var_scene_interp=np.load(pathTmp+'var_scene_interp.npy')


    # Load trained models of ichunk
    P00 = np.load(pathTmp + 'trained_P00_' + str(ichunk)+'.npy')
    P01 = np.load(pathTmp + 'trained_P01_' + str(ichunk)+'.npy')
    P10 = np.load(pathTmp + 'trained_P10_' + str(ichunk)+'.npy')
    P11 = np.load(pathTmp + 'trained_P11_' + str(ichunk)+'.npy')
    ECDF_pcp = np.load(pathTmp + 'trained_ECDF_pcp_' + str(ichunk)+'.npy')

    if nproc > 1:
        MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step
    if iproc == 0:
        shutil.rmtree(pathTmp)

    # Create empty array for results
    scene_ndates = len(scene_dates)
    est = np.zeros((scene_ndates, npoints_ichunk))

    # Define times
    times = scene_dates
    nDays = len(times)

    # Select points from chunk
    X = var_scene_interp
    del var_scene_interp

    # Goes through all points of the chunk
    for ipoint in range(npoints_ichunk):

        # Prints for monitoring
        if ipoint % 100 == 0:
            print('--------------------')
            print('ichunk:	', ichunk, '/', n_chunks)
            print('downscaling', targetVar, methodName, scene, model, round(100*ipoint/npoints_ichunk, 2), '%')

        # Select ipoint
        x = X[:, ipoint]
        p00 = P00[ipoint]
        p01 = P01[ipoint]
        p10 = P10[ipoint]
        p11 = P11[ipoint]
        ecdf_pcp =  ECDF_pcp[ipoint]

        # Get intervals
        iIntervals = np.zeros((nDays))
        for ith in range(nthresholds):
            th = thresholds[ith]
            # Identify days in the interval. If interval empty, idates will be taken from last not empty interval
            if ith == nthresholds-1:
                idates_tmp = np.where(x > th)[0]
            else:
                idates_tmp = np.where((x > th) * (x < thresholds[ith+1]))[0]
            if len(idates_tmp) > 0:
                idates = idates_tmp
            iIntervals[idates] = ith
        iIntervals = np.asarray(iIntervals, dtype='int')

        # Calculate intensity (all days will be wet days for now)
        p00_ipoint = p00[iIntervals]
        p01_ipoint = p01[iIntervals]
        p10_ipoint = p10[iIntervals]
        p11_ipoint = p11[iIntervals]
        est_ipoint = ecdf_pcp[iIntervals, np.random.randint(0, 101, nDays)]

        # Add noise to avoid repited values (if the noise addition leads to dry values, undo)
        noise = np.random.uniform(-.05, .05, est_ipoint.size)
        est_ipoint *= (1 + noise)
        undoNoise = np.where(est_ipoint < 1)
        est_ipoint[undoNoise] /= noise[undoNoise]

        # Calculate wet/dry days with a first order Markov chain and conditioned transitions
        rainy = np.zeros((nDays))
        rainy[0] = (x[0] >= 1)
        for iday in range(1, nDays):
            rainyPrevDay = rainy[iday-1]
            if rainyPrevDay == True:
                pRainy = p11_ipoint[iday]
            else:
                pRainy = p01_ipoint[iday]
            if pRainy >= np.random.uniform(0, 1):
                rainy[iday] = True
            else:
                rainy[iday] = False
        est_ipoint[rainy == False] = 0

        # Save to general array
        est[:, ipoint] = est_ipoint

    # Saves results to file
    np.save(pathOut + 'ichunk_' + str(ichunk) + '.npy', est)

########################################################################################################################
def downscale_chunk(targetVar, methodName, family, mode, fields, scene, model, iproc=0, nproc=1):
    """This function redirects to one or another WG method"""
    if methodName == 'WG-PDF':
        downscale_chunk_WG_PDF(targetVar, methodName, family, mode, fields, scene, model, iproc, nproc)
    elif methodName == 'WG-NMM':
        downscale_chunk_WG_NMM(targetVar, methodName, family, mode, fields, scene, model, iproc, nproc)

########################################################################################################################
def collect_chunks(targetVar, methodName, family, mode, fields, scene, model, n_chunks=1):
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
            periodFilename  = historicalPeriodFilename
            scene_dates = historical_dates
        else:
            periodFilename  = sspPeriodFilename
            scene_dates = ssp_dates
        # Read dates (can be different for different calendars)
        path = '../input_data/models/'
        ncVar = modNames[targetVar]
        modelName, modelRun = model.split('_')[0], model.split('_')[1]
        filename = ncVar + '_' + modelName + '_' + scene +'_'+ modelRun + '_'+periodFilename  + '.nc'
        model_dates = np.ndarray.tolist(read.netCDF(path, filename, ncVar)['times'])
        model_dates = [x for x in model_dates if x.year >= scene_dates[0].year and x.year <= scene_dates[-1].year]

    # Create empty array and accumulate results
    est = np.zeros((len(model_dates), 0))
    for ichunk in range(n_chunks):
        path = '../tmp/ESTIMATED_'+ '_'.join((targetVar, methodName, scene, model)) + '/'
        filename = path + '/ichunk_' + str(ichunk) + '.npy'
        est = np.append(est, np.load(filename), axis=1)
    shutil.rmtree(path)

    if targetVar == 'pr':
        est[est < 0.01] = 0

    # Save to file
    if experiment == 'EVALUATION':
        pathOut = '../results/'+experiment+'/'+targetVar.upper()+'/'+methodName+'/daily_data/'
    elif experiment == 'PSEUDOREALITY':
        aux = np.zeros((len(scene_dates), hres_npoints[targetVar]))
        aux[:] = np.nan
        idates = [i for i in range(len(scene_dates)) if scene_dates[i] in model_dates]
        aux[idates] = est
        est = aux
        del aux
        pathOut = '../results/'+experiment+'/'+ GCM_longName + '_' + RCM + '/'+targetVar.upper()+'/'+methodName+'/daily_data/'
    else:
        aux = np.zeros((len(scene_dates), hres_npoints[targetVar]))
        aux[:] = np.nan
        idates = [i for i in range(len(scene_dates)) if scene_dates[i] in model_dates]
        aux[idates] = est
        est = aux
        del aux
        pathOut = '../results/'+experiment+'/'+targetVar.upper()+'/'+methodName+'/daily_data/'

    # Save results
    hres_lats = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/hres_lats.npy')
    hres_lons = np.load(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp_mode+'/hres_lons.npy')

    # Set units
    units = predictands_units[targetVar]
    if units == None:
        units = ''

    if split_mode[:4] == 'fold':
        sufix = '_' + split_mode
    else:
        sufix = ''

    # Special values are set to nan
    warnings.filterwarnings("ignore", message="invalid value encountered in less")
    est[np.abs(est-predictands_codification[targetVar]['special_value']) < 0.01] = np.nan
    print('-------------------------------------------------------------------------')
    print('results contain', 100*int(np.where(np.isnan(est))[0].size/est.size), '% of nans')
    print('-------------------------------------------------------------------------')

    # Force to theoretical range
    minAllowed, maxAllowed = predictands_range[targetVar]['min'], predictands_range[targetVar]['max']
    if  minAllowed != None:
        est[est < minAllowed] == minAllowed
    if  maxAllowed != None:
        est[est > maxAllowed] == maxAllowed

    # Save data to netCDF file
    write.netCDF(pathOut, model+'_'+scene+sufix+'.nc', targetVar, est, units, hres_lats, hres_lons, scene_dates, regular_grid=False)
    # print(est[0, :10], est.shape)

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
