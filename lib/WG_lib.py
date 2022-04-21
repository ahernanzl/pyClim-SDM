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
import gui_lib
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
def train_chunk_WG_PDF(var, methodName, family, mode, fields, iproc=0, nproc=1):
    '''
    Calibrates regression for all points, divided in chunks if run at HPC, for different distribution parameters,
    aggregating in three moths blocks: TOT, MEAN, VAR, P1, P00, P01, P10, P11.
    '''

    # Define pathOut
    pathOut = '../tmp/TRAINING_'  + var + '_' + methodName + '/'

    aggMonths = aggregation_pcp_WG_PDF

    # Define times
    times = training_dates
    years = set([x.year for x in times])
    nYears = len(years)
    n_blocks = 12*nYears

    # create chunks
    n_chunks = nproc
    len_chunk = int(math.ceil(float(hres_npoints[var[0]]) / n_chunks))
    points_chunk = []
    for ichunk in range(n_chunks):
        points_chunk.append(list(range(hres_npoints[var[0]]))[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    ichunk = iproc
    npoints_ichunk = len(points_chunk[ichunk])

    # Declares variables for father process, who creates pathOut
    if iproc == 0:
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)
        i_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_dict[mode]+'/i_4nn.npy')
        j_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_dict[mode]+'/j_4nn.npy')
        w_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_dict[mode]+'/w_4nn.npy')
        obs = read.hres_data(var, period='training')['data']
        var_calib = np.load(pathAux+'STANDARDIZATION/VAR/'+var+'_training.npy')
        var_calib_interp = np.zeros((var_calib.shape[0], hres_npoints[var[0]]))
        for ipoint in range(hres_npoints[var[0]]):
            if ipoint % 1000 == 0:
                print('interpolating', ipoint)
            var_calib_interp[:, ipoint] = grids.interpolate_predictors(var_calib,
                                      i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_dict[mode])[:, 0]
        del var_calib

        # Save X, Y (chunks)
        for i in range(nproc):
            np.save(pathOut + 'var_calib_' + str(i), var_calib_interp[:, points_chunk[i]])
            np.save(pathOut + 'obs_' + str(i), obs[:, points_chunk[i]])
        del obs, var_calib_interp

    # Waits for all subprocesses to complete last step
    if nproc > 1:
        MPI.COMM_WORLD.Barrier()

    # Read X, Y (chunks)
    X = np.load(pathOut + 'var_calib_' + str(iproc) + '.npy')
    Y = np.load(pathOut + 'obs_' + str(iproc) + '.npy')

    # Create empty arrays for results
    PARAM1_X = np.zeros((n_blocks, npoints_ichunk))
    PARAM2_X = np.zeros((n_blocks, npoints_ichunk))
    PARAM1_Y = np.zeros((n_blocks, npoints_ichunk))
    PARAM2_Y = np.zeros((n_blocks, npoints_ichunk))

    # For each month it takes a 3-month aggregation (with the previous and following months)
    i_block = 0
    for year in years:
        for month in range(1, 13):
            print('i_block', i_block, '/', n_blocks)

            if (var[0] == 't') or (aggMonths == 1):
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
            Y_month = Y[idates]

            if var[0] == 't':
                # mean and std
                PARAM1_X[i_block] = np.nanmean(X_month, axis=0)
                PARAM1_Y[i_block] = np.nanmean(Y_month, axis=0)
                PARAM2_X[i_block] = np.nanstd(X_month, axis=0)
                PARAM2_Y[i_block] = np.nanstd(Y_month, axis=0)
            else:
                # R01 and mean
                PARAM1_X[i_block] = np.nanmean(X_month>=1, axis=0)
                PARAM1_Y[i_block] = np.nanmean(Y_month>=1, axis=0)
                PARAM2_X[i_block] = np.nanmean(X_month, axis=0)
                PARAM2_Y[i_block] = np.nanmean(Y_month, axis=0)

            i_block += 1

    del X, Y

    # Add ichunk to pathOut
    pathOut += 'ichunk_' + str(iproc) + '/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Create empty arrays for regressors
    PARAM1_reg = npoints_ichunk * [None]
    PARAM2_reg = npoints_ichunk * [None]

    # Transponse
    PARAM1_X = PARAM1_X.T
    PARAM2_X = PARAM2_X.T
    PARAM1_Y = PARAM1_Y.T
    PARAM2_Y = PARAM2_Y.T

    # Train regressors
    for ipoint in range(npoints_ichunk):
        if ipoint % 100 == 0:
            print('training', ipoint, '/', npoints_ichunk)

        PARAM1_reg[ipoint] = LinearRegression().fit(PARAM1_X[ipoint].reshape(-1, 1), PARAM1_Y[ipoint].reshape(-1, 1))
        PARAM2_reg[ipoint] = LinearRegression().fit(PARAM2_X[ipoint].reshape(-1, 1), PARAM2_Y[ipoint].reshape(-1, 1))

    # Save chunks
    outfile = open(pathOut + 'PARAM1_reg', 'wb')
    pickle.dump(PARAM1_reg, outfile)
    outfile.close()
    outfile = open(pathOut + 'PARAM2_reg', 'wb')
    pickle.dump(PARAM2_reg, outfile)
    outfile.close()


########################################################################################################################
def collect_chunks_WG_PDF(var, methodName, family, n_chunks=1):
    """
    This function collects the results of downscale_chunk() and saves them into a final single file.
    """

    print('--------------------------------------')
    print('collect chunks', n_chunks)

    # Define paths
    pathIn = '../tmp/TRAINING_'  + var + '_' + methodName + '/'
    pathOut = pathAux + 'TRAINED_'+methodName+'/' + var.upper() + '/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Create empty regressors to be filled
    PARAM1_reg = np.zeros((0, ))
    PARAM2_reg = np.zeros((0, ))

    # Read trained models chunks and accumulate them
    for ichunk in range(n_chunks):
        infile = open(pathIn+'ichunk_'+str(ichunk)+'/PARAM1_reg', 'rb')
        PARAM1_reg_ichunk = pickle.load(infile)
        infile.close()
        PARAM1_reg = np.append(PARAM1_reg, PARAM1_reg_ichunk, axis=0)

        infile = open(pathIn+'ichunk_'+str(ichunk)+'/PARAM2_reg', 'rb')
        PARAM2_reg_ichunk = pickle.load(infile)
        infile.close()
        PARAM2_reg = np.append(PARAM2_reg, PARAM2_reg_ichunk, axis=0)
    shutil.rmtree(pathIn)

    # Save to file
    outfile = open(pathOut + methodName + '_PARAM1_reg', 'wb')
    pickle.dump(PARAM1_reg, outfile)
    outfile.close()
    outfile = open(pathOut + methodName + '_PARAM2_reg', 'wb')
    pickle.dump(PARAM2_reg, outfile)
    outfile.close()


########################################################################################################################
def train_chunk_WG_NMM(var, methodName, family, mode, fields, iproc=0, nproc=1):
    '''
    Calculates, for different intervals of pcp given by the reanalysis, the transition probabilities and ECDF.
    '''

    # Define pathOut
    pathOut = '../tmp/TRAINING_'  + var + '_' + methodName + '/'

    thresholds = thresholds_WG_NMM
    nthresholds = len(thresholds)

    # Define times
    times = training_dates
    years = set([x.year for x in times])
    nYears = len(years)

    # create chunks
    n_chunks = nproc
    len_chunk = int(math.ceil(float(hres_npoints[var[0]]) / n_chunks))
    points_chunk = []
    for ichunk in range(n_chunks):
        points_chunk.append(list(range(hres_npoints[var[0]]))[ichunk * len_chunk:(ichunk + 1) * len_chunk])
    ichunk = iproc
    npoints_ichunk = len(points_chunk[ichunk])

    # Declares variables for father process, who creates pathOut
    if iproc == 0:
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)
        i_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_dict[mode]+'/i_4nn.npy')
        j_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_dict[mode]+'/j_4nn.npy')
        w_4nn = np.load(pathAux+'ASSOCIATION/'+var[0].upper()+'_'+interp_dict[mode]+'/w_4nn.npy')
        obs = read.hres_data(var, period='training')['data']
        var_calib = np.load(pathAux+'STANDARDIZATION/VAR/'+var+'_training.npy')
        var_calib_interp = np.zeros((var_calib.shape[0], hres_npoints[var[0]]))
        for ipoint in range(hres_npoints[var[0]]):
            if ipoint % 1000 == 0:
                print('interpolating', ipoint)
            var_calib_interp[:, ipoint] = grids.interpolate_predictors(var_calib,
                                      i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_dict[mode])[:, 0]
        del var_calib

        # Save X, Y (chunks)
        for i in range(nproc):
            np.save(pathOut + 'var_calib_' + str(i), var_calib_interp[:, points_chunk[i]])
            np.save(pathOut + 'obs_' + str(i), obs[:, points_chunk[i]])
        del obs, var_calib_interp

    # Waits for all subprocesses to complete last step
    if nproc > 1:
        MPI.COMM_WORLD.Barrier()

    # Read X, Y (chunks)
    X = np.load(pathOut + 'var_calib_' + str(iproc) + '.npy')
    Y = np.load(pathOut + 'obs_' + str(iproc) + '.npy')
    print(iproc, X.shape, Y.shape, len(times))

    # Add ichunk to pathOut
    pathOut += 'ichunk_' + str(iproc) + '/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Create empty arrays for probabilities of transition and ECDFs with 101 intervals
    P00 = np.zeros((npoints_ichunk, nthresholds))
    P01 = np.zeros((npoints_ichunk, nthresholds))
    P10 = np.zeros((npoints_ichunk, nthresholds))
    P11 = np.zeros((npoints_ichunk, nthresholds))
    ECDF_pcp = np.zeros((npoints_ichunk, nthresholds, 101))

    # Train regressors
    for ipoint in range(npoints_ichunk):
        if ipoint % 100 == 0:
            print('training', ipoint, '/', npoints_ichunk)

        # Select data for ipoint
        x, y = X[:, ipoint], Y[:, ipoint]

        # Calculate transitions 00 01 10 11. First day is removed, as it has no previous day.
        y_wet = (y >= 1)
        trans = y_wet[1:] + y_wet[:-1]/2
        x, y = x[1:], y[1:]

        # Go through all low resolution pcp intervals
        for ith in range(nthresholds):

            th = thresholds[ith]

            # Identify days in the interval. If interval empty, idates will be taken from last not empty interval
            if ith == nthresholds-1:
                idates_tmp = np.where(x > th)[0]
            else:
                idates_tmp = np.where((x > th) * (x < thresholds[ith+1]))[0]
            if len(idates_tmp) > 0:
                idates = idates_tmp

            # Select data
            x_idates, y_idates, trans_idates = x[idates], y[idates], trans[idates]

            # Calculate transition probabilities
            n00 = np.where(trans_idates == 0)[0].size
            n01 = np.where(trans_idates == 1)[0].size
            n10 = np.where(trans_idates == .5)[0].size
            n11 = np.where(trans_idates == 1.5)[0].size
            if (n00 + n01) != 0:
                p00 = round(n00 / (n00 + n01), 2)
                p01 = round(n01 / (n00 + n01), 2)
            else:
                p00 = 0
                p01 = 1
            if (n10 + n11) != 0:
                p10 = round(n10 / (n10 + n11), 2)
                p11 = round(n11 / (n10 + n11), 2)
            else:
                p10 = 1
                p11 = 0

            # Save to general arrays
            P00[ipoint, ith] = p00
            P01[ipoint, ith] = p01
            P10[ipoint, ith] = p10
            P11[ipoint, ith] = p11

            # For intensity, select wet days and saves ECDF
            y_wet = y_idates[y_idates >= 1]
            ECDF_pcp[ipoint, ith] = np.percentile(y_wet, np.arange(101))

    # Save to files
    np.save(pathOut + 'P00', P00)
    np.save(pathOut + 'P01', P01)
    np.save(pathOut + 'P10', P10)
    np.save(pathOut + 'P11', P11)
    np.save(pathOut + 'ECDF_pcp', ECDF_pcp)

    # P00 = np.load(pathOut + 'P00.npy')
    # P01 = np.load(pathOut + 'P01.npy')
    # P10 = np.load(pathOut + 'P10.npy')
    # P11 = np.load(pathOut + 'P11.npy')
    # ECDF_pcp = np.load(pathOut + 'ECDF_pcp.npy')


########################################################################################################################
def collect_chunks_WG_NMM(var, methodName, family, n_chunks=1):
    """
    This function collects the results of downscale_chunk() and saves them into a final single file.
    """

    print('--------------------------------------')
    print('collect chunks', n_chunks)

    # Define paths
    pathIn = '../tmp/TRAINING_'  + var + '_' + methodName + '/'
    pathOut = pathAux + 'TRAINED_'+methodName+'/' + var.upper() + '/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)


    thresholds = thresholds_WG_NMM
    nthresholds = len(thresholds)

    # Create empty arrays for probabilities of transition and ECDFs with 101 intervals
    P00 = np.zeros((hres_npoints[var[0]], nthresholds))
    P01 = np.zeros((hres_npoints[var[0]], nthresholds))
    P10 = np.zeros((hres_npoints[var[0]], nthresholds))
    P11 = np.zeros((hres_npoints[var[0]], nthresholds))
    ECDF_pcp = np.zeros((hres_npoints[var[0]], nthresholds, 101))

    # Read trained models chunks and accumulate them
    ipoint = 0
    for ichunk in range(n_chunks):
        p00 = np.load(pathIn + 'ichunk_'+str(ichunk) + '/P00.npy')
        p01 = np.load(pathIn + 'ichunk_'+str(ichunk) + '/P01.npy')
        p10 = np.load(pathIn + 'ichunk_'+str(ichunk) + '/P10.npy')
        p11 = np.load(pathIn + 'ichunk_'+str(ichunk) + '/P11.npy')
        ecdf_pcp = np.load(pathIn + 'ichunk_'+str(ichunk) + '/ECDF_pcp.npy')
        npoints_chunk = p00.shape[0]
        P00[ipoint:ipoint+npoints_chunk] = p00
        P01[ipoint:ipoint+npoints_chunk] = p01
        P10[ipoint:ipoint+npoints_chunk] = p10
        P11[ipoint:ipoint+npoints_chunk] = p11
        ECDF_pcp[ipoint:ipoint+npoints_chunk] = ecdf_pcp
        ipoint += npoints_chunk

    shutil.rmtree(pathIn)

    # Save to file
    np.save(pathOut + 'P00', P00)
    np.save(pathOut + 'P01', P01)
    np.save(pathOut + 'P10', P10)
    np.save(pathOut + 'P11', P11)
    np.save(pathOut + 'ECDF_pcp', ECDF_pcp)



########################################################################################################################
def train_chunk(var, methodName, family, mode, fields, iproc=0, nproc=1):
    """This function redirects to one or another WG method"""
    if methodName == 'WG-PDF':
        train_chunk_WG_PDF(var, methodName, family, mode, fields, iproc, nproc)
    elif methodName == 'WG-NMM':
        train_chunk_WG_NMM(var, methodName, family, mode, fields, iproc, nproc)

########################################################################################################################
def collect_chunks(var, methodName, family, n_chunks=1):
    """This function redirects to one or another WG method"""
    if methodName == 'WG-PDF':
        collect_chunks_WG_PDF(var, methodName, family, n_chunks)
    elif methodName == 'WG-NMM':
        collect_chunks_WG_NMM(var, methodName, family, n_chunks)



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

    train_chunk(var, methodName, family, mode, fields, iproc, nproc)
    MPI.COMM_WORLD.Barrier()            # Waits for all subprocesses to complete last step
    if iproc==0:
        collect_chunks(var, methodName, family, nproc)