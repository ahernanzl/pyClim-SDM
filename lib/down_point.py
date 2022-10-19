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
def ANA_pr(syn_dist, weather_type_id, iana, pred_scene, var_scene, pred_calib, var_calib, obs, corr,
             i_4nn, j_4nn, w_4nn, methodName, special_value):
    """

    :param syn_dist: (n_analogs_preselection,)
    :param weather_type_id:
    :param iana: (n_analogs_preselection,)
    :param pred_scene: (1, npreds, nlats, nlons)
    :param pred_calib: (ndays, npreds, nlats, nlons)
    :param obs: (ndays,)
    :param corr: (k_clusters, npreds)
    :param estimation_method:
    :return: est
    """

    mode = 'PP'

    # Define analogy_mode and estimation_mode
    analogy_mode = methodName.split('-')[1]
    estimation_mode = methodName.split('-')[2]

    # Get local distances
    if analogy_mode in ('SYN', 'PCP'):
        loc_dist = syn_dist
        nSigPreds = 0
    elif analogy_mode == 'LOC':
        sigPreds = np.where(corr[weather_type_id] == True)[0]
        nSigPreds = sigPreds.size
        # Zero significant predictors, does not consider local distances
        if nSigPreds == 0:
            loc_dist = syn_dist
        else:
            # Missing values at model predictors, does not consider local distances
            if len(np.where(np.isnan(pred_scene))[0]) != 0:
                loc_dist = syn_dist
            # Consider local distances
            else:
                pred_scene = grids.interpolate_predictors(pred_scene,  i_4nn, j_4nn, w_4nn, interp_mode)
                pred_calib = grids.interpolate_predictors(pred_calib[iana], i_4nn, j_4nn, w_4nn, interp_mode)
                loc_dist = ANA_lib.get_local_distances(pred_calib, pred_scene, sigPreds)
    elif analogy_mode == 'HYB':
        loc_dist = syn_dist
        nSigPreds = 0
        var_scene = grids.interpolate_predictors(var_scene[np.newaxis, :, :, :],  i_4nn, j_4nn, w_4nn, interp_mode)[:, 0]
        var_calib = grids.interpolate_predictors(var_calib[iana], i_4nn, j_4nn, w_4nn, interp_mode)[:, 0]

    # Remove days with no predictand, unless there is no day with predictand
    valid = np.where(obs < special_value)[0]
    if ((valid.size != obs.size) and (valid.size != 0)):
        i = [i for i in range(iana.size) if iana[i] in valid]
        syn_dist, loc_dist, iana = syn_dist[i], loc_dist[i], iana[i]

    # Synop + local combined distance is used only if there are sigPreds
    if nSigPreds != 0:
        dist = (syn_dist + loc_dist) / 2
    else:
        dist = syn_dist

    # Selects the most similar days
    aux_index = np.argsort(dist)[:kNN]
    index_final = iana[aux_index]

    # Gets distances and precipitation of sorted analogs
    dist = dist[aux_index]

    # Select analogs
    obs_analogs = obs[index_final]

    if analogy_mode == 'HYB':
        var_analogs = var_calib[aux_index]
        dist_pcp = np.abs(var_analogs - var_scene)
        aux_index = np.argsort(dist_pcp)
        est = obs_analogs[aux_index][0]

    else:
        # Gets weight based on distances
        dist[dist==0] = 0.000000001
        W = np.ones(dist.shape) / dist

        # Estimates precipitation with one decimal
        if estimation_mode == '1NN':
            est = obs_analogs[0]
        elif estimation_mode == 'kNN':
            est = np.average(obs_analogs, weights=W)
        elif estimation_mode == 'rand':
            est = np.random.choice(obs_analogs, p=W/W.sum())

    return est


########################################################################################################################
def ANA_others(targetVar, iana, pred_scene, var_scene, pred_calib, var_calib, obs, coef, intercept, dist_centroid,
           i_4nn, j_4nn, w_4nn, methodName, th_metric='median'):

    """
    This function downscale a particular point.
    Return: array of estimated temperature
    """

    mode = 'PP'

    # Interpolate and reshapes for regression
    X = grids.interpolate_predictors(pred_scene,  i_4nn, j_4nn, w_4nn, interp_mode)

    # If there are missing predictors, or regression is not precalibrated by clusters, or distance of problem day to
    # centroid too large, or intercept is nan because there where not enough valid data to pre-calibrate the regression,
    # set train_regressor to True and interpolate X_train
    train_regressor = False
    missing_preds = np.where(np.isnan(X))[1]

    # Get dist_th
    if th_metric == 'median':
        dist_th = np.median(np.load(pathAux + 'WEATHER_TYPES/dist.npy'))
    elif th_metric == 'max':
        dist_th = np.max(np.load(pathAux + 'WEATHER_TYPES/dist.npy'))
    elif th_metric == 'p90':
        dist_th = np.percentile(np.load(pathAux + 'WEATHER_TYPES/dist.npy'), 90)

    if (len(missing_preds) != 0) or (methodName!='MLR-WT') \
            or (dist_centroid>dist_th) or (np.isnan(intercept) == True):
        train_regressor=True

    if train_regressor==True:
        X_train = grids.interpolate_predictors(pred_calib[iana],  i_4nn, j_4nn, w_4nn, interp_mode)
        Y_train = obs[iana]

    # Remove missing predictors
    if len(missing_preds) != 0:
        valid_preds = [x for x in range(X.shape[1]) if x not in missing_preds]
        X_train = X_train[:,valid_preds]
        X = X[:,valid_preds]

    # Checks for missing predictands and remove them from training datasets
    special_value = int(100 * predictands_codification[targetVar]['special_value'])

    if train_regressor == True:
        valid = np.where(Y_train < special_value)[0]
        # If not enough data for calibration
        if valid.size < 30:
            est = special_value
        else:
            # Remove missing predictands
            if valid.size != Y_train.size:
                X_train = X_train[valid]
                Y_train = Y_train[valid]

            # Train regression
            reg = RidgeCV()
            reg.fit(X_train, Y_train)
            est = reg.predict(X)[0]
    else:
        # Regression by clusters precalibrated, apply coefficients
        est = np.sum(X*coef)+intercept[0]

    return est


########################################################################################################################
def TF_others(X, reg_ipoint):

    """
    This function downscale a particular point.
    Return: array of estimated temperature
    """

    try:
        Y = reg_ipoint.predict(X, verbose=0)
    except:
        Y = reg_ipoint.predict(X)

    if Y.ndim > 1:
        Y = Y[:, 0]

    return Y

########################################################################################################################
def TF_pr(methodName, X, clf_ipoint, reg_ipoint):
    """
    This function downscale a particular point.
    Return: array of estimated precipitation
    """

    if 'sklearn' in str(type(clf_ipoint)):
        if classifier_mode == 'probabilistic':
            # A point is rainy if its probability of pcp is greater or equal than a random number between 0 and 1.
            odds_rainy = clf_ipoint.predict_proba(X)
            israiny = (odds_rainy[:, 1] >= np.random.uniform(size=(odds_rainy[:, 1].shape)))
        elif classifier_mode == 'deterministic':
            # A point is rainy if so it is classified
            israiny = clf_ipoint.predict(X)
    else:
        try:
            odds_rainy = clf_ipoint.predict(X, verbose=0)
        except:
            odds_rainy = clf_ipoint.predict(X)
        if classifier_mode == 'probabilistic':
            # A point is rainy if its probability of pcp is greater or equal than a random number between 0 and 1.
            israiny = (odds_rainy[:, 0] >= np.random.uniform(size=(odds_rainy[:, 0].shape)))
        elif classifier_mode == 'deterministic':
            # A point is rainy if its probability of pcp is greater or equal than 0.5
            israiny = (odds_rainy[:, 0] >= .5)

    # Predicts estimated target
    try:
        Y = reg_ipoint.predict(X, verbose=0)
    except:
        Y = reg_ipoint.predict(X)

    if Y.ndim > 1:
        Y = Y[:, 0]

    # Transforms target if exponential regression
    if methodName == 'GLM-EXP':
        Y = np.exp(Y, dtype='float128')
    elif methodName == 'GLM-CUB':
        Y = Y**3

    # # Set to 0.99 mm points classified as not rainy, set to 1 mm points classified as rainy where the regression
    # # predicts no rain, and set to zero negative values
    # Y[(Y >= 100*wetDry_th)*(israiny == False)] = 99*wetDry_th
    # Y[(Y < 100*wetDry_th)*(israiny == True)] = 100*wetDry_th
    # Y[Y < 0] = 0

    # Set to zero points classified as not rainy
    Y[(Y >= 100*wetDry_th)*(israiny == False)] = 0

    # Set to zero negative values
    Y[Y < 0] = 0

    return Y
