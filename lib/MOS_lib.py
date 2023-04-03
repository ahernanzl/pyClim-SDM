import sys

import numpy as np

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
def treat_frecuencies(obs, hist, sce, th=0.05, jitter=0.01):
    """
    This function is a preprocess for QDM, in order to maintain the expected wet/dry frecuency and, at the same time,
    to avoid artifacted delta changes when bias correcting Transfer Function methods for precipitation, where the
    simulated distribution is discontinuous (it has zeros by the classifier and wet days far from zero by the regressor,
    i.e. there is a marked discontinuity in the PDF).
    The strategy is:
    1 - To avoid artifacted high delta by setting the problematic interval of wet days at scene to dry
    2 - To preserve the expected frequency by forzing wet days in obs
    """
    obs_dryFreq = np.sum(obs<th)/obs.size
    hist_dryFreq = np.sum(hist<th)/hist.size
    sce_dryFreq = np.sum(sce<th)/sce.size

    # If too many wet days (which QDM does not automatically correct)
    if hist_dryFreq > sce_dryFreq:

        # Setting the problematic interval of wet days at scene to dry
        p1, p2 = 100*sce_dryFreq, 100*hist_dryFreq
        iTodry = np.where((sce>np.percentile(sce, p1)) * (sce<np.percentile(sce, p2)))[0]
        sce[iTodry] = 0

        # Forze needed observed zeros to dry
        expected_dryFreq = obs_dryFreq + sce_dryFreq - hist_dryFreq
        p1, p2 = 100*expected_dryFreq, 100*obs_dryFreq
        iToWet = np.where((obs>np.percentile(obs, p1)) * (obs<np.percentile(obs, p2)))[0]
        obs[iToWet] = th + np.random.uniform(low=0, high=jitter, size=iToWet.shape)

    return obs, hist, sce


########################################################################################################################
def quantile_mapping(obs, hist, sce, targetVar):
    """
   Quantile Mapping. Basic version of empirical QM (Themeßl et al., 2011). Additive correction both all target variables

    Args:
    * obs (nDaysObs, nPoints): the observational data
    * hist (nDaysHist, nPoints): the model data at the reference period
    * sce (nDaysSce, nPoints): the scenario data that shall be corrected

    Themeßl, M.J., Gobiet, A. and Leuprecht, A. (2011), Empirical-statistical downscaling and error correction of daily
    precipitation from regional climate models. Int. J. Climatol., 31: 1530-1544. https://doi.org/10.1002/joc.2168
    """

    # Define parameters and variables
    nPoints, nDays_ref, nDays_sce = obs.shape[1], obs.shape[0], sce.shape[1]
    sce_corrected = 1*sce

    # Go through all points
    for ipoint in range(nPoints):
        if ipoint % 100 == 0:
            print(ipoint)

        # Select data from one point
        obs_data = obs.T[ipoint]
        hist_data = hist.T[ipoint]
        sce_data = sce.T[ipoint]

        # Remove missing data from obs and hist
        obs_data = obs_data[abs(obs_data - predictands_codification[targetVar]['special_value']) > 0.01]
        hist_data = hist_data[np.isnan(hist_data) == False]

        if hist_data.size == 0:
            sce_corrected.T[ipoint] = np.nan
        else:
            # Select valid data from sce
            ivalid = np.where(np.isnan(sce_data) == False)
            sce_data = sce_data[ivalid]

            # Calculate correction
            hist_ecdf = ECDF(hist_data)
            p = hist_ecdf(sce_data) * 100
            corr = np.percentile(obs_data, p) - np.percentile(hist_data, p)

            # Add correction
            sce_corrected.T[ipoint][ivalid] = sce_data + corr

    return sce_corrected




########################################################################################################################
def detrended_quantile_mapping(obs, hist, sce, targetVar, th=0.05):
    """
    Detrendend Quantile Mapping: remove trend and mean, and then apply empirical quantile mapping (Cannon et al., 2015).
    Additive or multiplicative correction for each targetVar, configurable at advanced_settings.py

    Args:
    * obs (nDaysObs, nPoints): the observational data
    * hist (nDaysHist, nPoints): the model data at the reference period
    * sce (nDaysSce, nPoints): the scenario data that shall be corrected

    Adapted from https://github.com/pacificclimate/ClimDown
    For temperature the implementation differs from climDown. Here series are detrended, not only substracted of their
    mean. This way the problem of extrapolation is avoided.

    Cannon, A.J., S.R. Sobie, and T.Q. Murdock (2015) Bias Correction of GCM Precipitation by Quantile Mapping: How Well
    Do Methods Preserve Changes in Quantiles and Extremes?. J. Climate, 28, 6938–6959,
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """

    # Define parameters and variables
    nPoints, nDays_ref, nDays_sce = obs.shape[1], obs.shape[0], sce.shape[1]
    sce_corrected = 1*sce

    # Go through all points
    for ipoint in range(nPoints):
        if ipoint % 100 == 0:
            print(ipoint)

        # Select data from one point
        obs_data = obs.T[ipoint]
        hist_data = hist.T[ipoint]
        sce_data = sce.T[ipoint]

        # Remove missing data from obs and hist
        obs_data = obs_data[abs(obs_data - predictands_codification[targetVar]['special_value']) > 0.01]
        hist_data = hist_data[np.isnan(hist_data) == False]

        if hist_data.size == 0:
            sce_corrected.T[ipoint] = np.nan
        else:
            # Select valid data from sce
            ivalid = np.where(np.isnan(sce_data) == False)
            sce_data = sce_data[ivalid]

            # For multiplicative correction
            if bc_mode_dict[targetVar] == 'rel':

                # Treat zeros
                obs_data[obs_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(obs_data < th)[0].shape))
                hist_data[hist_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(hist_data < th)[0].shape))
                sce_data[sce_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(sce_data < th)[0].shape))

                # Remove change in the mean value
                sce_mean = np.mean(sce_data)
                sce_data *= np.mean(hist_data) / np.mean(sce_data)

                # Calculate correction
                hist_ecdf = ECDF(hist_data)
                p = hist_ecdf(sce_data) * 100
                corr = np.percentile(obs_data, p) / np.percentile(hist_data, p)

                # Add correction and change in the mean value
                sce_corrected.T[ipoint][ivalid] = sce_data * corr * sce_mean / np.mean(hist_data)
                sce_corrected.T[ipoint][ivalid][sce_corrected.T[ipoint][ivalid] < th] = 0

            # For additive correction
            else:

                # Remove mean and detrend
                obs_detrended = detrend(obs_data)
                hist_detrended = detrend(hist_data)
                sce_detrended = detrend(sce_data)

                # Calculate correction using data without mean and trend
                hist_ecdf = ECDF(hist_detrended)
                p = hist_ecdf(sce_detrended) * 100
                corr = np.percentile(obs_detrended, p) - np.percentile(hist_detrended, p)

                # Add correction and mean values removed while detrending
                sce_corrected.T[ipoint][ivalid] = sce_data + corr + np.mean(obs_data) - np.mean(hist_data)

    return sce_corrected


########################################################################################################################
def quantile_delta_mapping(obs, hist, sce, targetVar, default_th=0.05,
                maxExtrapolationFactor=1):
    """
    Quantile Delta Mapping: apply delta change correction to all quantiles (Cannon et al., 2015).
    Additive or multiplicative correction for each targetVar, configurable at advanced_settings.py

    Args:
    * obs (nDaysObs, nPoints): the observational data
    * hist (nDaysHist, nPoints): the model data at the reference period
    * sce (nDaysSce, nPoints): the scenario data that shall be corrected
    * default_th: threshold for zero values by default. Inside the function specific thresholds for each target variable
                are defined.
    * maxExtrapolationFactor: for multiplicative (rel) correction, extreme values can produce artifacts. They are limited
        to maxExtrapolationFactor times the maximum observed value

    Adapted from https://github.com/pacificclimate/ClimDown

    Cannon, A.J., S.R. Sobie, and T.Q. Murdock (2015) Bias Correction of GCM Precipitation by Quantile Mapping: How Well
    Do Methods Preserve Changes in Quantiles and Extremes?. J. Climate, 28, 6938–6959,
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """

    # Define specific thresholds for 'zero value' for each targetVar
    th_dict = {'tas': None, 'tasmax': None, 'tasmin': None, 'pr': .2, 'uas': None, 'vas': None, 'sfcWind': None,
        'hurs': None, 'huss': .00001, 'clt': None, 'rsds': None, 'rlds': None, 'psl': None, 'ps': None,
        'evspsbl': None, 'evspsblpot': None, 'mrro': None, 'mrso': None,}
    th = th_dict[targetVar]
    if th == None:
        th = default_th

    # Define jitter as maximum machine precision
    eps_obs = np.finfo(type(obs[0][0])).eps
    eps_hist = np.finfo(type(hist[0][0])).eps
    eps_sce = np.finfo(type(sce[0][0])).eps
    jitter = max(eps_obs, eps_hist, eps_sce)

    # Define parameters and variables
    nPoints, nDays_ref, nDays_sce = obs.shape[1], obs.shape[0], sce.shape[1]
    sce_corrected = 1*sce

    # Go through all points
    for ipoint in range(nPoints):
        if ipoint % 100 == 0:
            print(ipoint)

        # Select data from one point
        obs_data = obs.T[ipoint]
        hist_data = hist.T[ipoint]
        sce_data = sce.T[ipoint]
        #
        # obs_auxPlot = 1* obs_data
        # hist_auxPlot = 1* hist_data
        # sce_auxPlot = 1*sce_data

        # Remove missing data from obs and hist
        obs_data = obs_data[abs(obs_data - predictands_codification[targetVar]['special_value']) > 0.01]
        hist_data = hist_data[np.isnan(hist_data) == False]

        if hist_data.size == 0:
            sce_corrected.T[ipoint] = np.nan
        else:

            # Select valid data from sce
            ivalid = np.where(np.isnan(sce_data) == False)
            sce_data = sce_data[ivalid]

            # For multiplicative correction
            if bc_mode_dict[targetVar] == 'rel':
                # Multiply hist, sce by a factor so artifacted high deltas for TF methods are avoided
                aux = 1*hist_data; aux[aux<=th] = np.nan; m1 = np.nanmin(aux)
                aux = 1*sce_data; aux[aux<=th] = np.nan; m2 = np.nanmin(aux)
                # print(m1, m2)
                factor = min(m1, m2)/th
                hist_data /= factor
                sce_data /= factor
                # Add noise to zeros
                obs_data[obs_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(obs_data < th)[0].shape))
                hist_data[hist_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(hist_data < th)[0].shape))
                sce_data[sce_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(sce_data < th)[0].shape))
                # # Add a small amount of noise to accomodate ties due to limited precision
                # obs_data += np.random.uniform(low=-jitter, high=jitter, size=obs_data.shape)
                # hist_data += np.random.uniform(low=-jitter, high=jitter, size=hist_data.shape)
                # sce_data += np.random.uniform(low=-jitter, high=jitter, size=sce_data.shape)
            # Calculate percentiles
            sce_ecdf = ECDF(sce_data)
            p = sce_ecdf(sce_data) * 100

            # Calculate and apply delta correction
            # For multiplicative correction
            if bc_mode_dict[targetVar] == 'rel':
                delta = sce_data / np.percentile(hist_data, p)
                sce_corrected.T[ipoint][ivalid] = np.percentile(obs_data, p) * delta
                sce_corrected.T[ipoint][ivalid][sce_corrected.T[ipoint][ivalid] < th] = 0
                maxAllowedValue = maxExtrapolationFactor*np.nanmax(obs_data)
                if np.nanmax(sce_corrected.T[ipoint][ivalid] > maxAllowedValue):
                    iOut = np.where(sce_corrected.T[ipoint][ivalid] > maxAllowedValue)
                    sce_corrected.T[ipoint][iOut] = maxAllowedValue

            # For additive corretcion
            else:
                delta = sce_data - np.percentile(hist_data, p)
                sce_corrected.T[ipoint][ivalid] = np.percentile(obs_data, p) + delta

        # obs_auxPlot = np.sort(obs_auxPlot)
        # hist_auxPlot = np.sort(hist_auxPlot)
        # sce_auxPlot = np.sort(sce_auxPlot)
        # pred_data = sce_corrected.T[ipoint]
        # pred_data = np.sort(pred_data)
        #
        # hist_wetDays_freq = np.sum(hist_auxPlot>th) / hist_auxPlot.size
        # sce_wetDays_freq = np.sum(sce_auxPlot>th) / sce_auxPlot.size
        # if hist_wetDays_freq < sce_wetDays_freq:
        #     case = 'need to add wet days'
        # else:
        #     case = 'auto fixed'
        #
        # plt.plot(obs_auxPlot, ECDF(obs_auxPlot)(obs_auxPlot), label='obs', color='k')
        # plt.plot(hist_auxPlot, ECDF(hist_auxPlot)(hist_auxPlot), label='hist', color='b')
        # plt.plot(sce_auxPlot, ECDF(sce_auxPlot)(sce_auxPlot), label='sce', color='green')
        # plt.plot(pred_data, ECDF(pred_data)(pred_data), label='sce_corrected', color='r')
        # # plt.ylim((0,1))
        # plt.title('DQMs '+bc_mode_dict[targetVar]+ '\n' + case)
        # plt.legend()
        # plt.show()
        # exit()

    return sce_corrected


########################################################################################################################
def scaled_distribution_mapping(obs, hist, sce, targetVar, *args, **kwargs):
    """
    Scaled Distribution Mapping (Switanek et al., 2021)
    Parametric adjustment using a gamma distribution for precipitation and a normal distribution for the rest
    For all variables except for precipitation, data is previously detrended, and for precipitation the relative
    frequency is explicitly adjusted.

    Args:
    * obs (nDaysObs, nPoints): the observational data
    * hist (nDaysHist, nPoints): the model data at the reference period
    * sce (nDaysSce, nPoints): the scenario data that shall be corrected

    Kwargs:
    * low_lim (float): assume values below low_lim to be zero (default: 0.1)
    * cdf_th (float): limit of the cdf-values (default: .99999999)
    * min_sample_size (int): minimal number of samples (e.g. wet days) for the gamma fit (default: 10)

    Adapted from https://github.com/wegener-center/pyCAT

    Switanek, M.B., Troch, P.A., Castro, C.L., Leuprecht, A., Chang, H.-I., Mukherjee, R., and Demaria, E.M.C. (2017):
    Scaled distribution mapping: a bias correction method that preserves raw climate model projected changes, Hydrol.
    Earth Syst. Sci., 21, 2649–2666, https://doi.org/10.5194/hess-21-2649-2017
    """

    cdf_th = kwargs.get('cdf_th', 0.99999)
    if targetVar == 'pr':
        # low_lim = kwargs.get('low_lim', 0.1)
        low_lim = kwargs.get('low_lim', 1)
        min_sample_size = kwargs.get('min_sample_size', 10)

    # Define parameters and variables
    nPoints, nDays_ref, nDays_sce = obs.shape[1], obs.shape[0], sce.shape[1]
    sce_corrected = 1*sce

    # Go through all points
    for ipoint in range(nPoints):
        if ipoint % 100 == 0:
            print(ipoint)

        # Select data from one point
        obs_data = obs.T[ipoint]
        hist_data = hist.T[ipoint]
        sce_data = sce.T[ipoint]

        # Remove missing data from obs and hist
        obs_data = obs_data[abs(obs_data - predictands_codification[targetVar]['special_value']) > 0.01]
        hist_data = hist_data[np.isnan(hist_data) == False]

        # Select valid data from sce
        ivalid = np.where(np.isnan(sce_data) == False)
        sce_data = sce_data[ivalid]

        # Temperature
        if targetVar != 'pr':

            # Extract info from data
            obs_lenth = len(obs_data)
            hist_lenth = len(hist_data)
            obs_mean = np.mean(obs_data)
            hist_mean = np.mean(hist_data)

            # Detrend the data
            obs_detrended = detrend(obs_data)
            hist_detrended = detrend(hist_data)

            # Fit normal distribution
            obs_norm = norm.fit(obs_detrended)
            hist_norm = norm.fit(hist_detrended)

            obs_cdf = norm.cdf(np.sort(obs_detrended), *obs_norm)
            hist_cdf = norm.cdf(np.sort(hist_detrended), *hist_norm)
            obs_cdf = np.maximum(np.minimum(obs_cdf, cdf_th), 1 - cdf_th)
            hist_cdf = np.maximum(np.minimum(hist_cdf, cdf_th), 1 - cdf_th)

            sce_lenth = len(sce_data)
            sce_mean = np.mean(sce_data)

            sce_detrended = detrend(sce_data)
            sce_diff = sce_data - sce_detrended
            sce_argsort = np.argsort(sce_detrended)

            sce_norm = norm.fit(sce_detrended)
            sce_cdf = norm.cdf(np.sort(sce_detrended), *sce_norm)
            sce_cdf = np.maximum(np.minimum(sce_cdf, cdf_th), 1 - cdf_th)

            # interpolate cdf-values for obs and hist to the length of the scenario
            obs_cdf_interpol = np.interp(np.linspace(1, obs_lenth, sce_lenth), np.linspace(1, obs_lenth, obs_lenth), obs_cdf)
            hist_cdf_interpol = np.interp(np.linspace(1, hist_lenth, sce_lenth), np.linspace(1, hist_lenth, hist_lenth), hist_cdf)

            # adapt the observation cdfs split the tails of the cdfs around the center
            obs_cdf_shift = obs_cdf_interpol - .5
            hist_cdf_shift = hist_cdf_interpol - .5
            sce_cdf_shift = sce_cdf - .5
            obs_inv = 1. / (.5 - np.abs(obs_cdf_shift))
            hist_inv = 1. / (.5 - np.abs(hist_cdf_shift))
            sce_inv = 1. / (.5 - np.abs(sce_cdf_shift))
            adapted_cdf = np.sign(obs_cdf_shift) * (1. - 1. / (obs_inv * sce_inv / hist_inv))
            adapted_cdf[adapted_cdf < 0] += 1.
            adapted_cdf = np.maximum(np.minimum(adapted_cdf, cdf_th), 1 - cdf_th)

            x_vals = norm.ppf(np.sort(adapted_cdf), *obs_norm) + obs_norm[-1] / hist_norm[-1] \
                    * (norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *hist_norm))
            x_vals -= x_vals.mean()
            x_vals += obs_mean + (sce_mean - hist_mean)

            correction = np.zeros(sce_lenth)
            correction[sce_argsort] = x_vals
            correction += sce_diff - sce_mean
            sce_corrected.T[ipoint][ivalid] = correction

        # Precipitation
        else:

            obs_wetdays = obs_data[obs_data >= low_lim]
            hist_wetdays = hist_data[hist_data >= low_lim]
            sce_wetdays = sce_data[sce_data >= low_lim]

            if obs_wetdays.size < min_sample_size or hist_wetdays.size < min_sample_size or sce_wetdays.size < min_sample_size:
                continue

            obs_freq = 1. * obs_wetdays.shape[0] / obs_data.shape[0]
            hist_freq = 1. * hist_wetdays.shape[0] / hist_data.shape[0]
            obs_gamma = gamma.fit(obs_wetdays, floc=0)
            hist_gamma = gamma.fit(hist_wetdays, floc=0)

            obs_cdf = gamma.cdf(np.sort(obs_wetdays), *obs_gamma)
            hist_cdf = gamma.cdf(np.sort(hist_wetdays), *hist_gamma)
            obs_cdf[obs_cdf > cdf_th] = cdf_th
            hist_cdf[hist_cdf > cdf_th] = cdf_th

            sce_freq = 1. * sce_wetdays.shape[0] / sce_data.shape[0]
            sce_argsort = np.argsort(sce_data)
            sce_gamma = gamma.fit(sce_wetdays, floc=0)

            expected_sce_wetdays = min(
                int(np.round(
                    len(sce_data) * obs_freq * sce_freq
                    / hist_freq)),
                len(sce_data))

            sce_cdf = gamma.cdf(np.sort(sce_wetdays), *sce_gamma)
            sce_cdf[sce_cdf > cdf_th] = cdf_th

            # interpolate cdf-values for obs and hist to the length of the scenario
            obs_cdf_interpol = np.interp(
                np.linspace(1, len(obs_wetdays), len(sce_wetdays)),
                np.linspace(1, len(obs_wetdays), len(obs_wetdays)),
                obs_cdf
            )
            hist_cdf_interpol = np.interp(
                np.linspace(1, len(hist_wetdays), len(sce_wetdays)),
                np.linspace(1, len(hist_wetdays), len(hist_wetdays)),
                hist_cdf
            )

            # adapt the observation cdfs
            obs_inv = 1. / (1 - obs_cdf_interpol)
            hist_inv = 1. / (1 - hist_cdf_interpol)
            sce_inv = 1. / (1 - sce_cdf)
            adapted_cdf = 1 - 1. / (obs_inv * sce_inv / hist_inv)
            adapted_cdf[adapted_cdf < 0.] = 0.

            # correct by adapted observation cdf-values
            x_vals = gamma.ppf(np.sort(adapted_cdf), *obs_gamma) * \
                    gamma.ppf(sce_cdf, *sce_gamma) / \
                    gamma.ppf(sce_cdf, *hist_gamma)

            # interpolate to the expected length of future raindays
            correction = np.zeros(len(sce_data))
            if len(sce_wetdays) > expected_sce_wetdays:
                x_vals = np.interp(
                    np.linspace(1, len(sce_wetdays), expected_sce_wetdays),
                    np.linspace(1, len(sce_wetdays), len(sce_wetdays)),
                    x_vals
                )
            else:
                x_vals = np.hstack(
                    (np.zeros(expected_sce_wetdays -
                              len(sce_wetdays)), x_vals))

            correction[sce_argsort[-expected_sce_wetdays:]] = x_vals
            sce_corrected.T[ipoint][ivalid] = correction


    return sce_corrected



########################################################################################################################
def biasCorrect_as_postprocess(obs, hist, sce, targetVar, ref_times, sce_times):
    """
    This function performs the season selection if needed and call the bc functions.
    * obs (nDaysObs, nPoints): the observational data
    * hist (nDaysHist, nPoints): the model data at the reference period
    * sce (nDaysSce, nPoints): the scenario data that shall be corrected
    :return:
    """

    if targetVar == 'huss':
        print('huss modification /1000...')
        obs /= 1000

    if apply_bc_bySeason == False:
        # Correct bias
        if bc_method == 'QM':
            scene_bc = quantile_mapping(obs, hist, sce, targetVar)
        elif bc_method == 'DQM':
            scene_bc = detrended_quantile_mapping(obs, hist, sce, targetVar)
        elif bc_method == 'QDM':
            scene_bc = quantile_delta_mapping(obs, hist, sce, targetVar)
        elif bc_method == 'PSDM':
            scene_bc = scaled_distribution_mapping(obs, hist, sce, targetVar)
    else:
        # print(obs.shape, hist.shape, sce.shape)

        scene_bc = np.zeros(sce.shape)

        # Select season
        for season in season_dict:
            if season != annualName:
                print('bias correction by season', season)
                obs_season = postpro_lib.get_season(obs, ref_times, season)['data']
                hist_season = postpro_lib.get_season(hist, ref_times, season)['data']
                aux = postpro_lib.get_season(sce, sce_times, season)
                sce_season = aux['data']
                sce_times_season = aux['times']
                idates = [i for i in range(len(sce_times)) if sce_times[i] in sce_times_season]

                # print('season', season, obs_season.shape, hist_season.shape, sce_season.shape, len(idates))

                # Correct bias for season
                if bc_method == 'QM':
                    scene_bc[idates] = quantile_mapping(obs_season, hist_season, sce_season, targetVar)
                elif bc_method == 'DQM':
                    scene_bc[idates] = detrended_quantile_mapping(obs_season, hist_season, sce_season, targetVar)
                elif bc_method == 'QDM':
                    scene_bc[idates] = quantile_delta_mapping(obs_season, hist_season, sce_season, targetVar)
                elif bc_method == 'PSDM':
                    scene_bc[idates] = scaled_distribution_mapping(obs_season, hist_season, sce_season, targetVar)

    # Force to theoretical range
    minAllowed, maxAllowed = predictands_range[targetVar]['min'], predictands_range[targetVar]['max']
    if minAllowed is not None:
        scene_bc[scene_bc < minAllowed] = minAllowed
    if maxAllowed is not None:
        scene_bc[scene_bc > maxAllowed] = maxAllowed

    return scene_bc