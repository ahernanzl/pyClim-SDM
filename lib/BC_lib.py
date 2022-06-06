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
def quantile_mapping(obs, hist, sce, var):
    """
   Quantile Mapping. Basic version of empirical QM (Themeßl et al., 2011). Additive correction both for temperature and
   for precipitation.

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
        # if ipoint % 100 == 0:
            # print(ipoint)

        # Select data from one point
        obs_data = obs.T[ipoint]
        hist_data = hist.T[ipoint]
        sce_data = sce.T[ipoint]

        # Remove missing data from obs and hist
        obs_data, hist_data = obs_data[np.isnan(obs_data) == False], hist_data[np.isnan(hist_data) == False]

        # Select valid data from sce
        ivalid = np.where(np.isnan(sce_data) == False)
        sce_data = sce_data[ivalid]

        # Calculate correction
        hist_ecdf = ECDF(hist_data)
        p = hist_ecdf(sce_data) * 100
        corr = np.percentile(obs_data, p) - np.percentile(hist_data, p)

        # Add correction
        sce_corrected.T[ipoint][ivalid] = sce_data + corr

    # For precipitation, set negative values to zero
    if var == 'pcp':
        sce_corrected[sce_corrected < 0] = 0

    return sce_corrected




########################################################################################################################
def detrended_quantile_mapping(obs, hist, sce, var, th=0.05):
    """
    Detrendend Quantile Mapping: remove trend and mean, and then apply empirical quantile mapping (Cannon et al., 2015).
    Additive correction for temperature and ratio correction for precipitation.

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
        # if ipoint % 100 == 0:
            # print(ipoint)

        # Select data from one point
        obs_data = obs.T[ipoint]
        hist_data = hist.T[ipoint]
        sce_data = sce.T[ipoint]

        # Remove missing data from obs and hist
        obs_data, hist_data = obs_data[np.isnan(obs_data) == False], hist_data[np.isnan(hist_data) == False]

        # Select valid data from sce
        ivalid = np.where(np.isnan(sce_data) == False)
        sce_data = sce_data[ivalid]

        if var[0] == 't':
            # Remove mean and detrend
            obs_detrended = detrend(obs_data)
            hist_detrended = detrend(hist_data)
            sce_detrended = detrend(sce_data)

            # Calculate correction using data without mean and trend
            hist_ecdf = ECDF(hist_detrended)
            p = hist_ecdf(sce_detrended) * 100
            corr = np.percentile(obs_detrended, p) - np.percentile(hist_detrended, p)

            # Add correction and mean values removed while detrending
            sce_corrected.T[ipoint][ivalid] = sce_data + corr + np.mean(obs) - np.mean(hist)

        else:
            # Treat zeros
            obs_data[obs_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(obs_data < th)[0].shape))
            hist_data[hist_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(hist_data < th)[0].shape))
            sce_data[sce_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(sce_data < th)[0].shape))

            # Remove change in the mean value
            sce_mean = np.mean(sce_data)
            sce_data *= np.mean(hist) / np.mean(sce)

            # Calculate correction
            hist_ecdf = ECDF(hist_data)
            p = hist_ecdf(sce_data) * 100
            corr = np.percentile(obs_data, p) / np.percentile(hist_data, p)

            # Add correction and change in the mean value
            sce_corrected.T[ipoint][ivalid] = sce_data * corr * sce_mean / np.mean(hist)

    # For precipitation, set negative values to zero
    if var == 'pcp':
        sce_corrected[sce_corrected < th] = 0

    return sce_corrected


########################################################################################################################
def quantile_delta_mapping(obs, hist, sce, var, th=0.05, jitter=0.01):
    """
    Quantile Delta Mapping: apply delta change correction to all quantiles (Cannon et al., 2015).
    Additive correction for temperature and ratio correction for precipitation.

    Args:
    * obs (nDaysObs, nPoints): the observational data
    * hist (nDaysHist, nPoints): the model data at the reference period
    * sce (nDaysSce, nPoints): the scenario data that shall be corrected

    Adapted from https://github.com/pacificclimate/ClimDown

    Cannon, A.J., S.R. Sobie, and T.Q. Murdock (2015) Bias Correction of GCM Precipitation by Quantile Mapping: How Well
    Do Methods Preserve Changes in Quantiles and Extremes?. J. Climate, 28, 6938–6959,
    https://doi.org/10.1175/JCLI-D-14-00754.1
    """

    # if var == 'pcp':
    #     print('quantile_delta_mapping only implemented with additive correction for temperature')
    #     print('Current version not recomended for precipitation')
    #     exit()

    # Define parameters and variables
    nPoints, nDays_ref, nDays_sce = obs.shape[1], obs.shape[0], sce.shape[1]
    sce_corrected = 1*sce

    # Add a small amount of noise to accomodate ties due to limited precision
    obs += np.random.uniform(low=-jitter, high=jitter, size=obs.shape)
    hist += np.random.uniform(low=-jitter, high=jitter, size=hist.shape)
    sce += np.random.uniform(low=-jitter, high=jitter, size=sce.shape)

    # Go through all points
    for ipoint in range(nPoints):
        # if ipoint % 100 == 0:
            # print(ipoint)

        # Select data from one point
        obs_data = obs.T[ipoint]
        hist_data = hist.T[ipoint]
        sce_data = sce.T[ipoint]

        # Remove missing data from obs and hist
        obs_data, hist_data = obs_data[np.isnan(obs_data) == False], hist_data[np.isnan(hist_data) == False]

        # Select valid data from sce
        ivalid = np.where(np.isnan(sce_data) == False)
        sce_data = sce_data[ivalid]

        # Treat zeros
        if var == 'pcp':
            obs_data[obs_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(obs_data < th)[0].shape))
            hist_data[hist_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(hist_data < th)[0].shape))
            sce_data[sce_data < th] = np.random.uniform(low=0.0001, high=th, size=(np.where(sce_data < th)[0].shape))

        # Calculate percentiles
        sce_ecdf = ECDF(sce_data)
        p = sce_ecdf(sce_data) * 100

        # Calculate and apply delta correction
        if var[0] == 't':
            delta = sce_data - np.percentile(hist_data, p)
            sce_corrected.T[ipoint][ivalid] = np.percentile(obs_data, p) + delta
        else:
            delta = sce_data / np.percentile(hist_data, p)
            sce_corrected.T[ipoint][ivalid] = np.percentile(obs_data, p) * delta

    # For precipitation, set negative values to zero
    if var == 'pcp':
        sce_corrected[sce_corrected < th] = 0

    return sce_corrected


########################################################################################################################
def scaled_distribution_mapping(obs, hist, sce, var, *args, **kwargs):
    """
    Scaled Distribution Mapping (Switanek et al., 2021)
    Parametric adjustment using a normal distribution for temperature and a gamma distribution for precipitation.
    For temperature, data is previously detrended, and for precipitation the relative frequency is explicitly
    adjusted.

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
    if var == 'pcp':
        low_lim = kwargs.get('low_lim', 0.1)
        min_sample_size = kwargs.get('min_sample_size', 10)

    # Define parameters and variables
    nPoints, nDays_ref, nDays_sce = obs.shape[1], obs.shape[0], sce.shape[1]
    sce_corrected = 1*sce

    # Go through all points
    for ipoint in range(nPoints):
        # if ipoint % 100 == 0:
        #     print(ipoint)

        # Select data from one point
        obs_data = obs.T[ipoint]
        hist_data = hist.T[ipoint]
        sce_data = sce.T[ipoint]

        # Remove missing data from obs and hist
        obs_data, hist_data = obs_data[np.isnan(obs_data) == False], hist_data[np.isnan(hist_data) == False]

        # Select valid data from sce
        ivalid = np.where(np.isnan(sce_data) == False)
        sce_data = sce_data[ivalid]

        # Temperature
        if var[0] == 't':

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
        elif var == 'pcp':

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

            # Correct negative values
            sce_corrected[sce_corrected < 0] = 0

    return sce_corrected



########################################################################################################################
def biasCorrect_as_postprocess(obs, hist, sce, var, ref_times, sce_times):
    """
    This function performs the season selection if needed and call the bc functions.
    * obs (nDaysObs, nPoints): the observational data
    * hist (nDaysHist, nPoints): the model data at the reference period
    * sce (nDaysSce, nPoints): the scenario data that shall be corrected
    :return:
    """


    if apply_bc_bySeason == False:
        # Correct bias
        if bc_method == 'QM':
            scene_bc = quantile_mapping(obs, hist, sce, var)
        elif bc_method == 'DQM':
            scene_bc = detrended_quantile_mapping(obs, hist, sce, var)
        elif bc_method == 'QDM':
            scene_bc = quantile_delta_mapping(obs, hist, sce, var)
        elif bc_method == 'PSDM':
            scene_bc = scaled_distribution_mapping(obs, hist, sce, var)
    else:
        # print(obs.shape, hist.shape, sce.shape)

        scene_bc = np.zeros(sce.shape)

        # Select season
        for season in season_dict.values():
            if season != 'ANNUAL':
                obs_season = postpro_lib.get_season(obs, ref_times, season)['data']
                hist_season = postpro_lib.get_season(hist, ref_times, season)['data']
                aux = postpro_lib.get_season(sce, sce_times, season)
                sce_season = aux['data']
                sce_times_season = aux['times']
                idates = [i for i in range(len(sce_times)) if sce_times[i] in sce_times_season]

                # print(season, obs_season.shape, hist_season.shape, sce_season.shape, len(idates))

                # Correct bias for season
                if bc_method == 'QM':
                    scene_bc[idates] = quantile_mapping(obs_season, hist_season, sce_season, var)
                elif bc_method == 'DQM':
                    scene_bc[idates] = detrended_quantile_mapping(obs_season, hist_season, sce_season, var)
                elif bc_method == 'QDM':
                    scene_bc[idates] = quantile_delta_mapping(obs_season, hist_season, sce_season, var)
                elif bc_method == 'PSDM':
                    scene_bc[idates] = scaled_distribution_mapping(obs_season, hist_season, sce_season, var)

    return scene_bc