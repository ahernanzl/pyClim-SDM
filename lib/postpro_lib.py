import sys
sys.path.append('../config/')
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


########################################################################################################################
def calculate_all_climdex(pathOut, filename, var, data, times, ref, times_ref):
    """
    Calculate all climdex/season and save them into files.
    """
    start = datetime.datetime.now()

    # Create path for results
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    # Select climdex
    for climdex_name in climdex_names[var]:
        print(climdex_name)

        # Get percentile calendars
        if climdex_name in ('TX90p', 'TN90p', 'WSDI'):
            percCalendar = get_perc_calendar(var, times_ref, ref, 90)
        elif climdex_name in ('TX10p', 'TN10p', 'CSDI'):
            percCalendar = get_perc_calendar(var, times_ref, ref, 10)
        elif climdex_name in ('R95p', 'R95pFRAC'):
            # Five values are to be calculated, one for each season. With each value a calendar of 365 days is built
            perc95Calendar = {}
            for season in season_dict.values():
                aux = get_season(ref, times_ref, season)['data']
                aux[aux < 1] = np.nan
                perc95Calendar.update({season: np.repeat(np.nanpercentile(aux, 95, axis=0)[np.newaxis, :], 365, axis=0)})
        elif climdex_name in ('R99p', 'R99pFRAC'):
            # Five values are to be calculated, one for each season. With each value a calendar of 365 days is built
            perc99Calendar = {}
            for season in season_dict.values():
                aux = get_season(ref, times_ref, season)['data']
                aux[aux < 1] = np.nan
                perc99Calendar.update({season: np.repeat(np.nanpercentile(aux, 99, axis=0)[np.newaxis, :], 365, axis=0)})
        else:
            percCalendar = np.zeros((365, ref.shape[1]))


        # Select season
        for season in season_dict.values():
            aux = get_season(data, times, season)
            data_season = aux['data']
            times_season = aux['times']
            times_percCalendar = [datetime.date(1981, 1, 1) + datetime.timedelta(days=i) for i in range(365)]
            if climdex_name in ('R95p', 'R95pFRAC'):
                percCalendar = perc95Calendar[season]
            elif climdex_name in ('R99p', 'R99pFRAC'):
                percCalendar = perc99Calendar[season]
            aux = get_season(percCalendar, times_percCalendar, season)
            percCalendar_season = aux['data']
            times_percCalendar_seaon = aux['times']
            del aux

            # Calculate climdex for obs and est
            data_climdex = calculate_climdex(climdex_name, data_season, percCalendar_season,
                                                  times_season, times_percCalendar_seaon)['data']

            # Save results
            np.save(pathOut+'_'.join((climdex_name, filename, season)), data_climdex)

    print(var, filename, 'calculate_all_climdex', str(datetime.datetime.now() - start))


########################################################################################################################
def calculate_climdex(climdex_name, data, ref, times, times_ref):
    """
    Calculate climdex_name of input data. (https://www.climdex.org/learn/indices/)
    All climdex correspond to 1 value per year/season.
    Some indexes have two different definitions, one taken from the climdex web page and the other inherited from
    previous woks.

    TXm: mean tmax

    TNm: mean tmin

    Pm: mean pcp

    FD: Number of frost days. Annual count of days when TN (daily minimum temperature) < 0°C.
        Let TNij be daily minimum temperature on day i in year j. Count the number of days where TNij < 0 °C.

    SU: Number of summer days. Annual count of days when TX (daily maximum temperature) > 25°C. Let TXij be daily
        maximum temperature on day i in year j. Count the number of days where TXij > 25 °C.

    ID: Number of icing days. Annual count of days when TX (daily maximum temperature) < 0 °C. Let TXijbe daily maximum
        temperature on day i in year j. Count the number of days where TXij < 0 °C.

    TR: Number of tropical nights. Annual count of days when TN (daily minimum temperature) > 20 °C. Let TNij be daily
        minimum temperature on day i in year j. Count the number of days where TNij > 20 °C.

    TN10p: Percentage of days when TN < 10th percentile
        Let TNij be the daily minimum temperature on day i in period j and let TNin10 be the calendar day 10th
        percentile centred on a 5-day window for the base period 1961-1990. The percentage of time for the base period
        is determined where: TNij < TNin10. To avoid possible inhomogeneity across the in-base and out-base periods, the
         calculation for the base period (1961-1990) requires the use of a bootstrap procedure. Details are described in
        Zhang et al. (2005).

    TN90p: Percentage of days when TN < 90th percentile (see TN10p)

    TX10p: Percentage of days when TX < 10th percentile (see TN10p)

    TX90p: Percentage of days when TX < 90th percentile (see TN10p)

    WSDI:
        Warm spell duration index: annual count of days with at least 6 consecutive days when TX > 90th percentile.
        Let TXij be the daily maximum temperature on day i in period j and let TXin90 be the calendar day 90th
        percentile centred on a 5-day window for the base period 1961-1990. Then the number of days per period is summed
        where, in intervals of at least 6 consecutive days, TXij > TXin90.

    CSDI: Cold spell duration index: annual count of days with at least 6 consecutive days when TN < 10th percentile
        (see WSDI)

    SDII: Simple precipitation intensity index
        Let RRwj be the daily precipitation amount on wet days, w (RR ≥ 1mm) in period j. If W represents number of wet
        days in j, then: SDIIj=∑Ww=1RRwj/W

    R01: Annual count of days when PRCP ≥ 1mm
        Let RRij be the daily precipitation amount on day i in period j. Count the number of days where RRij ≥ 1mm

    CDD: Maximum length of dry spell: maximum number of consecutive days with RR < 1mm
        Let RRij be the daily precipitation amount on day i in period j. Count the largest number of consecutive days
        where RRij < 1mm.

    CWD: Maximum length of wet spell: maximum number of consecutive days with RR ≥ 1mm
        Let RRij be the daily precipitation amount on day i in period j. Count the largest number of consecutive days
        where RRij ≥ 1mm.

    R95p:
        Annual total PRCP when RR > 95th percentile
        Let RRwj be the daily precipitation amount on a wet day w (RR ≥ 1.0mm) in period i and let RRwn95 be the 95th
        percentile of precipitation on wet days in the 1961-1990 period. If W represents the number of wet days in the
        period, then: R95p=W∑w=1RRwjwhereRRwj>RRwn95

    R95pFRAC:
        R95p / PRCPTOT

    R99p:
        Annual total PRCP when RR > 99th percentile
        Let RRwj be the daily precipitation amount on a wet day w (RR ≥ 1.0mm) in period i and let RRwn99 be the 99th
        percentile of precipitation on wet days in the 1961-1990 period. If W represents the number of wet days in the
        period, then: R99p=W∑w=1RRwjwhereRRwj>RRwn99

    R99pFRAC:
        R99p / PRCPTOT

    PRCPTOT: Annual total precipitation on wet days
        Let RRij be the daily pre If i represents the number of days in j, then: PRCPTOTj=I∑i=1RRij

    p1, p5, p10, p90, p95, p99: percentiles.

    """

    warnings.filterwarnings("ignore", message="invalid value encountered in greater")
    warnings.filterwarnings("ignore", message="invalid value encountered in less")
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    if climdex_name in ('p1', 'p5', 'p10', 'p90', 'p95', 'p99'):
        times_results = [x for x in range(times[0].year, times[-1].year + 1)]
        results = np.nanpercentile(data, int(climdex_name[1:]), axis=0)
        results = results[np.newaxis, :]
        results = np.repeat(results, len(times_results), axis=0)

    else:
        # Get years range
        minYear, maxYear = times[0].year, times[-1].year
        nYears = 1 + maxYear - minYear

        # Sepecific calculations for each climdex
        results, times_results = [], []

        # Go through all years
        for iyear in range(nYears):

            # Select data from year
            year = minYear + iyear
            times_results.append(year)
            data_year = data[[i for i in range(len(times)) if times[i].year == year]]
            times_year = [x for x in times if x.year == year]
            data_nDays, ref_nDays = data_year.shape[0], ref.shape[0]

            # Deal with leap years
            if data_nDays == ref_nDays:
                aux = 1*ref
            elif data_nDays == ref_nDays + 1:
                i_28feb = [times_ref.index(x) for x in times_ref if ((x.month == 2) and (x.day == 28))][0]
                aux = 0*data_year
                aux[:i_28feb+1] = ref[:i_28feb+1]
                aux[i_28feb+1] = ref[i_28feb]
                aux[i_28feb+1:] = ref[i_28feb:]
            else:
                exit(times_year, len(times_year), ref_nDays)

            # # Remove days with nan
            # iNans = np.unique(np.where(np.isnan(data_year))[0])
            # data_year = np.delete(data_year, iNans, axis=0)
            # aux = np.delete(aux, iNans, axis=0)

            # Calculate climdex for iyear
            if climdex_name in ('TXm', 'TNm', 'Pm'):
                results.append(np.nanmean(data_year, axis=0))
            elif climdex_name in ('TX90p', 'TN90p'):
                results.append(np.nanmean(100.*(data_year > aux), axis=0))
            elif climdex_name in ('TX10p', 'TN10p'):
                results.append(np.nanmean(100.*(data_year < aux), axis=0))
            elif climdex_name in ('TXx', 'TNx'):
                results.append(np.nanmax(data_year, axis=0))
            elif climdex_name in ('TXn', 'TNn'):
                results.append(np.nanmin(data_year, axis=0))
            elif climdex_name == 'WSDI':
                results.append(get_spell_duration(data_year > aux, climdex_name))
            elif climdex_name == 'CSDI':
                results.append(get_spell_duration(data_year < aux, climdex_name))
            elif climdex_name == 'FD':
                results.append(np.nansum(data_year < 0, axis=0))
            elif climdex_name == 'SU':
                results.append(np.nansum(data_year > 25, axis=0))
            elif climdex_name == 'ID':
                results.append(np.nansum(data_year < 0, axis=0))
            elif climdex_name == 'TR':
                results.append(np.nansum(data_year > 20, axis=0))
            elif climdex_name == 'R01':
                results.append(np.nansum(data_year >= 1, axis=0))
            elif climdex_name == 'SDII':
                # nWetDays = np.nansum(data_year >= 1, axis=0)
                # nWetDays[nWetDays == 0] = 0.001
                # data_year[data_year < 1] = 0
                # results.append(np.nansum(data_year, axis=0) / nWetDays)
                data_year[data_year < 1] = np.nan
                sdii = np.nanmean(data_year, axis=0)
                sdii[np.isnan(sdii)] = 0
                results.append(sdii)
            elif climdex_name == 'PRCPTOT':
                data_year[data_year < 1] = 0
                results.append(np.nansum(data_year, axis=0))
            elif climdex_name == 'CDD':
                results.append(get_spell_duration(data_year < 1, climdex_name))
            elif climdex_name in ('R95p', 'R99p'):
                data_year[data_year < aux] = 0
                data_year[data_year < 1] = 0
                results.append(np.nansum(data_year, axis=0))
            elif climdex_name in ('R95pFRAC', 'R99pFRAC'):
                data_year[data_year < 1] = 0
                total = np.nansum(data_year, axis=0)
                data_year[data_year < aux] = 0
                heavy = np.nansum(data_year, axis=0)
                total[total == 0] = -1
                heavy[total == -1] = 0
                results.append(100.*heavy/total)
            elif climdex_name == 'CWD':
                results.append(get_spell_duration(data_year >= 1, climdex_name))
            elif climdex_name == 'Rx1day':
                results.append(np.nanmax(data_year,  axis=0))
            elif climdex_name == 'Rx5day':
                acc5days = data_year[0:-4] + data_year[1:-3] + data_year[2:-2] + data_year[3:-1] + data_year[4:]
                results.append(np.nanmax(acc5days,  axis=0))
            elif climdex_name == 'R10mm':
                results.append(np.nansum(data_year >= 10, axis=0))
            elif climdex_name == 'R20mm':
                results.append(np.nansum(data_year >= 20, axis=0))

        results = np.asarray(results)

    return {'data': results, 'times': times_results}


########################################################################################################################
def get_perc_calendar(var, times, data, q):
    """
    times: list
    data: numpy array (ntimes, npoints)
    :return: percCalendar: numpy array (365, npoints)
    """

    data = (100 * data).astype(predictands_codification[var]['type'])

    # Create empty percCalendar
    percCalendar = np.zeros((365, data.shape[1]))

    # Create 5 days window
    data5 = np.repeat(data[np.newaxis, :], 5, axis=0)
    data5[0][:-2] = data[2:]
    data5[1][:-1] = data[1:]
    data5[3][1:] = data[:-1]
    data5[4][2:] = data[:-2]
    del(data)

    # Create year dates
    yearDates = [datetime.date(1981, 1, 1) + datetime.timedelta(days=i) for i in range(365)]

    # For each day of the year
    for idate in range(365):
        # Select same day from all years and calculates percentile
        idates = [i for i in range(len(times)) if ((times[i].month==yearDates[idate].month) and (times[i].day==yearDates[idate].day))]
        auxData = np.swapaxes(np.swapaxes(data5, 0, 1)[idates], 0, 1).T.reshape(data5.shape[-1], -1).T
        percCalendar[idate] = np.nanpercentile(auxData, q, axis=0)

    percCalendar = percCalendar.astype('float64') / 100.

    return percCalendar


########################################################################################################################
def get_season(data, times, season):
    """
    :param data: (ndays, npoints)
    :param times: list of dates
    :param season: 'ANNUAL', 'DJF', 'MMA', 'JJA', 'SON' taken from season_dict defined at settings
    :return: data: (ndays_season, npoints)
            times_season: list of dates season
    """

    # Define season months
    season_months = {season_dict['ANNUAL']: range(1, 13), season_dict['DJF']: (12, 1, 2), season_dict['MAM']: (3, 4, 5), season_dict['JJA']: (6, 7, 8), season_dict['SON']: (9, 10, 11)}

    # Get season positions
    idates = [i for i in range(len(times)) if times[i].month in season_months[season]]
    data = data[idates]
    times = [times[i] for i in idates]

    return {'data': data, 'times': times}


########################################################################################################################
def get_spell_duration(a, climdex_name):
    """
    :param a: (nDays, nPoints)
    :param climdex_name:
    :return: (nPoints): for each point, the sum of days with value 1, only counting those inside groups of at least 6
        consequtive.
    """

    nPoints = a.shape[1]

    # Define threshold for warm spell (in order to be considered warm spell, the number of consecutive warm days has to
    # overpass this threshold
    if climdex_name in ('WSDI', 'CSDI'):
        wsd_th = 6 # as in climdex web page
        # wsd_th = 5 # as in Guia AR5
    elif climdex_name in ('CDD', 'CWD'):
        wsd_th = 0

    a = a.T
    iszero = np.asarray(np.zeros((a.shape[0], a.shape[1] + 2)), dtype='int64')
    iszero[:, 1:-1] = a
    absdiff = np.abs(np.diff(iszero))
    lens = np.diff(np.where(absdiff == 1)[1].reshape(-1, 2)).T[0]
    irow = np.where(absdiff == 1)[0][::2][lens >= wsd_th]
    lens = lens[lens >= wsd_th]

    df = pd.DataFrame({'ipoint': irow, 'lens': lens})
    if climdex_name in ('WSDI', 'CSDI'):
        df = df.groupby('ipoint').agg('sum') # as in climdex web page
        # df = df.groupby('ipoint').agg('max') # as in Guia AR5
    elif climdex_name in ('CDD', 'CWD'):
        df = df.groupby('ipoint').agg('max')

    ipoints = df.index.values
    results = np.zeros((nPoints))
    results[ipoints] = df['lens'].values.astype(float)

    return results

########################################################################################################################
def get_data_eval(var, methodName):
    """
    Reads data for evaluation.
    :return: dictionaty with 'ref', 'times_ref', 'obs', 'est', 'times_scene', 'path'
    """

    pathIn = '../results/EVALUATION/' + var.upper() + '/' + methodName + '/daily_data/'
    aux = read.hres_data(var, period='testing')
    times_scene = aux['times']
    obs = aux['data']
    est = read.netCDF(pathIn, 'reanalysis_TESTING.nc', var)['data']
    del aux

    special_value = predictands_codification[var]['special_value']
    obs[obs==special_value] = np.nan
    aux = read.hres_data(var, period='reference')
    ref = aux['data']
    times_ref = aux['times']

    return {'ref': ref, 'times_ref': times_ref, 'obs': obs, 'est': est, 'times_scene': times_scene}


########################################################################################################################
def get_data_projections(nYears, npoints, climdex_name, season, pathIn, iaux):
    """
    Get data projections. Return ssp_dict with model names and change.
    """

    if experiment == 'EVALUATION':
        print('Invalid experiment for get_data_projections')
        exit()

    ssp_dict = {}
    for scene in scene_list:
        if scene != 'historical':
            models = []
            all_data = np.zeros((0, nYears, npoints))
            for model in model_list:
                fileIn = '_'.join((climdex_name, scene, model, season)) + '.npy'
                # Check if scene/model exists
                if os.path.isfile(pathIn + fileIn):
                    # Read data and select region
                    data = np.load(pathIn + fileIn)[:, iaux]
                    ref = np.load(
                        pathIn + '_'.join((climdex_name, 'REFERENCE', model, season)) + '.npy')[:, iaux]
                    ref_mean = np.repeat(np.mean(ref, axis=0)[np.newaxis, :], nYears, axis=0)

                    # # Plot reference mean maps for climdex control, to detect possible errors
                    # if (regType == typeCompleteRegion) and (plotAllRegions == True):
                    #     title = ' '.join((climdex_name, 'REFERENCE', model, season))
                    #     plot.map(ref_mean[0], None,
                    #              path=pathOut + 'reference_mean_maps/' + climdex_name + '/' +
                    #                   season + '/', filename=model, title=title)
                    #     plt.close()

                    # Calculate change (absolute)
                    if climdex_name in ('TXm', 'TX90p', 'TX10p', 'TXx', 'TXn', 'WSDI',
                                        'TNm', 'TN90p', 'TN10p', 'TNx', 'TNn', 'CSDI', 'FD',
                                        'CDD', 'CWD', 'R01', 'R95p', 'R95pFRAC', 'R99p', 'R99pFRAC'):
                        change = data - ref_mean

                    # Relative
                    elif climdex_name in ('Pm', 'PRCPTOT', 'SDII'):
                        ref_mean[ref_mean == 0] = 0.001
                        change = 100 * (data - ref_mean) / ref_mean

                    else:
                        # change = np.nan
                        change = np.zeros((data.shape))

                    del data, ref, ref_mean

                    # Accumulate results
                    models.append(model)
                    all_data = np.append(all_data, change[np.newaxis, :, :], axis=0)

            # Acumulate results
            ssp_dict.update({scene: {'models': models, 'data': all_data}})

    return ssp_dict



########################################################################################################################
def figures_projections(lan='EN'):
    """
    Read previously calculated climdex, change them into anomalies and plot maps and evolution graphs.

    For maps, each model/scene is reduced to a single value for all the period (2 decades), and then the meand and std
        among all models is plotted.

    For evolution graphs, all the region is reduced to one value per year/model/scene.
    """

    if bc_method == None:
        sufix = ''
    else:
        sufix = '_BC-' + bc_method


    implemented_climdex = ['TXm', 'TXx', 'TXn', 'TX90p', 'TX10p', 'WSDI', 
                           'TNm', 'TNx', 'TNn', 'TN90p', 'TN10p', 'CSDI', 'FD',
                           'Pm', 'R01', 'SDII', 'PRCPTOT', 'CDD', 'R95p', 'R95pFRAC', 'R99p', 'R99pFRAC', 'CWD']

    # Go through all methods
    for method_dict in methods:
        var, methodName = method_dict['var'], method_dict['methodName']
        print('figures_projections', var, methodName)

        # Define and create paths
        path = '../results/PROJECTIONS'+sufix+'/' + var.upper() + '/' + methodName + '/'
        pathIn = path + 'climdex/'
        pathRaw = '../results/PROJECTIONS'+sufix+'/' + var.upper() + '/RAW/climdex/'
        pathOut = path + 'climdex/figures/'
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Read regions csv
        df_reg = pd.read_csv(pathAux + 'ASSOCIATION/regions.csv')

        # Define ylimit and ylabel
        # ylim_dict = {'TXm': (-1, 10), 'TX90p': (0, 70), 'TX10p': (-70, 0), 'WSDI': (-20, 80),
        #              'TNm': (-1, 10), 'TN90p': (0, 70), 'TN10p': (-70, 0), 'CSDI': (-80, 20), 'FD': (-60, 20),
        #              'Pm': (-60, 60), 'R01': (-60, 20), 'SDII': (-60, 60), 'PRCPTOT': (-60, 60),
        #              'CDD': (-20, 60), 'R95p': (-20, 20), 'R95pFRAC': (-20, 20), 'R99p': (-100, 100), 'R99pFRAC': (-100, 100), 'CWD': (-60, 20)}
        ylim_dict = {'TXm': (-1, 10), 'TXx': (-1, 12), 'TXn': (-1, 8), 'TX90p': (0, 70), 'TX10p': (-70, 0), 'WSDI': (0, 300),
                     'TNm': (-1, 10), 'TNx': (-1, 12), 'TNn': (-1, 8), 'TN90p': (0, 70), 'TN10p': (-70, 0), 'CSDI': (-80, 20), 'FD': (-60, 20),
                     'Pm': (-60, 60), 'R01': (-80, 40), 'SDII': (-40, 40), 'PRCPTOT': (-60, 20),
                     'CDD': (-20, 60), 'R95p': (-100, 100), 'R95pFRAC': (-100, 100), 'CWD': (-60, 20),
                     'R99p': (-100, 100), 'R99pFRAC': (-100, 100)}

        if lan == 'ES':
            xlabel = 'Año'
            ylabel_dict = {'TXm': 'Cambio en la temperatura máxima (' + degree_sign + ')',
                           'TXx': 'Cambio en la máxima de la temperatura máxima (' + degree_sign + ')',
                           'TXn': 'Cambio en la mínima de la temperatura máxima (' + degree_sign + ')',
                           'TX90p': 'Cambio en días cálidos (%)',
                           'TX10p': 'Cambio en días fríos (%)',
                           'WSDI': 'Cambio duración olas de calor (días)',
                           'TNm': 'Cambio en la temperatura mínima (' + degree_sign + ')',
                           'TNx': 'Cambio en la máxima de la temperatura mínima (' + degree_sign + ')',
                           'TNn': 'Cambio en la mínima de la temperatura mínima (' + degree_sign + ')',
                           'TN90p': 'Cambio en noches cálidas (%)',
                           'TN10p': 'Cambio en noches frías (%)',
                           'CSDI': 'Cambio duración olas de frío (días)',
                           'FD': 'Cambio en número de días de helada (días)',
                           'Pm': 'Cambio en la precipitación (%)',
                           'R01': 'Cambio en número de días de lluvia (días)',
                           'SDII': 'Cambio en la precipitación (%)',
                           'PRCPTOT': 'Cambio en la precipitación total (%)',
                           'CDD': 'Cambio duración periodo seco (dias)',
                           'R95p': 'Cambio en precipitaciones intensas (%)',
                           'R95pFRAC': 'Cambio en precipitaciones intensas (%)',
                           'R99p': 'Cambio en precipitaciones intensas (%)',
                           'R99pFRAC': 'Cambio en precipitaciones intensas (%)',
                           'CWD': 'Cambio duración periodo húmedo (días)'}
        elif lan == 'EN':
            xlabel = 'year'
            ylabel_dict = {'TXm': 'Change in TXm (' + degree_sign + ')',
                           'TXx': 'Change in TXx (' + degree_sign + ')',
                           'TXn': 'Change in TXn (' + degree_sign + ')',
                           'TX90p': 'Change in TX90p (%)',
                           'TX10p': 'Change in TX10p (%)',
                           'WSDI': 'Change in WSDI (days)',
                           'TNm': 'Change in TNm (' + degree_sign + ')',
                           'TNx': 'Change in TNx (' + degree_sign + ')',
                           'TNn': 'Change in TNn (' + degree_sign + ')',
                           'TN90p': 'Change in TN90p (%)',
                           'TN10p': 'Change in TN10p (%)',
                           'CSDI': 'Change in CSDI (days)',
                           'FD': 'Change in FD (days)',
                           'Pm': 'Change in Pm (%)',
                           'R01': 'Change in R01 (days)',
                           'SDII': 'Change in SDII (%)',
                           'PRCPTOT': 'Change in PRCPTOT (%)',
                           'CDD': 'Change in CDD (dias)',
                           'R95p': 'Change in R95p (%)',
                           'R95pFRAC': 'Change in R95pFRAC (%)',
                           'R99p': 'Change in R99p (%)',
                           'R99pFRAC': 'Change in R99pFRAC (%)',
                           'CWD': 'Change in CWD (days)'}

        # Define years
        years = [x for x in range(ssp_years[0], ssp_years[1] + 1)]
        nYears = len(years)

        # Go through all regions
        for index, row in df_reg.iterrows():
            if plotAllRegions == True or ((plotAllRegions == False) and (index == 0)):
                regType, regName, subDir = row['regType'], row['regName'], row['subDir']
                iaux = [int(x) for x in row['ipoints'][1:-1].split(', ')]
                npoints = len(iaux)
                print('-----------------------------------------------------------------')
                print(regType, regName, npoints, str(index) + '/' + str(df_reg.shape[0]))

                # Go through all climdex, seasons, scenes and models
                for climdex_name in climdex_names[var]:
                    if climdex_name in implemented_climdex:
                        for season in season_dict.values():

                            # Get data
                            ssp_dict = get_data_projections(nYears, npoints, climdex_name, season, pathIn, iaux)
                            raw_ssp_dict = get_data_projections(nYears, npoints, climdex_name, season, pathRaw, iaux)

                            # Evolution figures of mean trend in the whole region vs RAW
                            if (regType == typeCompleteRegion):
                                trend_raw(pathOut, subDir, ssp_dict['ssp585'], raw_ssp_dict['ssp585'], climdex_name, years,
                                          ylim_dict[climdex_name], ylabel_dict[climdex_name], season, var, methodName,
                                          xlabel)

                            # Csv with data for evolution graphs
                            # if (season == season_dict['ANNUAL']) or (climdex_name in ('TXm', 'TNm', 'Pm', 'PRCPTOT')):
                            csv_evol(pathOut, subDir, nYears, ssp_dict, years, climdex_name, season)

                            # Spaghetti plot
                            # if climdex_name in ('TXm', 'TNm'):
                            spaghetti(pathOut, subDir, ssp_dict, years, ylim_dict[climdex_name], climdex_name,
                                                  ylabel_dict[climdex_name], season, var, methodName, regType, regName, xlabel)

                            # Mean and spread ensemble tube plot
                            # if (season == season_dict['ANNUAL']) or (climdex_name in ('TXm', 'TNm', 'Pm', 'PRCPTOT')):
                            tube(pathOut, subDir, ssp_dict, climdex_name, years, ylim_dict[climdex_name],
                                 ylabel_dict[climdex_name], season, var, methodName, regType, regName, xlabel)

                            # Change maps
                            if (regType == typeCompleteRegion) and (climdex_name in ('TXm', 'TNm', 'Pm', 'PRCPTOT')):
                                change_maps(ssp_dict, years, var, methodName, season, climdex_name, pathOut, scene_names_dict)


########################################################################################################################
def trend_raw(pathOut, subDir, ssp_dict, raw_ssp_dict, climdex_name, years, ylim, ylabel, season, var, methodName, xlabel):

    if methodName != 'RAW':
        if var == 'pcp':
            colors = p_methods_colors
            linestyles = p_methods_linestyles
            sign_ylabel = '%'
            title_size = 28
        else:
            colors = t_methods_colors
            linestyles = t_methods_linestyles
            sign_ylabel = degree_sign
            title_size = 20

        color = [colors[x['methodName']] for x in methods if x['var'] == VAR]
        linestyle = linestyles[methodName]

        # method
        data = np.nanmean(ssp_dict['data'], axis=2)
        if var == 'pcp':
            data = gaussian_filter1d(data, 2)
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        plt.plot(years, mean, label=methodName, color=color, linestyle=linestyle)
        plt.fill_between(years, mean - std, mean + std, color=color, alpha=0.8)

        # RAW
        data = np.nanmean(raw_ssp_dict['data'], axis=2)
        if var == 'pcp':
            data = gaussian_filter1d(data, 2)
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        plt.plot(years, mean, label='RAW', color='dimgray', linestyle=linestyles['RAW'])
        plt.fill_between(years, mean - std, mean + std, color=colors['RAW'], alpha=0.7)

        plt.ylim(ylim)
        plt.ylabel(sign_ylabel)

        # plt.xlabel(xlabel)
        plt.plot(years, np.zeros((len(years, ))), color='k', linewidth=0.2)
        # plt.show()
        # exit()
        filename = '_'.join(('PROJECTIONS', 'evolTrendRaw', 'climdex', var, climdex_name, methodName, season))
        if (plotAllRegions == False) and ((season == season_dict['ANNUAL']) or (climdex_name in ('TXm', 'TNm', 'Pm'))):
            plt.title(methodName, fontsize=title_size)
            plt.savefig(pathFigures + filename + '.png')
        elif plotAllRegions == True:
            plt.legend(loc='upper left')
            title = regName.upper() + '\n' + season
            plt.title(title)
            plt.savefig(pathOut + 'evolution/' + subDir + filename + '.png')
        plt.close()

########################################################################################################################
def csv_evol(pathOut, subDir, nYears, ssp_dict, years, climdex_name, season):
    """Write csv with data for evolution graphs"""

    if not os.path.exists(pathOut + 'csv/' + subDir):
        os.makedirs(pathOut + 'csv/' + subDir)
    data_allScenes = np.zeros((nYears, 0))
    models_allScenes = []
    for scene in collections.OrderedDict(sorted(ssp_dict.items(), reverse=True)).keys():
        models = [x + '_' + scene for x in ssp_dict[scene]['models']]
        for model in models:
            models_allScenes.append(model)
        data = ssp_dict[scene]['data'].mean(axis=2).T
        data_allScenes = np.append(data_allScenes, data, axis=1)
    models_allScenes.insert(0, 'Año')
    data_allScenes = np.insert(data_allScenes, 0, years, axis=1)
    np.savetxt(pathOut + 'csv/' + subDir + climdex_name + '_' + season + '.csv', data_allScenes,
               fmt=['%.i'] + ['%.1f'] * (len(models_allScenes) - 1), delimiter=';',
               header=';'.join(models_allScenes))

########################################################################################################################
def spaghetti(pathOut, subDir, ssp_dict, years, ylim, climdex_name, ylabel, season, var,
              methodName, regType, regName, xlabel):
    """Plot evolution graphs all models"""

    if not os.path.exists(pathOut + 'evolution/' + subDir):
        os.makedirs(pathOut + 'evolution/' + subDir)
    color_dict = {'ssp119': 'Greys', 'ssp126': 'Blues', 'ssp245': 'Greens', 'ssp370': 'Oranges', 'ssp585': 'Reds'}
    for scene in collections.OrderedDict(sorted(ssp_dict.items(), reverse=True)).keys():
        models = ssp_dict[scene]['models']
        nModels = len(models)
        all_data = ssp_dict[scene]['data']

        # Define colors
        evenly_spaced_interval = np.linspace(0.2, 0.8, int(nModels / 4) + 1)
        cm = plt.get_cmap(color_dict[scene])
        colors = [cm(x) for x in evenly_spaced_interval]
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--',
                      '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':',
                      '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--',
                      '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']

        for i in range(nModels):
            model = models[i]
            data = all_data[i].mean(axis=1)
            color = colors[int(i / 4)]
            linestyle = linestyles[i]
            plt.plot(years, data, label=model, color=color, linestyle=linestyle)

    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(years, np.zeros((len(years, ))), color='k', linewidth=0.2)
    plt.legend(loc='upper left', fontsize='xx-small', ncol=3)
    # plt.show()
    # exit()
    if (plotAllRegions == False) and ((season == season_dict['ANNUAL']) or (climdex_name in ('TXm', 'TNm', 'Pm'))):
        filename = '_'.join(('PROJECTIONS', 'evolSpaghetti', 'climdex', var, climdex_name, methodName, season))
        plt.savefig(pathFigures + filename + '.png')
    elif plotAllRegions == True:
        filename = '_'.join(('evolSpaghetti', climdex_name, season))
        title = regName.upper() + '\n' + season
        plt.title(title)
        plt.savefig(pathOut + 'evolution/' + subDir + filename + '.png')
    plt.close()

########################################################################################################################
def tube(pathOut, subDir, ssp_dict, climdex_name, years, ylim, ylabel, season, var,
         methodName, regType, regName, xlabel):
    """Plot evolution graphs mean and spread"""

    if not os.path.exists(pathOut + 'evolution/' + subDir):
        os.makedirs(pathOut + 'evolution/' + subDir)
    color_dict = {'ssp119': 'darkblue', 'ssp126': 'lightblue', 'ssp245': 'orange', 'ssp370': 'salmon', 'ssp585': 'darkred'}
    for scene in collections.OrderedDict(sorted(ssp_dict.items(), reverse=True)).keys():
        models = ssp_dict[scene]['models']
        nModels = len(models)
        data = ssp_dict[scene]['data'].mean(axis=2)
        if climdex_name in ('PRCPTOT', 'Pm'):  # At the moment only Pm and PRCPTOT are smoothed, but this is optional
            data = gaussian_filter1d(data, 2)
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        scene_legend = scene_names_dict[scene]
        plt.plot(years, mean, label=scene_legend + '   (' + str(nModels) + ')', color=color_dict[scene])
        plt.fill_between(years, mean - std, mean + std, color=color_dict[scene], alpha=0.3)

    plt.legend(loc='upper left')
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(years, np.zeros((len(years, ))), color='k', linewidth=0.2)
    # plt.show()
    # exit()
    if (plotAllRegions == False) and ((season == season_dict['ANNUAL']) or (climdex_name in ('TXm', 'TNm', 'Pm'))):
        filename = '_'.join(('PROJECTIONS', 'evolTube', 'climdex', var, climdex_name, methodName, season))
        plt.savefig(pathFigures + filename + '.png')
    elif plotAllRegions == True:
        filename = '_'.join(('evolTube', climdex_name, season))
        title = regName.upper() + '\n' + season
        plt.title(title)
        plt.savefig(pathOut + 'evolution/' + subDir + filename + '.png')
    plt.close()

########################################################################################################################
def change_maps(ssp_dict, years, var, methodName, season, climdex_name, pathOut, scene_names_dict):
    """Plot change maps"""

    # Go through all scenes
    for scene in ssp_dict.keys():

        # Go through all periods
        for term_years in (shortTerm_years, longTerm_years):
            # if ((plotAllRegions == True) or ((scene == 'ssp585') and (term_years == longTerm_years))):
            period = str(term_years[0]) + '-' + str(term_years[1])
            years_list = [x for x in range(term_years[0], term_years[1] + 1)]
            iYears = [i for i in range(len(years)) if years[i] in years_list]
            dataTerm = np.mean(ssp_dict[scene]['data'][:, iYears, :], axis=1)

            # Delete nan
            iNans = np.unique(np.where(np.isnan(dataTerm))[0])
            dataTerm = np.delete(dataTerm, iNans, axis=0)

            # Calculatean and plot mean and std
            mean = np.mean(dataTerm, axis=0)
            if plotAllRegions == False:
                filename = '_'.join(
                    ('PROJECTIONS', 'meanChangeMap', 'climdex', var, climdex_name, methodName, season))
                plot.map(mean, 'change_' + climdex_name + '_mean', path=pathFigures,
                         filename=filename, title='')
            else:
                filename = '_'.join(('meanChangeMap', climdex_name, scene, season, period))
                # title = ' '.join(('mod_mean', climdex_name, scene, season, period))
                title = scene_names_dict[scene]+'   '+period+'\n'+season
                plot.map(mean, 'change_' + climdex_name + '_mean', path=pathOut + 'maps/',
                         filename=filename, title=title)
            std = np.std(dataTerm, axis=0)
            if plotAllRegions == False:
                filename = '_'.join(
                    ('PROJECTIONS', 'stdChangeMap', 'climdex', var, climdex_name, methodName, season))
                plot.map(mean, 'change_' + climdex_name + '_mean', path=pathFigures,
                         filename=filename, title='')
                plot.map(std, 'change_' + climdex_name + '_std', path=pathFigures,
                         filename=filename, title='')
            else:
                filename = '_'.join(('stdChangeMap', climdex_name, scene, season, period))
                # title = ' '.join(('mod_std', climdex_name, scene, season, period))
                title = scene_names_dict[scene]+'   '+period+'\n'+season
                plot.map(std, 'change_' + climdex_name + '_std', path=pathOut + 'maps/',
                         filename=filename, title=title)


########################################################################################################################
def format_web_AEMET():
    """
    This function prepare all files needed for web_AEMET: dailyData, maps, evol_graphs, and csv.
    It needs to be adapted. Before runing, make sure to define filenames, etc. properly.
    """

    # Define CMIP number
    cmipNumber = '5'

    # Go through all methods
    for method_dict in methods:
        var, methodName = method_dict['var'], method_dict['methodName']

        # Define fields for REJILLA vs ESTACIONES
        df_targetType = pd.DataFrame(columns=['targetType', 'nameForDir', 'nameForDailyData', 'nameForFigAndCsv'])
        df_targetType = df_targetType.append({'targetType': 'gridded_data', 'nameForDir': 'REJILLA', 'nameForDailyData': '_REJ', 'nameForFigAndCsv': 'R_'}, ignore_index=True)
        df_targetType = df_targetType.append({'targetType': 'stations', 'nameForDir': 'ESTACIONES', 'nameForDailyData': '', 'nameForFigAndCsv': ''}, ignore_index=True)

        # Define fields for each var
        df_vars = pd.DataFrame(columns=['var', 'nameForMaps', 'nameForDailyData', 'nameForFigAndCsv'])
        df_vars = df_vars.append({'var': 'tmax', 'nameForMaps': 'txm', 'nameForDailyData': 'tmax', 'nameForFigAndCsv': 'Tx'}, ignore_index=True)
        df_vars = df_vars.append({'var': 'tmin', 'nameForMaps': 'tim', 'nameForDailyData': 'tmin', 'nameForFigAndCsv': 'Tm'}, ignore_index=True)
        df_vars = df_vars.append({'var': 'pcp', 'nameForMaps': 'prm', 'nameForDailyData': 'precip', 'nameForFigAndCsv': 'P'}, ignore_index=True)

        # Define fields for each climdex
        df_climdex = pd.DataFrame(columns=['var', 'climdex', 'nameForFigAndCsv'])
        df_climdex = df_climdex.append({'var': 'tmax', 'climdex': 'TXm', 'nameForFigAndCsv': 'Tx'}, ignore_index=True)
        df_climdex = df_climdex.append({'var': 'tmax', 'climdex': 'TX90p', 'nameForFigAndCsv': 'WD'}, ignore_index=True)
        df_climdex = df_climdex.append({'var': 'tmax', 'climdex': 'WSDI', 'nameForFigAndCsv': 'LOC'}, ignore_index=True)
        df_climdex = df_climdex.append({'var': 'tmin', 'climdex': 'TNm', 'nameForFigAndCsv': 'Tm'}, ignore_index=True)
        df_climdex = df_climdex.append({'var': 'tmin', 'climdex': 'TN90p', 'nameForFigAndCsv': 'WN'}, ignore_index=True)
        df_climdex = df_climdex.append({'var': 'tmin', 'climdex': 'FD', 'nameForFigAndCsv': 'FD'}, ignore_index=True)
        df_climdex = df_climdex.append({'var': 'pcp', 'climdex': 'Pm', 'nameForFigAndCsv': 'P'}, ignore_index=True)
        df_climdex = df_climdex.append({'var': 'pcp', 'climdex': 'R01', 'nameForFigAndCsv': 'R1'}, ignore_index=True)
        df_climdex = df_climdex.append({'var': 'pcp', 'climdex': 'CDD', 'nameForFigAndCsv': 'LPS'}, ignore_index=True)
        df_climdex = df_climdex.append({'var': 'pcp', 'climdex': 'R95pFRAC', 'nameForFigAndCsv': 'CP95'}, ignore_index=True)

        # Define fields for each method
        # ANÁLOGOS
        df_methods = pd.DataFrame(columns=['var', 'methodName', 'nameForDir', 'nameForDailyData', 'nameForFigAndCsv', 'nameForMaps'])
        df_methods = df_methods.append({'var': 'tmax', 'methodName': 'ANA', 'nameForDir': 'ANALOGOS', 'nameForDailyData': 'ANALOGOS', 'nameForFigAndCsv': 'A', 'nameForMaps': 'A'}, ignore_index=True)
        df_methods = df_methods.append({'var': 'tmin', 'methodName': 'ANA', 'nameForDir': 'ANALOGOS', 'nameForDailyData': 'ANALOGOS', 'nameForFigAndCsv': 'A', 'nameForMaps': 'A'}, ignore_index=True)
        df_methods = df_methods.append({'var': 'pcp', 'methodName': 'ANA-LOC-N', 'nameForDir': 'ANALOGOS', 'nameForDailyData': 'ANALOGOS', 'nameForFigAndCsv': 'A', 'nameForMaps': 'A'}, ignore_index=True)
        # REGRESIÓN
        df_methods = df_methods.append({'var': 'tmax', 'methodName': 'MLR', 'nameForDir': 'REGRESION', 'nameForDailyData': 'SDSM', 'nameForFigAndCsv': 'R', 'nameForMaps': 'R'}, ignore_index=True)
        df_methods = df_methods.append({'var': 'tmin', 'methodName': 'MLR', 'nameForDir': 'REGRESION', 'nameForDailyData': 'SDSM', 'nameForFigAndCsv': 'R', 'nameForMaps': 'R'}, ignore_index=True)
        df_methods = df_methods.append({'var': 'pcp', 'methodName': 'GLM-EXP', 'nameForDir': 'REGRESION', 'nameForDailyData': 'SDSM', 'nameForFigAndCsv': 'R', 'nameForMaps': 'R'}, ignore_index=True)
        # REDES NEURONALES
        df_methods = df_methods.append({'var': 'tmax', 'methodName': 'ANN', 'nameForDir': 'RRNN', 'nameForDailyData': 'RED_NEURONAL', 'nameForFigAndCsv': 'N', 'nameForMaps': 'N'}, ignore_index=True)
        df_methods = df_methods.append({'var': 'tmin', 'methodName': 'ANN', 'nameForDir': 'RRNN', 'nameForDailyData': 'RED_NEURONAL', 'nameForFigAndCsv': 'N', 'nameForMaps': 'N'}, ignore_index=True)
        df_methods = df_methods.append({'var': 'pcp', 'methodName': 'ANN', 'nameForDir': 'RRNN', 'nameForDailyData': 'RED_NEURONAL', 'nameForFigAndCsv': 'N', 'nameForMaps': 'N'}, ignore_index=True)

        # Define fields for seasons
        df_seasons = pd.DataFrame(columns=['season', 'nameForMaps', 'nameForFigAndCsv'])
        df_seasons = df_seasons.append({'season': 'ANNUAL', 'nameForMaps': 'anu', 'nameForFigAndCsv': 'Anual'}, ignore_index=True)
        df_seasons = df_seasons.append({'season': 'DJF', 'nameForMaps': 'inv', 'nameForFigAndCsv': 'Invierno'}, ignore_index=True)
        df_seasons = df_seasons.append({'season': 'MAM', 'nameForMaps': 'pri', 'nameForFigAndCsv': 'Primavera'}, ignore_index=True)
        df_seasons = df_seasons.append({'season': 'JJA', 'nameForMaps': 'ver', 'nameForFigAndCsv': 'Verano'}, ignore_index=True)
        df_seasons = df_seasons.append({'season': 'SON', 'nameForMaps': 'oto', 'nameForFigAndCsv': 'Otono'}, ignore_index=True)

        # Define fields for regions
        regions_file = '../config/regions_web_'+nameCompleteRegion+'.csv'
        try:
            df_regions = pd.read_csv(regions_file)
        except:
            print('Copy ../aux/ASSOCIATION/regions.csv to ' + regions_file + ' and fill nameForMaps and nameForFigAndCsv manually')
            print('Remove column of points and add columns nameForMaps and nameForFigAndCsv')
            exit()

        # Select case from df_targetType, df_methods and df_var
        df_targetType = df_targetType[df_targetType['targetType'] == target_type]
        df_methods = df_methods[(df_methods['var'] == var) & (df_methods['methodName'] == methodName)]
        df_vars = df_vars[df_vars['var'] == var]

        # Define and create paths
        pathBase = '../results/web_AEMET/' + df_methods['nameForDir'].values[0] + '_' + df_targetType['nameForDir'].values[0] + '/'
        if not os.path.exists(pathBase):
            os.makedirs(pathBase)
        if not os.path.exists(pathBase + 'DATOS_DIARIOS/'):
            os.makedirs(pathBase + 'DATOS_DIARIOS/')
        if not os.path.exists(pathBase + 'MAPAS_PROYECCIONES/IMAGENES/'):
            os.makedirs(pathBase + 'MAPAS_PROYECCIONES/IMAGENES/')
        for regType in (typeCompleteRegion, 'CCAA', 'PROV', 'CCHH'):
            if not os.path.exists(pathBase + 'GRAFICOS_EVOLUCION/CSV/'+regType+'/'):
                os.makedirs(pathBase + 'GRAFICOS_EVOLUCION/CSV/'+regType+'/')
            if not os.path.exists(pathBase + 'GRAFICOS_EVOLUCION/JPEG/'+regType+'/'):
                os.makedirs(pathBase + 'GRAFICOS_EVOLUCION/JPEG/'+regType+'/')


        # Change maps
        format_web_AEMET_maps(var, methodName, df_methods, df_targetType, df_seasons, df_regions, df_vars, cmipNumber)

        # Evolution (png and csv)
        format_web_AEMET_evolution(var, methodName, df_methods, df_targetType, df_seasons, df_regions, df_vars, cmipNumber)

        # Daily data
        format_web_AEMET_dailyData(var, methodName, df_methods, df_targetType, df_vars, to_nc=False)



########################################################################################################################
def format_web_AEMET_maps(var, methodName, df_methods, df_targetType, df_seasons, df_regions, df_vars,
                          cmipNumber):
    """Copy change maps with dirs/filenames for the website"""

    # Define paths
    pathIn = '../results/PROJECTIONS/' + var.upper() + '/' + methodName + '/' + '/'.join(('climdex', 'figures', 'maps')) + '/'
    pathOut = '../results/web_AEMET/' + df_methods['nameForDir'].values[0] + '_' + df_targetType['nameForDir'].values[0] + '/MAPAS_PROYECCIONES/IMAGENES/'

    # Select climdex
    for climdex in climdex_names[var]:
        if climdex in ('TXm', 'TNm', 'Pm'):

            # Select season
            for index, row_seasons in df_seasons.iterrows():

                # Select scene
                for scene in scene_list:
                    if scene != 'historical':

                        # Select period
                        for term_years in (shortTerm_years, longTerm_years):
                            period = str(term_years[0]) + '-' + str(term_years[1])

                            # Select stat (mean and std)
                            for stat in ('mean', 'std'):
                                fileOld = '_'. join((stat+'ChangeMap', climdex, scene, season_dict[row_seasons['season']], period)) + '.png'
                                fileNew = cmipNumber + '_EST_' + df_methods['nameForMaps'].values[0] + '_' + \
                                          df_regions[df_regions['regType'] == typeCompleteRegion]['nameForMaps'].values[0] + \
                                          '_' + stat[0] + '-d' + df_vars['nameForMaps'].values[0] + \
                                          row_seasons['nameForMaps'] + '.' + \
                                            '.'.join((scene.lower(), period, '1', 'png'))

                                # Copy file
                                pathOld, pathNew = pathIn, pathOut
                                shutil.copyfile(pathOld+fileOld, pathNew+fileNew)

########################################################################################################################
def format_web_AEMET_evolution(var, methodName, df_methods, df_targetType, df_seasons, df_regions, df_vars,
                          cmipNumber):
    """Copy evolution with dirs/filenames for the website"""

    # Define paths
    pathIn = '../results/PROJECTIONS/' + var.upper() + '/' + methodName + '/' + '/'.join(('climdex', 'figures')) + '/'
    pathOut = '../results/web_AEMET/' + df_methods['nameForDir'].values[0] + '_' + df_targetType['nameForDir'].values[0] + '/GRAFICOS_EVOLUCION/'

    # Select  region
    for index, row_regions in df_regions.iterrows():
        print(var, methodName, index, '/', df_regions.shape[0])

        # Select climdex
        for climdex in climdex_names[var]:

            # Select season
            for index, row_seasons in df_seasons.iterrows():

                # Select scene
                for scene in scene_list:
                    if scene != 'historical':

                        # Spaghetti
                        if climdex in ('TXm', 'TNm'):
                            pathOld = pathIn+'evolution/'+row_regions['subDir']
                            pathNew = pathOut+'JPEG/'+row_regions['regType']+'/'
                            fileOld = '_'. join(('evolSpaghetti', climdex, season_dict[row_seasons['season']])) + '.png'
                            fileNew = '_'.join((cmipNumber, 'EST', df_methods['nameForFigAndCsv'].values[0],
                                                'Modelos_Anomal', df_vars['nameForFigAndCsv'].values[0],
                                                row_regions['nameForFigAndCsv'], row_seasons['nameForFigAndCsv'])) + '.png'
                            shutil.copyfile(pathOld+fileOld, pathNew+fileNew)

                        # Tubes
                        if (climdex in ('TXm', 'TNm', 'Pm')) or (row_seasons['season'] == 'ANNUAL'):
                            pathOld = pathIn+'evolution/'+row_regions['subDir']
                            pathNew = pathOut+'JPEG/'+row_regions['regType']+'/'
                            fileOld = '_'. join(('evolTube', climdex, season_dict[row_seasons['season']])) + '.png'
                            fileNew = '_'.join((cmipNumber, 'EST', df_methods['nameForFigAndCsv'].values[0],
                                                'MedModelos_Anomal', df_vars['nameForFigAndCsv'].values[0],
                                                row_regions['nameForFigAndCsv'], row_seasons['nameForFigAndCsv'])) + '.png'
                            shutil.copyfile(pathOld+fileOld, pathNew+fileNew)

                        # Csv (they are duplicated to go both with spaghettis and tubes)
                        if (climdex in ('TXm', 'TNm', 'Pm')) or (row_seasons['season'] == 'ANNUAL'):
                            pathOld = pathIn+'csv/'+row_regions['subDir']
                            pathNew = pathOut+'CSV/'+row_regions['regType']+'/'
                            fileOld = '_'. join((climdex, season_dict[row_seasons['season']])) + '.csv'
                            fileNew1 = '_'.join((cmipNumber, 'EST', df_methods['nameForFigAndCsv'].values[0],
                                                'Modelos_Anomal', df_vars['nameForFigAndCsv'].values[0],
                                                row_regions['nameForFigAndCsv'], row_seasons['nameForFigAndCsv'])) + '.csv'
                            fileNew2 = '_'.join((cmipNumber, 'EST', df_methods['nameForFigAndCsv'].values[0],
                                                'MedModelos_Anomal', df_vars['nameForFigAndCsv'].values[0],
                                                 row_regions['nameForFigAndCsv'],
                                                 row_seasons['nameForFigAndCsv'])) + '.csv'
                            shutil.copyfile(pathOld+fileOld, pathNew+fileNew1)
                            shutil.copyfile(pathOld + fileOld, pathNew + fileNew2)


########################################################################################################################
def format_web_AEMET_dailyData(var, methodName, df_methods, df_targetType, df_vars, to_nc=False):
    """Copy dailyData with dirs/filenames for the website"""

    # Define and create paths
    pathIn = '../results/PROJECTIONS/' + var.upper() + '/' + methodName + '/daily_data/'
    pathOut = '../results/web_AEMET/' + df_methods['nameForDir'].values[0] + '_' + df_targetType['nameForDir'].values[0] + '/DATOS_DIARIOS/'
    if not os.path.exists(pathOut+'ASCII/'):
        os.makedirs(pathOut+'ASCII/')
    if to_nc == True:
        if not os.path.exists(pathOut+'NETCDF/'):
            os.makedirs(pathOut+'NETCDF/')

    if var == 'pcp':
        units = 'décimas de mm'
    else:
        units = degree_sign

    # Go through all models and scenes
    for model in model_list:
        for scene in scene_list:
            if scene == 'historical':
                period = str(historical_years[0]) + '-' + str(historical_years[1])
            else:
                period = str(rcp_years[0]) + '-' + str(rcp_years[1])

            # Check if scene/model exists
            if os.path.isfile(pathIn + model + '_' + scene + '.nc'):
                # Read scene data
                nc = read.netCDF(pathIn, model + '_' + scene + '.nc', var)
                times = nc['times']
                data = nc['data']
                del nc

                # Same filename for netCDF and ASCII
                fileOut_noExt = '.'.join((df_vars['nameForDailyData'].values[0], model, scene, period,
                                df_methods['nameForDailyData'].values[0] + df_targetType['nameForDailyData'].values[0]))

                # netCDF: For gridded data netCDFs are transformed to rotated grid
                if to_nc == True:
                    print('Rotated netCDFs take too much space in disk.')
                    print('Rotated netCDFs break because of memory.')
                    print('Rotated netCDFs need a header.')
                    exit()
                    print('writing netCDF', model, scene)
                    if df_targetType['targetType'].values[0] == 'gridded_data':
                        write.netCDF_rotated(pathOut+'NETCDF/', fileOut_noExt+'.nc', var, data, times)
                    elif df_targetType['targetType'].values[0] == 'stations':
                        shutil.copyfile(pathIn+model+'_'+scene+'.nc', pathOut+'NETCDF/'+fileOut_noExt+'.nc')

                # ASCII: id_stations is added as header and dates is added as first column
                print('writing ASCII', model, scene)
                id = list(read.hres_metadata().index.values)
                times = np.array([10000 * x.year + 100 * x.month + x.day for x in times])
                data = np.append(times[:, np.newaxis], data, axis=1)
                id.insert(0, 'YYYYMMDD')
                id = [str(x) for x in id]
                header = ' '.join((model, scene, period, df_vars['nameForDailyData'].values[0], '(', units, ')',
                                   df_methods['nameForDailyData'].values[0])) + '\n' + ';'.join(id)
                np.savetxt(pathOut+'ASCII/'+fileOut_noExt+'.csv', data, fmt=['%.i'] + ['%.2f'] * (len(id) - 1), delimiter=';', header=header)


