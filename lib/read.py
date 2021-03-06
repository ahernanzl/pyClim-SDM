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
def netCDF(dataPath, filename, nc_variable, grid=None, level=None):
    """
    This function reads netCDF files.
    Data with _FillValue are set to Nan.
    Return dictionary with data, times, lats, lons and calendar.
    Because some models have calendars not supported by datetime, non supported dates are removed. Otherwise dates
    should be managed as strings yyyymmdd in the whole project.
    """

    start = datetime.datetime.now()
    nc = Dataset(dataPath + filename)

    # # print(nc)
    # for var in nc.variables:
    # 	print('----------------------------')
    # 	print(var)
    # 	print(nc.variables[var])
    # 	# print(nc.variables[var][:])
    # 	print(nc.variables[var][:].shape)
    # exit()


    # Define dimension names
    time_name = 'time'
    if 'lat' in nc.variables:
        lat_name, lon_name = 'lat', 'lon'
    elif 'latitude' in nc.variables:
        lat_name, lon_name = 'latitude', 'longitude'
    else:
        print('lat_name not recognized')
        exit()
    if level != None:
        if 'level' in nc.variables:
            level_name = 'level'
        elif 'plev' in nc.variables:
            level_name = 'plev'
        else:
            print('level_name not recognized')
            exit()

    calendar = nc.variables[time_name].calendar
    units = nc.variables[time_name].units
    times_num = nc.variables[time_name][:].data
    times = num2date(times_num, units=units, calendar=calendar)

    # bounds = num2date(nc.variables['time_bnds'][:], units=units, calendar=calendar)
    # print(bounds[0])
    # print(times[0])

    lats = nc.variables[lat_name][:]
    lons = nc.variables[lon_name][:]
    lons[lons > 180] -= 360
    nlats, nlons = len(lats), len(lons)

    if level == None:
        data = nc.variables[nc_variable][:]
    else:
        if level_name == 'plev':
            level_factor = 100
        else:
            level_factor = 1
        ilevel = np.where(abs((nc.variables[level_name][:] - int(level*level_factor))) < 0.001)[0]
        dim=nc.variables[nc_variable][:].shape
        data = nc.variables[nc_variable][:, ilevel, :, :].reshape(dim[0], dim[2], dim[3])


    # Set invalid data to Nan
    if hasattr(nc.variables[nc_variable], '_FillValue'):
        data[data==nc.variables[nc_variable]._FillValue] = np.nan
    if np.ma.is_masked(data):
        data = np.ma.MaskedArray.filled(data, fill_value=np.nan)


    if calendar == '360_day':
        valid_idates = []
        for i in range(times.size):
            if not ((times[i].month == 2) and (times[i].day >= 29)):
                valid_idates.append(i)
                year = times[i].year
                month = times[i].month
                day = times[i].day
                times[i] = datetime.datetime(year, month, day, 12, 0)
        data = data[valid_idates]
        times = times[valid_idates]
    else:
        for i in range(times.size):
            year, month, day, hour, minute = times[i].year, times[i].month, times[i].day, 12, 0
            times[i] = datetime.datetime(year, month, day, hour, minute)

    data = data.__array__()
    # data = np.frombuffer(data.getdata).reshape(data.shape)

    # Selects a specific gr defined at settings
    if grid != None:
        if grid == 'ext':
            grid_lats, grid_lons = ext_lats, ext_lons
        elif grid == 'saf':
            grid_lats, grid_lons = saf_lats, saf_lons
        elif grid == 'pred':
            grid_lats, grid_lons = pred_lats, pred_lons
        ilats = [i for i in range(nlats) if lats[i] in grid_lats]
        ilons = [i for i in range(nlons) if lons[i] in grid_lons]
        lats, lons = lats[ilats], lons[ilons]
        data = data[:, ilats]
        data = data[:, :, ilons]


    # Get units
    try:
        units = nc.variables[nc_variable].units
    except:
        units = 'Unknown'

    end = datetime.datetime.now()
    if ((end-start).seconds > 30):
        print('......................................................................')
        print(nc_variable, level, end-start, 'access to var_time very slow')

    return {'data': data, 'times': times, 'lats': lats, 'lons': lons,  'calendar': calendar, 'units': units}


########################################################################################################################
def one_direct_predictor(predName, level=None, grid=None, model='reanalysis', scene=None):
    """
    Reads one direct predictor from reanalysis or model. The purpose of this function is to avoid defining path, filename,
    etc., each time a predictor is read, so they can be read easily no matter whether we are using reanalysis or models.
    tmax, tmin and pcp are explicitly defined too because they are used (for RAW, MOS, WG...) even if they are not in
    pred_list for TF.
    Check units and forces tmax and tmin to Celcious degrees, pcp to mm/day, tcc to %, and z to m
    When using different reanalysis some modifications might be needed (for pcp)
    :param predName: as defined in preds dictionaries in settings
    :return: dictionary with data, times, lats, lons and calendar
    """

    if level != None:
        predName += str(level)

    if model == 'reanalysis':
        pathIn = '../input_data/reanalysis/'
        if predName in ('tmax', 'tmin', 'pcp'):
            ncVar = reaNames[predName]
        else:
            for aux_level in all_levels:
                predName = predName.replace(str(aux_level), '')
            ncVar = reaNames[predName]
        filename = ncVar+'_'+reanalysisName+'_'+reanalysisPeriodFilename+'.nc'
    else:
        if scene in ('historical', 'HISTORICAL'):
            periodFilename = historicalPeriodFilename
        else:
            periodFilename = sspPeriodFilename
        pathIn = '../input_data/models/'
        if predName in ('tmax', 'tmin', 'pcp'):
            ncVar = modNames[predName]
        else:
            for aux_level in all_levels:
                predName = predName.replace(str(aux_level), '')
            ncVar = modNames[predName]
        modelName, modelRun = model.split('_')[0], model.split('_')[1]
        filename = ncVar+'_'+modelName+'_'+scene+'_'+modelRun+'_'+periodFilename+'.nc'

    nc = netCDF(pathIn, filename, ncVar, grid=grid, level=level)

    # Force units
    # print(predName, nc['units'], print(np.nanmean(nc['data'])))
    if predName in ('tmax', 'tmin'):
        if nc['units'] == 'K':
            nc.update({'data': nc['data']-273.15})
            nc.update({'units': degree_sign})
        elif (nc['units'] == 'Unknown') and (np.nanmean(nc['data']) > 100):
            nc.update({'data': nc['data']-273.15})
            nc.update({'units': degree_sign})
    elif predName == 'pcp':
        if nc['units'] == 'kg m-2 s-1':
            nc.update({'data': 24 * 60 * 60 * nc['data']}) # from mm/s to mm/day
            nc.update({'units': 'mm'})
        elif (nc['units'] == 'Unknown') and (model == 'reanalysis'):
            nc.update({'data': 1000 * nc['data']}) # from m/day to mm/day
            nc.update({'units': 'mm'})
    elif predName == 'tcc':
        if nc['units'] != '%' and (int(np.nanmax(nc['data'])) <= 1):
            nc.update({'data': 100 * nc['data']})
            nc.update({'units': '%'})
    elif predName == 'z':
        if nc['units'] != 'm':
            nc.update({'data': nc['data'] / 9.8}) # from m2/s2 to m
            nc.update({'units': 'm'})
    # print(predName, nc['units'], print(np.nanmean(nc['data'])))
    # exit()
    return nc





########################################################################################################################
def lres_data(var, field, grid=None, model='reanalysis', scene=None, predName=None, period=None):
    """
    var: tmax, tmin or pcp, in order to use one or another predictor list
    If no grid is specified, each field will use its own grid. But a grid can be specified so, for example, this
    function can read preds in a saf_grid, or any other combination.
    Possible fields: var, pred, saf
    Possible grids: pred, saf
    If predName is passed as argument, only that predictor will be read.
    Different periods can be extracted. If period==None, the whole calibration period is returned for reanalysis and for
    GCMs the whole scene (hisotical/SSP). For reanalysis and historical GCMs also the reference period can be extracted
    by specifying period='reference'.
    Beware that different GCMs can have different calendars
    return: data (ndays, npreds, nlats, nlons) and times
    """

    # Define variables
    var0 = var[0]
    if field == 'var':
        nvar = 1
    elif field == 'saf':
        nvar = nsaf
        preds = saf_dict
    elif field == 'pred':
        if var0 == 'p':
            nvar = n_preds_p
            preds = preds_dict['p']
        if var0 == 't':
            nvar = n_preds_t
            preds = preds_dict['t']
        if predName != None:
            nvar = 1
            try:
                preds = {predName: preds[predName]}
            except:
                print('Activate', predName, 'as predictor')
                exit()
    else:
        print('field', field, 'not valid.')
        exit()

    # Define var_aux for dates
    if var0 == 'p':
        var_aux = 'pcp'
    else:
        var_aux = 'tmax'

    # Define dates
    if model == 'reanalysis':
        dates = calibration_dates
    else:
        dates = np.ndarray.tolist(read.one_direct_predictor(var_aux, grid='ext', model=model, scene=scene)['times'])
    ndates = len(dates)

    data = np.zeros((nvar, ndates, ext_nlats, ext_nlons))

    # Read all data in ext_grid
    if model == 'reanalysis':
        # Calibration dates are extracted from files
        aux_times = one_direct_predictor(var_aux, grid='ext', model=model, scene=scene)['times']
        idates = [i for i in range(len(aux_times)) if aux_times[i] in dates]

        # var
        if field == 'var':
            if var[0] == 't':
                data[0] = one_direct_predictor(var, level=None, grid='ext', model=model, scene=scene)['data'][idates]
            else:
                data[0] = one_direct_predictor(var, level=None, grid='ext', model=model, scene=scene)['data'][idates]


        # pred / saf
        elif field in ('pred', 'saf'):
            i = 0

            # tmax
            if 'tmax' in preds:
                data[i] = one_direct_predictor('tmax', level=None, grid='ext', model=model, scene=scene)['data'][idates]; i += 1
            # tmin
            if 'tmin' in preds:
                data[i] = one_direct_predictor('tmin', level=None, grid='ext', model=model, scene=scene)['data'][idates]; i += 1
            # pcp
            if 'pcp' in preds:
                data[i] = one_direct_predictor('pcp', level=None, grid='ext', model=model, scene=scene)['data'][idates]; i += 1
            # mslp
            if 'mslp' in preds:
                data[i] = one_direct_predictor('mslp', level=None, grid='ext', model=model, scene=scene)['data'][idates]; i += 1
            # mslp_trend
            if 'mslp_trend' in preds:
                data[i] = derived_predictors.mslp_trend(model=model, scene=scene)['data'][idates]; i += 1
            # ins
            if 'ins' in preds:
                data[i] = derived_predictors.insolation(model=model, scene=scene)['data'][idates]; i += 1
            # u10, v10
            for var in ('u10', 'v10'):
                if var in preds:
                    data[i] = one_direct_predictor(var, level=None, grid='ext', model=model, scene=scene)['data'][idates]; i += 1
            # t2m
            if 't2m' in preds:
                data[i] = one_direct_predictor('t2m', level=None, grid='ext', model=model, scene=scene)['data'][idates]; i += 1
            # d2m
            if 'd2m' in preds:
                data[i] = derived_predictors.dew_point('sfc')['data'][idates]; i += 1
            # q2m
            if 'q2m' in preds:
                data[i] = derived_predictors.specific_humidity('sfc')['data'][idates]; i += 1
            # r2m
            if 'r2m' in preds:
                data[i] = derived_predictors.relative_humidity('sfc')['data'][idates]; i += 1

            # u, v, t, z (direct predictors)
            for var in ['u', 'v', 't', 'z']:
                for level in preds_levels:
                    if var + str(level) in preds:
                        data[i] = one_direct_predictor(var, level=level, grid='ext', model=model, scene=scene)['data'][idates]; i += 1
            # q
            var = 'q'
            for level in preds_levels:
                if var + str(level) in preds:
                    data[i] = derived_predictors.specific_humidity(level)['data'][idates]; i += 1

            # r
            var = 'r'
            for level in preds_levels:
                if var + str(level) in preds:
                    data[i] = derived_predictors.relative_humidity(level)['data'][idates]; i += 1

            # td
            var = 'td'
            for level in preds_levels:
                if var + str(level) in preds:
                    data[i] = derived_predictors.dew_point(level)['data'][idates]; i += 1

            # Dtd
            var = 'Dtd'
            for level in preds_levels:
                if var + str(level) in preds:
                    t = one_direct_predictor('t', level=level, grid='ext', model=model, scene=scene)['data'][idates]
                    td = derived_predictors.dew_point(level)['data'][idates]
                    data[i] = t - td; i += 1

            # vort, div
            for var in ['vort', 'div']:
                for level in preds_levels:
                    if var + str(level) in preds:
                        data[i] = derived_predictors.vorticity_and_divergence(model=model, scene=scene, level=level)['data'][var][idates]; i += 1
            # thermal vertical gradients (tvg)
            for (level0, level1) in [(1000, 850), (850, 700), (700, 500)]:
                var = 'vtg_' + str(level0) + '_' + str(level1)
                if var in preds:
                    data[i] = derived_predictors.vtg(level0, level1, model=model, scene=scene)['data'][idates]; i += 1
            # ugsl, vgsl
            for var in ('u', 'v'):
                if var+'gsl' in preds:
                    data[i] = derived_predictors.geostrophic(model=model, scene=scene)['data'][var+'gsl'][idates]; i += 1
            # vortgsl, divgsl
            for var in ('vort', 'div'):
                if var+'gsl' in preds:
                    data[i] = derived_predictors.vorticity_and_divergence(model=model, scene=scene, level='sl')['data'][var][idates]; i += 1
            # Instability indexes
            if 'K_index' in preds:
                data[i] = derived_predictors.K_index(model=model, scene=scene)['data'][idates]; i += 1
            if 'TT_index' in preds:
                data[i] = derived_predictors.TT_index(model=model, scene=scene)['data'][idates]; i += 1
            if 'SSI_index' in preds:
                data[i] = derived_predictors.SSI_index(model=model, scene=scene)['data'][idates]; i += 1
            if 'LI_index' in preds:
                data[i] = derived_predictors.LI_index(model=model, scene=scene)['data'][idates]; i += 1

    else:

        # var
        if field == 'var':
            if var[0] == 't':
                data[0] = one_direct_predictor(var, level=None, grid='ext', model=model, scene=scene)['data']
            else:
                data[0] = one_direct_predictor(var, level=None, grid='ext', model=model, scene=scene)['data']

        # pred / saf
        elif field in ('pred', 'saf'):
            i = 0
            # tmax
            if 'tmax' in preds:
                data[i] = one_direct_predictor('tmax', level=None, grid='ext', model=model, scene=scene)['data']; i += 1
            # tmin
            if 'tmin' in preds:
                data[i] = one_direct_predictor('tmin', level=None, grid='ext', model=model, scene=scene)['data']; i += 1
            # pcp
            if 'pcp' in preds:
                data[i] = one_direct_predictor('pcp', level=None, grid='ext', model=model, scene=scene)['data']; i += 1
            # mslp
            if 'mslp' in preds:
                data[i] = one_direct_predictor('mslp', level=None, grid='ext', model=model, scene=scene)['data']; i += 1
            # mslp_trend
            if 'mslp_trend' in preds:
                data[i] = derived_predictors.mslp_trend(model=model,scene=scene)['data']; i += 1
            # ins
            if 'ins' in preds:
                data[i] = derived_predictors.insolation(model=model,scene=scene)['data']; i += 1
            # u10, v10
            for var in ('u10', 'v10'):
                if var in preds:
                    data[i] = one_direct_predictor(var, level=None, grid='ext', model=model, scene=scene)['data']; i += 1
            # t2m
            if 't2m' in preds:
                data[i] = one_direct_predictor('t2m', level=None, grid='ext', model=model, scene=scene)['data']; i += 1
            # d2m
            if 'd2m' in preds:
                data[i] = derived_predictors.dew_point('sfc', model=model, scene=scene)['data']; i += 1
            # q2m
            if 'q2m' in preds:
                data[i] = derived_predictors.specific_humidity('sfc', model=model, scene=scene)['data']; i += 1
            # r2m
            if 'r2m' in preds:
                data[i] = derived_predictors.relative_humidity('sfc', model=model, scene=scene)['data']; i += 1

            # u, v, t (direct predictors)
            for var in ['u', 'v', 't']:
                for level in preds_levels:
                    if var + str(level) in preds:
                        data[i] = one_direct_predictor(var, level=level, grid='ext', model=model, scene=scene)['data']; i += 1
            # z
            for level in preds_levels:
                if 'z' + str(level) in preds:
                    data[i] = one_direct_predictor('z', level=level, grid='ext', model=model, scene=scene)['data']; i += 1
            # q
            for level in preds_levels:
                if 'q' + str(level) in preds:
                    data[i] = derived_predictors.specific_humidity(level, model=model, scene=scene)['data']; i += 1
            # r
            for level in preds_levels:
                if 'r' + str(level) in preds:
                    data[i] = derived_predictors.relative_humidity(level, model=model, scene=scene)['data']; i += 1
            # td
            for level in preds_levels:
                if 'td' + str(level) in preds:
                    data[i] = derived_predictors.dew_point(level, model=model, scene=scene)['data']; i += 1
            # Dtd
            for level in preds_levels:
                if 'Dtd' + str(level) in preds:
                        td = derived_predictors.dew_point(level, model=model, scene=scene)['data']
                        t = one_direct_predictor('t', level=level, grid='ext', model=model, scene=scene)['data']
                        data[i] = t - td; i+=1
            # vort, div
            for var in ['vort', 'div']:
                for level in preds_levels:
                    if var + str(level) in preds:
                        data[i] = derived_predictors.vorticity_and_divergence(model=model, scene=scene, level=level)['data'][var]; i += 1
            # thermal vertical gradients (tvg)
            for (level0, level1) in [(1000, 850), (850, 700), (700, 500)]:
                var = 'vtg_' + str(level0) + '_' + str(level1)
                if var in preds:
                    data[i] = derived_predictors.vtg(level0, level1, model=model,scene=scene)['data']; i += 1
            # ugsl, vgsl
            for var in ('u', 'v'):
                if var + 'gsl' in preds:
                    data[i] = derived_predictors.geostrophic(model=model, scene=scene)['data'][var + 'gsl']; i += 1
            # vortgsl, divgsl
            for var in ('vort', 'div'):
                if var + 'gsl' in preds:
                    data[i] = derived_predictors.vorticity_and_divergence(model=model, scene=scene, level='sl')['data'][var]; i += 1
            # Instability indexes
            if 'K_index' in preds:
                data[i] = derived_predictors.K_index(model=model, scene=scene)['data']; i += 1
            if 'TT_index' in preds:
                data[i] = derived_predictors.TT_index(model=model, scene=scene)['data']; i += 1
            if 'SSI_index' in preds:
                data[i] = derived_predictors.SSI_index(model=model, scene=scene)['data']; i += 1
            if 'LI_index' in preds:
                data[i] = derived_predictors.LI_index(model=model, scene=scene)['data']; i += 1

    # Select grid
    if grid == None:
        grid = field
    if grid in ('var', 'pred'):
        ilats, ilons = pred_ilats, pred_ilons
    elif grid == 'saf':
        ilats, ilons = saf_ilats, saf_ilons
    data = data[:, :, ilats]
    data = data[:, :, :, ilons]
    data = np.swapaxes(data, 0, 1)

    # If a selection of dates is desired
    if period != None:
        if model == 'reanalysis':
            if period == 'calibration':
                years = calibration_years
            elif period == 'reference':
                years = reference_years
            else:
                print(model, period, 'not valid')
                exit()
        else:
            if scene in ('historical', 'HISTORICAL') and period == 'reference':
                years = reference_years
            else:
                print(model, period, 'not valid')
                exit()

        idates = [i for i in range(len(dates)) if dates[i].year >= years[0] and dates[i].year <= years[1]]
        dates = list(np.asarray(dates)[idates])
        data = data[idates]

    return {'data': data, 'times': dates}


########################################################################################################################
def hres_metadata(var0, GCM_local=None, RCM_local=None, pathIn=None):
    """
    Read metadata of stations (high resolution grid) and returns a pandas dataframe with id, lon, lat and height
    :return:
    """

    if pathIn != None:
        dataPath = pathIn
    elif GCM_local != None:
        dataPath = '../input_data/OBS_PSEUDO/hres_' + GCM_local + '_' + RCM_local + '/'
    else:
        dataPath = pathHres

    masterFile = dataPath + var0 + '_hres_metadata.txt'

    #------------------------
    # read master file
    #------------------------
    npMaster = np.loadtxt(masterFile)
    # df = pd.DataFrame({'id': npMaster[:,0], 'lons': npMaster[:,1], 'lats': npMaster[:,2], 'h': npMaster[:,3]})
    df = pd.DataFrame({'id': npMaster[:,0], 'lons': npMaster[:,1], 'lats': npMaster[:,2]})
    df['id'] = df['id'].astype(int)
    df.set_index("id", inplace=True)

    return df

########################################################################################################################
def hres_data(var, period=None):
    """
    This function reads hres_data and returs dates and data in a dictionary.
    When there are predictands out of their codification range, it will inform and codification range might be changed
    at settings.
    PeriodFilename: returns only data from selected period
    """

    filename = pathHres + var + '_' + hresPeriodFilename[var[0]]

    minYear = int(hresPeriodFilename[var[0]].split('-')[0][:4])
    maxYear = int(hresPeriodFilename[var[0]].split('-')[1][:4])

    # If data is in ASCII, binary files are created for a faster reading
    if not os.path.isfile(filename +'.npy'):
        aux_lib.prepare_hres_data_ascii2npy(var)

    # ------------------------
    # read data
    # ------------------------
    data = np.load(filename +'.npy')

    # ------------------------
    # Creates list of dates
    # ------------------------
    times = []
    for time in data[:,0]:
        year = int(str(time)[:4])
        month = int(str(time)[4:6])
        day = int(str(time)[6:8])
        times.append(datetime.date(year=year, month=month, day=day))

    # Separates data from dates
    data = data[:, 1:]

    # Set period
    if period== None:
        first_date = datetime.datetime(minYear, 1, 1, 12, 0)
        last_date = datetime.datetime(maxYear, 12, 31, 12, 0)
        dates = [first_date + datetime.timedelta(days=i) for i in range((last_date - first_date).days + 1)]
    elif period== 'calibration':
        dates = calibration_dates
    elif period== 'training':
        dates = training_dates
    elif period== 'testing':
        dates = testing_dates
    elif period== 'reference':
        dates = reference_dates
    elif period== 'biasCorr':
        dates = biasCorr_dates

    # Select period
    dates = [x.date() for x in dates]
    idates = [times.index(time) for time in times if time in dates]
    data = data[idates]
    times = [times[i] for i in range(len(times)) if i in idates]

    # Checks for values out of range and counts percentage of missing data
    out_of_range = np.where((data < predictands_codification[var]['min_valid']) |
                         (data > predictands_codification[var]['max_valid']))[0].size
    data[np.isnan(data)] = predictands_codification[var]['special_value']
    special_value = predictands_codification[var]['special_value']
    missing_data = np.where(data == special_value)[0].size
    if missing_data > 0:
        perc = np.round(100*missing_data/data.size, 2)
        print('\nInfo: predictands contain', missing_data, 'missing data (', perc, '% )')
        print('Do not worry about the "RuntimeWarning: invalid value encountered"')
        print('It is showed because there are np.nan, but they are properly handled by the program.')
        aux = 1*(data == special_value)
        aux = 100*np.mean(aux, axis=0)
        if np.max(aux) >= max_perc_missing_predictands_allowed:
            print('At least one point contains too many missing data (', np.max(aux), '%) at', period, 'period')

    if out_of_range > 0:
        exit('Predictands contain values out of range. Change predictands_codification at advanced_settings')

    return {'data': data, 'times': times}