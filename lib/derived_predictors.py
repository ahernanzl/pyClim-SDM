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
def SSI_index(model='reanalysis', scene='TESTING'): # author: Carlos Correa ; email:ccorreag@aemet.es
    """    Showalter index:    SSI (K) = t500 - tp500   
    where:
    t500 is the measured temperature at 500 hPa
    tp500 is the temperature of the parcel at 500 hPa when lifted from 850 hPa"""
    
    #from scipy.optimize import fsolve --> it was added in /config/imports.py
    
    # Prepare times
    times = read.one_direct_predictor('t', level=850, grid='ext', model=model, scene=scene)['times']
    if model == 'reanalysis':
        dates = calibration_dates
    else:
        dates = times
    idates = [i for i in range(len(times)) if times[i] in dates]

    # Read data
    t850 = read.one_direct_predictor('t', level=850, grid='ext', model=model, scene=scene)['data'][idates]
    z850 = read.one_direct_predictor('z', level=850, grid='ext', model=model, scene=scene)['data'][idates]
    t500 = read.one_direct_predictor('t', level=500, grid='ext', model=model, scene=scene)['data'][idates]
    td850 =q2Td(850, model=model, scene=scene)
    
    # Constants
    cp = 1005 # Isobaric specific heat in dry air
    R = 287  # Dry air constant
    g = 9.8  # Gravity
    L = 2.5 * 10 ** 6 # Latent heat of vaporization of water
    
    # Calculate Lifted Condensation Level using Lawrence's simple formula 
    '''(visit: https://journals.ametsoc.org/downloadpdf/journals/bams/86/2/bams-86-2-225.pdf)'''
    LCL = 125*(t850 - td850) + z850

    # Calculate LCL temperature
    tLCL = t850 - LCL * g / cp

    # Define Magnus equation
    def magnus(t):
        return 6.11 * 100 * pow(10, 7.4475 * (t -273.15) / (234.07 + (t -273.15)))
	
    # Calculate saturation vapour pressure at LCL temperature
    es_tLCL = magnus(tLCL)

    # Calculate mixing ratio at t850
    r_t850 = 0.622 * magnus(td850) / (850 * 100 - magnus(td850))

    '''
    # Calculate mean value of lifting virtual temperature:
    (1) tvm = ((1 + 0.605 * r_t850) * t850 + (1 + 0.605 * rs_tLCL) * tLCL ) / 2
    # Calculate LCL pressure (hypsometric equation):
    (2) pLCL = 850 * 100 * np.exp(g * (z850 - LCL) / (R * tvm))
    # Calculate saturation mixing ratio at LCL temperature:
    (3) rs_tLCL = 0.622 * es_tLCL / (pLCL - es_tLCL)
    '''
    
    # Calculate LCL pressure solving implicit equation resulting from merging (1)&(2)&(3)
    def f_pLCL(pLCL):
        return pLCL - 850 * 100 * np.exp(g * (z850 - LCL) / (R * ((1 + 0.605 * r_t850) * t850 + (1 + 0.605 * 0.622 * es_tLCL / (pLCL - es_tLCL)) * tLCL ) / 2))
    pLCL = fsolve(f_pLCL,850 * 100)

    '''
    Calculation pseudoadiabatic equation constant KK: 
    cp * ln(t) - R * ln(p-es) + rs * L / t = constant value 
    where:
    cp is the isobaric specific heat in dry air
    T is the virtual temperature
    R is the dry air constant
    p is pressure
    es is saturation vapour pressure at t
    r is the mixing ratio at T and p
    L is the heat of vaporization of water at t
    '''
    
    # Calculate pseudoadiabatic equation constant
    KK = cp * np.log(tLCL) - R * np.log(pLCL - es_tLCL) + (0.622 * es_tLCL / (pLCL - es_tLCL)) * L / tLCL

    # Calculate Tp500 solving implicit pseudoadiabatic equation
    def f_tp500(tp500):
        return KK - cp * np.log(tp500) + R * np.log(500*100 - magnus(tp500)) - (0.622 * magnus(tp500) / (pLCL - magnus(tp500))) * L / tp500
    tp500 = fsolve(f_tp500,t500)
 
    # Calculate SSI index
    SSI_index = t500 - tp500

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'SSI_index.nc', 'SSI_index', SSI_index, 'K', ext_lats, ext_lons, dates)

    return SSI_index

########################################################################################################################
def LI_index(model='reanalysis', scene='TESTING'): # author: Carlos Correa ; email:ccorreag@aemet.es
    """    Lifted index:    LI (K) = t500 - tp500   
    where:
    t500 is the measured temperature at 500 hPa
    tp500 is the temperature of the parcel at 500 hPa when lifted from surface pressure"""

    #from scipy.optimize import fsolve --> it was added in /config/imports.py
    
    # Prepare times
    times = read.one_direct_predictor('t', level=500, grid='ext', model=model, scene=scene)['times']
    if model == 'reanalysis':
        dates = calibration_dates
    else:
        dates = times
    idates = [i for i in range(len(times)) if times[i] in dates]

    # Read data
    t500 = read.one_direct_predictor('t', level=500, grid='ext', model=model, scene=scene)['data'][idates]
    mslp = read.one_direct_predictor('mslp', level=None, grid='ext', model=model, scene=scene)['data'][idates]
    t2m = read.one_direct_predictor('t2m', level=None, grid='ext', model=model, scene=scene)['data'][idates]
    q1000 = read.one_direct_predictor('q', level=1000, grid='ext', model=model, scene=scene)['data'][idates] # q1000 used instead of q2m (huss ESGF and 2d(ID:168) to calculate huss in ERA5 are not available)
    q2m = q1000 # q1000 is used instead of q2m because huss (ESGF) and 2d (ID:168 ERA5) are not available)
    
    # Constants
    cp = 1005 # Isobaric specific heat in dry air
    R = 287  # Dry air constant
    Rv = 461 # Water vapour constant
    g = 9.8  # Gravity
    L = 2.5 * 10 ** 6 # Latent heat of vaporization of water
    
    # Calculate dew point
    td2m = 1 / (1 / 273 - (Rv / L) * np.log(mslp * q2m / (0.622 * 6.11 * 100)))
    
    # Calculate Lifted Condensation Level using Lawrence's simple formula 
    '''(visit: https://journals.ametsoc.org/downloadpdf/journals/bams/86/2/bams-86-2-225.pdf)'''
    LCL = 125*(t2m - td2m)

    # Calculate LCL temperature
    tLCL = t2m - LCL * g / cp

    # Define Magnus equation
    def magnus(t):
        return 6.11 * 100 * pow(10, 7.4475 * (t -273.15) / (234.07 + (t -273.15)))
	
    # Calculate saturation vapour pressure at LCL temperature
    es_tLCL = magnus(tLCL)

    # Calculate mixing ratio at the surface
    r_t2m = 0.622 * magnus(td2m) / ( mslp - magnus(td2m))

    '''
    # Calculate mean value of lifting virtual temperature:
    (1) tvm = ((1 + 0.605 * r_t2m) * t2m + (1 + 0.605 * rs_tLCL) * tLCL ) / 2
    # Calculate LCL pressure (hypsometric equation):
    (2) pLCL =  mslp * np.exp(g * (0 - LCL) / (R * tvm))
    # Calculate saturation mixing ratio at LCL temperature:
    (3) rs_tLCL = 0.622 * es_tLCL / (pLCL - es_tLCL)
    '''
    
    # Calculate LCL pressure solving implicit equation resulting from merging (1)&(2)&(3)
    def f_pLCL(pLCL):
        return pLCL - mslp * np.exp(g * (0 - LCL) / (R * ((1 + 0.605 * r_t2m) * t2m + (1 + 0.605 * 0.622 * es_tLCL / (pLCL - es_tLCL)) * tLCL ) / 2))
    pLCL = fsolve(f_pLCL,mslp)

    '''
    Calculation pseudoadiabatic equation constant KK: 
    cp * ln(t) - R * ln(p-es) + rs * L / t = constant value 
    where:
    cp is the isobaric specific heat in dry air
    T is the virtual temperature
    R is the dry air constant
    p is pressure
    es is saturation vapour pressure at t
    r is the mixing ratio at T and p
    L is the heat of vaporization of water at t
    '''
    
    # Calculate pseudoadiabatic equation constant
    KK = cp * np.log(tLCL) - R * np.log(pLCL - es_tLCL) + (0.622 * es_tLCL / (pLCL - es_tLCL)) * L / tLCL

    # Calculate Tp500 solving implicit pseudoadiabatic equation
    def f_tp500(tp500):
        return KK - cp * np.log(tp500) + R * np.log(500*100 - magnus(tp500)) - (0.622 * magnus(tp500) / (pLCL - magnus(tp500))) * L / tp500
    tp500 = fsolve(f_tp500,t500)
 
    # Calculate SSI index
    LI_index = t500 - tp500

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'LI_index.nc', 'LI_index', LI_index, 'K', ext_lats, ext_lons, dates)

    return LI_index

########################################################################################################################
def K_index(model='reanalysis', scene='TESTING'):
    """    Instability index:    K-Index (K) = (T850 - T500) + Td850 - (T700 - Td700)     """

    # Prepare times
    times = read.one_direct_predictor('t', level=850, grid='ext', model=model, scene=scene)['times']
    if model == 'reanalysis':
        dates = calibration_dates
    else:
        dates = times
    idates = [i for i in range(len(times)) if times[i] in dates]

    # Read data
    t850 = read.one_direct_predictor('t', level=850, grid='ext', model=model, scene=scene)['data'][idates]
    t700 = read.one_direct_predictor('t', level=700, grid='ext', model=model, scene=scene)['data'][idates]
    t500 = read.one_direct_predictor('t', level=500, grid='ext', model=model, scene=scene)['data'][idates]
    td850 = q2Td(850, model=model, scene=scene)
    td700 = q2Td(700, model=model, scene=scene)

    K_index = (t850 - t500) + td850 - (t700 - td700)

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'K_index.nc', 'K_index', K_index, 'K', ext_lats, ext_lons, dates)

    return K_index


########################################################################################################################
def TT_index(model='reanalysis', scene='TESTING'):
    """    Total Totals index:  TT = (T850 – T500) + (Td850 – T500)  =   T850 + Td850 – 2(T500)     """

    # Prepare times
    times = read.one_direct_predictor('t', level=850, grid='ext', model=model, scene=scene)['times']
    if model == 'reanalysis':
        dates = calibration_dates
    else:
        dates = times
    idates = [i for i in range(len(times)) if times[i] in dates]

    # Read data
    t850 = read.one_direct_predictor('t', level=850, grid='ext', model=model, scene=scene)['data'][idates]
    t500 = read.one_direct_predictor('t', level=500, grid='ext', model=model, scene=scene)['data'][idates]
    td850 = q2Td(850, model=model, scene=scene)

    TT_index = t850 + td850 - 2*t500

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'TT_index.nc', 'TT_index', TT_index, 'K', ext_lats, ext_lons, dates)

    return TT_index

########################################################################################################################
def q2r(level, model='reanalysis', scene='TESTING'):
    """specific humidity to relative humidity
    p in mb (hPa)
    t in Kelvin
    q dimensionless
    """

    # Read data
    if model == 'reanalysis':
        dates = calibration_dates
        aux = read.one_direct_predictor('t', level=level, grid='ext', model=model, scene=scene)
        times = aux['times']
        idates = [i for i in range(len(times)) if times[i] in dates]
        t = aux['data'][idates]
        q = read.one_direct_predictor('q', level=level, grid='ext', model=model, scene=scene)['data'][idates]
    else:
        aux = read.one_direct_predictor('t', level=level, grid='ext', model=model, scene=scene)
        dates = aux['times']
        t = aux['data']
        q = read.one_direct_predictor('q', level=level, grid='ext', model=model, scene=scene)['data']

    L = 2.5 * 10 ** 6
    Rv = 461
    p = level

    es = 6.11 * np.exp((L / Rv) * (1 / 273 - 1 / t))
    e = q * p / (0.622 + 0.378 * q)
    h = 100 * e / es

    # r = e / (p-e)
    # rs =  es / (p - es)
    # h = 100*r/rs

    warnings.filterwarnings("ignore", message="invalid value encountered in greater")
    warnings.filterwarnings("ignore", message="invalid value encountered in less")
    h[h < 0] = 0
    h[h > 100] = 100

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'r' + str(level) + '.nc', 'r', h, '%', ext_lats, ext_lons, dates)

    return h


# ########################################################################################################################
def q2Td(level, model='reanalysis', scene='TESTING'):
    """specific humidity to relative humidity
    p in mb (hPa)
    t in Kelvin
    q dimensionless
    """

    # Read data
    if model == 'reanalysis':
        dates = calibration_dates

        aux = read.one_direct_predictor('q', level=level, grid='ext', model=model, scene=scene)
        times = aux['times']
        idates = [i for i in range(len(times)) if times[i] in dates]
        q = aux['data'][idates]
    else:
        aux = read.one_direct_predictor('q', level=level, grid='ext', model=model, scene=scene)
        dates = aux['times']
        q = aux['data']

    L = 2.5 * 10 ** 6
    Rv = 461
    p = level

    td = 1 / (1 / 273 - (Rv / L) * np.log(p * q / (0.622 * 6.11)))

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'td' + str(level) + '.nc', 'td', td, 'K', ext_lats, ext_lons, dates)

    return td


########################################################################################################################
def vtg(level0, level1, model='reanalysis', scene='TESTING'):
    """Gradient thermal vertical between level0 and level1 hPa"""

    # Read data
    if model == 'reanalysis':
        dates = calibration_dates

        aux = read.one_direct_predictor('t', level=level1, grid='ext', model=model, scene=scene)
        times = aux['times']
        idates = [i for i in range(len(times)) if times[i] in dates]
        t_level1 = aux['data'][idates]
        t_level0 = read.one_direct_predictor('t', level=level0, grid='ext', model=model, scene=scene)['data'][idates]
    else:
        aux = read.one_direct_predictor('t', level=level1, grid='ext', model=model, scene=scene)
        dates = aux['times']
        t_level1 = aux['data']
        t_level0 = read.one_direct_predictor('t', level=level0, grid='ext', model=model, scene=scene)['data']

    # Calculate GTV
    vtg = t_level0 - t_level1

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'vtg_' + str(level0) + '_' + str(level1) + '.nc',
                     'vtg_' + str(level0) + '_' + str(level1), vtg, 'K',
                     ext_lats, ext_lons, dates)

    return vtg


########################################################################################################################
def vorticity_and_divergence(model='reanalysis', scene='TESTING', level=None):
    """Vorticity and divergence"""

    # Read data
    if model == 'reanalysis':
        if level == 'sl':
            dates = calibration_dates
            aux = read.netCDF(pathAux + 'DERIVED_PREDICTORS/', 'ugsl.nc', 'u', grid='ext')
            times = aux['times']
            idates = [i for i in range(len(times)) if times[i] in dates]
            u = aux['data'][idates]
            v = read.netCDF(pathAux + 'DERIVED_PREDICTORS/', 'vgsl.nc', 'v', grid='ext')['data'][idates]
            sufix = 'gsl'
        else:
            dates = calibration_dates
            aux = read.one_direct_predictor('u', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            idates = [i for i in range(len(times)) if times[i] in dates]
            u = aux['data'][idates]
            v = read.one_direct_predictor('v', level=level, grid='ext', model=model, scene=scene)['data'][idates]
            sufix = str(level)
    else:
        if level == 'sl':
            ncName = list(preds_dict['p'].keys())[0]
            dates = read.one_direct_predictor(ncName, level=None, grid='ext', model=model, scene=scene)['times']
            aux = geostrophic(model=model, scene=scene)
            u, v = aux['ugsl'], aux['vgsl']
            sufix = 'gsl'
        else:
            aux = read.one_direct_predictor('u', level=level, grid='ext', model=model, scene=scene)
            dates = aux['times']
            u = aux['data']
            v = read.one_direct_predictor('v', level=level, grid='ext', model=model, scene=scene)['data']
            sufix = str(level)

    # Calculate wind gradients
    ndates = len(dates)
    delta_x = []
    for lat in ext_lats:
        delta_x.append(dist((lat, -grid_res / 2), (lat, grid_res / 2)).km)
    delta_x = 1000 * np.asarray(delta_x)
    delta_x = np.repeat(delta_x, ext_nlons).reshape(ext_nlats, ext_nlons)
    delta_x = np.repeat(delta_x[np.newaxis, :, :], ndates, axis=0).reshape(ndates, ext_nlats, ext_nlons)
    delta_y = 1000 * dist((-grid_res / 2, 0), (grid_res / 2, 0)).km

    # Calculate vorticity
    grad_uy = -np.gradient(u)[1] / delta_y
    grad_vx = np.gradient(v)[2] / delta_x
    vort = grad_vx - grad_uy

    # Calculate divergence
    grad_ux = np.gradient(v)[1] / delta_x
    grad_vy = -np.gradient(u)[2] / delta_y
    div = grad_ux + grad_vy

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'vort' + sufix + '.nc', 'vort', vort, 's-1', ext_lats, ext_lons, dates)
        write.netCDF(pathOut, 'div' + sufix + '.nc', 'div', div, 's-1', ext_lats, ext_lons, dates)

    return {'vort': vort, 'div': div}


########################################################################################################################
def mslp_trend(model='reanalysis', scene='TESTING'):
    """mslp trend from previous day"""

    # Read data
    if model == 'reanalysis':
        dates = calibration_dates
        aux = read.one_direct_predictor('mslp', level=None, grid='ext', model=model, scene=scene)
        times = aux['times']
        idates = [i for i in range(len(times)) if times[i] in dates]
        mslp = aux['data'][idates]
    else:
        aux = read.one_direct_predictor('mslp', level=None, grid='ext', model=model, scene=scene)
        dates = aux['times']
        mslp = aux['data']

    # Calculate mslp_trend
    mslp_dayBefore = np.copy(mslp)
    mslp_dayBefore[1:][:][:] = mslp[:-1][:][:]
    mslp_trend = mslp - mslp_dayBefore

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'mslp_trend.nc', 'mslp_trend', mslp_trend, 'Pa', ext_lats, ext_lons, dates)

    return mslp_trend


########################################################################################################################
def insolation(model='reanalysis', scene='TESTING'):
    """Theoretical insolation as sin function (between 0 and 1)"""
    pi = 3.14

    # Read data
    if model == 'reanalysis':
        dates = calibration_dates
    else:
        ncName = list(preds_dict['p'].keys())[0]
        dates = read.one_direct_predictor(ncName, level=None, grid='ext', model=model, scene=scene)['times']

    # Calculate ins
    ins = []
    for date in dates:
        ndays = datetime.date(date.year, 12, 31).timetuple().tm_yday
        equinox = datetime.date(date.year, 3, 21).timetuple().tm_yday
        iday = date.timetuple().tm_yday
        ins.append(math.sin(2 * pi * (iday - equinox) / float(ndays)))

    ins = np.asarray(ins)
    ins = ins[:, np.newaxis, np.newaxis]
    ins = np.repeat(ins, ext_nlats, axis=1)
    ins = np.repeat(ins, ext_nlons, axis=2)

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'ins.nc', 'ins', ins, '', ext_lats, ext_lons, dates)

    return ins


########################################################################################################################
def geostrophic(model='reanalysis', scene='TESTING'):
    # Define constant parameters
    R = 287  # Dry air constant
    alpha = 0.0065  # Vertical temperature standar gradient
    g = 9.8  # Gravity
    pi = 3.14159
    omega = 2 * pi / 86400  # Angular velocity

    # Read data
    if model == 'reanalysis':
        level = 1000
        aux = read.one_direct_predictor('t', level=level, grid='ext', model=model, scene=scene)
        times = aux['times']
        idates = [i for i in range(len(times)) if times[i] in calibration_dates]
        dates = calibration_dates
        t1000 = aux['data'][idates]
        mslp = read.one_direct_predictor('mslp', level=None, grid='ext', model=model, scene=scene)['data'][idates]
        tsl = t1000 / (100000 / mslp) ** (R * alpha / g)
        denssl = mslp / (R * tsl)
    else:
        aux = read.one_direct_predictor('mslp', level=None, grid='ext', model=model, scene=scene)
        dates = aux['times']
        mslp = aux['data']
        denssl = 1.225  # Density

    ndates = len(dates)

    # Calculate lat lon distances in meters
    delta_x = []
    for lat in ext_lats:
        delta_x.append(dist((lat, -grid_res / 2), (lat, grid_res / 2)).km)
    delta_x = 1000 * np.asarray(delta_x)
    delta_x = np.repeat(delta_x, ext_nlons).reshape(ext_nlats, ext_nlons)
    delta_x = np.repeat(delta_x[np.newaxis, :, :], ndates, axis=0).reshape(ndates, ext_nlats, ext_nlons)
    delta_y = 1000 * dist((-grid_res / 2, 0), (grid_res / 2, 0)).km

    # Calculate coriolis parameter
    f = 2 * omega * np.sin(np.deg2rad(np.asarray(ext_lats)))
    f = np.repeat(f, ext_nlons).reshape(ext_nlats, ext_nlons)
    f = np.repeat(f[np.newaxis, :, :], ndates, axis=0).reshape(ndates, ext_nlats, ext_nlons)

    # Calculate gradient of pressure
    grad = np.gradient(mslp)
    grad_x = grad[2]
    grad_y = -grad[1]
    del grad
    grad_x = grad_x
    grad_y = grad_y
    grad_x /= delta_x
    grad_y /= delta_y
    ugsl = (-grad_y / f)
    vgsl = (grad_x / f)
    ugsl /= denssl
    vgsl /= denssl

    # Save to netCDF file
    if model == 'reanalysis':
        pathOut = pathAux + 'DERIVED_PREDICTORS/'
        write.netCDF(pathOut, 'ugsl.nc', 'u', ugsl, 'm/s', ext_lats, ext_lons, dates)
        write.netCDF(pathOut, 'vgsl.nc', 'v', vgsl, 'm/s', ext_lats, ext_lons, dates)

    return {'ugsl': ugsl, 'vgsl': vgsl}


########################################################################################################################
def reanalysis_all():
    """
    Calls to all derived predictors so .nc files are generated
    """

    print('calculating derived predictors reanalysis')

    # Create pathOut
    pathOut = pathAux + 'DERIVED_PREDICTORS/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)


    if 'gsl' in all_preds:
        geostrophic()
    for (level0, level1) in [(1000, 850), (850, 700), (700, 500)]:
        if 'vtg_' + str(level0) + '_' + str(level1) in all_preds:
            vtg(level0, level1)
    if ('vorgtsl' in all_preds) or ('divgsl' in all_preds):
        vorticity_and_divergence(level='sl')
    if 'mslp_trend' in all_preds:
        mslp_trend()
    if 'ins' in all_preds:
        insolation()
    if 'K_index' in all_preds:
        K_index()
    if 'TT_index' in all_preds:
        TT_index()
    if 'SSI_index' in all_preds:
        SSI_index()
    if 'LI_index' in all_preds:
        LI_index()

    for level in preds_levels:
        print('derived predictors', level)
        if ('vort' + str(level) in all_preds) or ('div' + str(level) in all_preds):
            vorticity_and_divergence(level=level)
        if 'r' + str(level) in all_preds:
            q2r(level)
        if ('td' + str(level) in all_preds) or ('Dtd' + str(level) in all_preds):
            q2Td(level)




