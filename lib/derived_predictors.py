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
def Clausius_Clapeyron(t, units_kelvin=True):
    """
    Calculate saturation vapor pressure from temperature
    :param t: in Kelvin (if Celcious, set units_kelvin to False)
    :return: saturation vapor pressure
    """

    t0 = 273.15
    L = 2.5 * 10 ** 6
    Rv = 461

    if units_kelvin == False:
        t += t0

    invalid = np.where(t == 0)
    t[invalid] = 1
    es = 6.11 * np.exp((L / Rv) * (1 / t0 - 1 / t))
    es[invalid] = np.nan

    return es


########################################################################################################################
def Clausius_Clapeyron_inverse(es):
    """
    Calculate temperature from saturation vapor pressure
    :return: temperature in Kelvin
    """

    t0 = 273.15
    L = 2.5 * 10 ** 6
    Rv = 461
    t = 1 / [(1 / t0) - (np.log(es / 6.11) / (L / Rv))]

    aux = (es / 6.11)
    invalid = np.where(aux == 0)
    aux[invalid] = 1
    t = 1 / [(1 / t0) - (np.log(es / 6.11) / (L / Rv))]
    t[invalid] = np.nan

    return t


########################################################################################################################
def SSI_index(model='reanalysis', scene='TESTING'):  # author: Carlos Correa ; email:ccorreag@aemet.es
    """    Showalter index:    SSI (K) = t500 - tp500
    where:
    t500 is the measured temperature at 500 hPa
    tp500 is the temperature of the parcel at 500 hPa when lifted from 850 hPa"""
    from scipy.optimize import fmin
    # from scipy.optimize import fsolve --> it was added in /config/imports.py

    # Prepare times
    times = read.one_direct_predictor('ta', level=850, grid='ext', model=model, scene=scene)['times']
    # if model == 'reanalysis':
    #     dates = calibration_dates
    # else:
    #     dates = times
    # idates = [i for i in range(len(times)) if times[i] in dates]

    # Read data
    t850 = read.one_direct_predictor('ta', level=850, grid='ext', model=model, scene=scene)['data']
    z850 = read.one_direct_predictor('zg', level=850, grid='ext', model=model, scene=scene)['data']
    t500 = read.one_direct_predictor('ta', level=500, grid='ext', model=model, scene=scene)['data']
    td850 = dew_point(850, model=model, scene=scene)['data']

    # Constants
    cp = 1005  # Isobaric specific heat in dry air
    R = 287  # Dry air constant
    g = 9.8  # Gravity
    L = 2.5 * 10 ** 6  # Latent heat of vaporization of water

    SSI_index_lst = []

    # convert arrays into 1-D vectors
    ravelz850 = z850.ravel()
    ravelt850 = t850.ravel()
    ravelt500 = t500.ravel()
    raveltd850 = td850.ravel()

    for j in range(0, len(ravelt500)):

        # % completed
        if j % 20000 == 0:
            print('calculating SSI_index: ' + str(round(j / len(ravelt500) * 100, 1)) + ' %')

        # select element
        jz850 = ravelz850[j]
        jt850 = ravelt850[j]
        jt500 = ravelt500[j]
        jtd850 = raveltd850[j]

        # Calculate Lifted Condensation Level using Lawrence's simple formula
        '''(visit: https://journals.ametsoc.org/downloadpdf/journals/bams/86/2/bams-86-2-225.pdf)'''
        LCL = 125 * (jt850 - jtd850) + jz850

        # Calculate LCL temperature
        tLCL = jt850 - LCL * g / cp

        # Define Magnus equation
        def magnus(t):
            return 6.11 * 100 * pow(10, 7.4475 * (t - 273.15) / (234.07 + (t - 273.15)))

        # Calculate saturation vapour pressure at LCL temperature
        es_tLCL = magnus(tLCL)

        # Calculate mixing ratio at t850
        r_t850 = 0.622 * magnus(jtd850) / (850 * 100 - magnus(jtd850))

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
            return pLCL - 850 * 100 * np.exp(g * (jz850 - LCL) / (R * (
                        (1 + 0.605 * r_t850) * jt850 + (1 + 0.605 * 0.622 * es_tLCL / (pLCL - es_tLCL)) * tLCL) / 2))

        pLCL = fsolve(f_pLCL, 850 * 100)

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
            return KK - cp * np.log(tp500) + R * np.log(500 * 100 - magnus(tp500)) - (
                        0.622 * magnus(tp500) / (pLCL - magnus(tp500))) * L / tp500

        tp500 = fsolve(f_tp500, jt500)

        # Calculate SSI index
        jSSI_index = jt500 - tp500
        SSI_index_lst.append(jSSI_index)

    SSI_index = np.array(SSI_index_lst).reshape(t500.shape)

    return {'data': SSI_index, 'times': times}


########################################################################################################################
def LI_index(model='reanalysis', scene='TESTING'):  # author: Carlos Correa ; email:ccorreag@aemet.es
    """    Lifted index:    LI (K) = t500 - tp500
    where:
    t500 is the measured temperature at 500 hPa
    tp500 is the temperature of the parcel at 500 hPa when lifted from surface pressure"""

    # from scipy.optimize import fsolve --> it was added in /config/imports.py

    # Prepare times
    times = read.one_direct_predictor('ta', level=500, grid='ext', model=model, scene=scene)['times']
    # if model == 'reanalysis':
    #     dates = calibration_dates
    # else:
    #     dates = times
    # idates = [i for i in range(len(times)) if times[i] in dates]

    # Read data
    t500 = read.one_direct_predictor('ta', level=500, grid='ext', model=model, scene=scene)['data']
    psl = read.one_direct_predictor('psl', level=None, grid='ext', model=model, scene=scene)['data']
    tas = read.one_direct_predictor('tas', level=None, grid='ext', model=model, scene=scene)['data']
    q1000 = read.one_direct_predictor('hus', level=1000, grid='ext', model=model, scene=scene)['data']
    # q1000 used instead of huss (huss ESGF and 2d(ID:168) to calculate huss in ERA5 are not available)
    huss = q1000  # q1000 is used instead of huss because huss (ESGF) and 2d (ID:168 ERA5) are not available)

    # Constants
    cp = 1005  # Isobaric specific heat in dry air
    R = 287  # Dry air constant
    Rv = 461  # Water vapour constant
    g = 9.8  # Gravity
    L = 2.5 * 10 ** 6  # Latent heat of vaporization of water

    LI_index_lst = []

    # convert arrays into 1-D vectors
    ravelt500 = t500.ravel()
    ravelpsl = psl.ravel()
    raveltas = tas.ravel()
    ravelhuss = huss.ravel()

    for j in range(0, len(ravelt500)):

        # % completed
        if j % 20000 == 0:
            print('calculating LI_index: ' + str(round(j / len(ravelt500) * 100, 1)) + ' %')

        # select element
        jt500 = ravelt500[j]
        jpsl = ravelpsl[j]
        jtas = raveltas[j]
        jhuss = ravelhuss[j]

        # Calculate dew point
        ttdps = 1 / (1 / 273 - (Rv / L) * np.log(jpsl * jhuss / (0.622 * 6.11 * 100)))

        # Calculate Lifted Condensation Level using Lawrence's simple formula
        '''(visit: https://journals.ametsoc.org/downloadpdf/journals/bams/86/2/bams-86-2-225.pdf)'''
        LCL = 125 * (jtas - ttdps)

        # Calculate LCL temperature
        tLCL = jtas - LCL * g / cp

        # Define Magnus equation
        def magnus(t):
            return 6.11 * 100 * pow(10, 7.4475 * (t - 273.15) / (234.07 + (t - 273.15)))

        # Calculate saturation vapour pressure at LCL temperature
        es_tLCL = magnus(tLCL)

        # Calculate mixing ratio at the surface
        r_tas = 0.622 * magnus(ttdps) / (jpsl - magnus(ttdps))

        '''
        # Calculate mean value of lifting virtual temperature:
        (1) tvm = ((1 + 0.605 * r_tas) * tas + (1 + 0.605 * rs_tLCL) * tLCL ) / 2
        # Calculate LCL pressure (hypsometric equation):
        (2) pLCL =  psl * np.exp(g * (0 - LCL) / (R * tvm))
        # Calculate saturation mixing ratio at LCL temperature:
        (3) rs_tLCL = 0.622 * es_tLCL / (pLCL - es_tLCL)
        '''

        # Calculate LCL pressure solving implicit equation resulting from merging (1)&(2)&(3)
        def f_pLCL(pLCL):
            return pLCL - jpsl * np.exp(g * (0 - LCL) / (
                        R * ((1 + 0.605 * r_tas) * jtas + (1 + 0.605 * 0.622 * es_tLCL / (pLCL - es_tLCL)) * tLCL) / 2))

        pLCL = fsolve(f_pLCL, jpsl)

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
            return KK - cp * np.log(tp500) + R * np.log(500 * 100 - magnus(tp500)) - (
                        0.622 * magnus(tp500) / (pLCL - magnus(tp500))) * L / tp500

        tp500 = fsolve(f_tp500, jt500)

        # Calculate LI index
        jLI_index = jt500 - tp500
        LI_index_lst.append(jLI_index)

    LI_index = np.array(LI_index_lst).reshape(t500.shape)

    return {'data': LI_index, 'times': times}


########################################################################################################################
def K_index(model='reanalysis', scene='TESTING'):
    """    Instability index:    K-Index (K) = (T850 - T500) + Td850 - (T700 - Td700)     """

    # Prepare times
    times = read.one_direct_predictor('ta', level=850, grid='ext', model=model, scene=scene)['times']

    # Read data
    t850 = read.one_direct_predictor('ta', level=850, grid='ext', model=model, scene=scene)['data']
    t700 = read.one_direct_predictor('ta', level=700, grid='ext', model=model, scene=scene)['data']
    t500 = read.one_direct_predictor('ta', level=500, grid='ext', model=model, scene=scene)['data']
    td850 = dew_point(850, model=model, scene=scene)['data']
    td700 = dew_point(700, model=model, scene=scene)['data']

    K_index = (t850 - t500) + td850 - (t700 - td700)

    return {'data': K_index, 'times': times}


########################################################################################################################
def TT_index(model='reanalysis', scene='TESTING'):
    """    Total Totals index:  TT = (T850 – T500) + (Td850 – T500)  =   T850 + Td850 – 2(T500)     """

    # Prepare times
    times = read.one_direct_predictor('ta', level=850, grid='ext', model=model, scene=scene)['times']

    # Read data
    t850 = read.one_direct_predictor('ta', level=850, grid='ext', model=model, scene=scene)['data']
    t500 = read.one_direct_predictor('ta', level=500, grid='ext', model=model, scene=scene)['data']
    td850 = dew_point(850, model=model, scene=scene)['data']

    TT_index = t850 + td850 - 2 * t500

    return {'data': TT_index, 'times': times}


########################################################################################################################
def aux_sfcWind_direct(level, model, scene):
    """get wind speed directly
    level: sfc or pressure level in mb
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('sfcWind', grid='ext', model=model, scene=scene)
            times = aux['times']
            sfcWind = aux['data']
        else:
            aux = read.one_direct_predictor('sfcWind', grid='ext', model=model, scene=scene)
            times = aux['times']
            sfcWind = aux['data']
    else:
        if model == 'reanalysis':
            aux = read.one_direct_predictor('sfcWind', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            sfcWind = aux['data']
        else:
            aux = read.one_direct_predictor('sfcWind', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            sfcWind = aux['data']

    return {'data': sfcWind, 'times': times}


########################################################################################################################
def aux_sfcWind_from_uas_vas(level, model, scene):
    """get wind speed indirectly from wind components
    level: sfc or pressure level in mb
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('uas', grid='ext', model=model, scene=scene)
            times = aux['times']
            u = aux['data']
            v = read.one_direct_predictor('vas', grid='ext', model=model, scene=scene)['data']
        else:
            aux = read.one_direct_predictor('uas', grid='ext', model=model, scene=scene)
            times = aux['times']
            u = aux['data']
            v = read.one_direct_predictor('vas', grid='ext', model=model, scene=scene)['data']
    else:
        if model == 'reanalysis':
            aux = read.one_direct_predictor('ua', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            u = aux['data']
            v = read.one_direct_predictor('va', level=level, grid='ext', model=model, scene=scene)['data']
        else:
            aux = read.one_direct_predictor('ua', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            u = aux['data']
            v = read.one_direct_predictor('va', level=level, grid='ext', model=model, scene=scene)['data']

    sfcWind = np.sqrt(u**2 + v**2)

    return {'data': sfcWind, 'times': times}


########################################################################################################################
def wind_speed(level, model='reanalysis', scene='TESTING'):
    """get wind speed directly or indirectly
    level: sfc or pressure level in mb
    """

    try:
        aux = aux_sfcWind_direct(level, model=model, scene=scene)
        sfcWind, dates = aux['data'], aux['times']
    except:
        print('wind speed', level, 'not available. Retrieving it indirectly')
        try:
            aux = aux_sfcWind_from_uas_vas(level, model=model, scene=scene)
            sfcWind, dates = aux['data'], aux['times']
        except:
            print('wind speed', level, 'not available neither directly nor indirectly')
            exit()

    warnings.filterwarnings("ignore", message="invalid value encountered in greater")
    warnings.filterwarnings("ignore", message="invalid value encountered in less")
    sfcWind[sfcWind < 0] = 0

    return {'data': sfcWind, 'times': dates}

########################################################################################################################
def aux_r_direct(level, model, scene):
    """get relative humidity directly
    level: sfc or pressure level in mb
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('hurs', grid='ext', model=model, scene=scene)
            times = aux['times']
            r = aux['data']
        else:
            aux = read.one_direct_predictor('hurs', grid='ext', model=model, scene=scene)
            times = aux['times']
            r = aux['data']
    else:
        if model == 'reanalysis':
            aux = read.one_direct_predictor('hur', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            r = aux['data']
        else:
            aux = read.one_direct_predictor('hur', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            r = aux['data']

    return {'data': r, 'times': times}


########################################################################################################################
def aux_r_from_q(level, model, scene):
    """get relative humidity indirectly from specific humidity
    level: sfc or pressure level in mb
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('tas', grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            q = read.one_direct_predictor('huss', grid='ext', model=model, scene=scene)['data']
            p = read.one_direct_predictor('ps', grid='ext', model=model, scene=scene)['data']
            p /= 100
        else:
            aux = read.one_direct_predictor('tas', grid='ext', model=model, scene=scene)
            dates = aux['times']
            t = aux['data']
            q = read.one_direct_predictor('huss', grid='ext', model=model, scene=scene)['data']
            p = read.one_direct_predictor('ps', grid='ext', model=model, scene=scene)['data']
            p /= 100
    else:
        if model == 'reanalysis':
            p = level
            aux = read.one_direct_predictor('ta', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            q = read.one_direct_predictor('hus', level=level, grid='ext', model=model, scene=scene)['data']
        else:
            p = level
            aux = read.one_direct_predictor('ta', level=level, grid='ext', model=model, scene=scene)
            dates = aux['times']
            t = aux['data']
            q = read.one_direct_predictor('hus', level=level, grid='ext', model=model, scene=scene)['data']

    es = Clausius_Clapeyron(t)

    aux = (0.622 + 0.378 * q)
    invalid = np.where(aux == 0)
    aux[invalid] = 1
    e = q * p / (0.622 + 0.378 * q)
    r = 100 * e / es
    r[invalid] = np.nan

    return {'data': r, 'times': times}


########################################################################################################################
def aux_r_from_Td(level, model, scene):
    """get relative humidity indirectly from specific humidity
    level: sfc or pressure level in mb
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('tas', grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            td = read.one_direct_predictor('tdps', grid='ext', model=model, scene=scene)['data']
        else:
            aux = read.one_direct_predictor('tas', grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            td = read.one_direct_predictor('tdps', grid='ext', model=model, scene=scene)['data']
    else:
        if model == 'reanalysis':
            aux = read.one_direct_predictor('ta', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            td = read.one_direct_predictor('Td', level=level, grid='ext', model=model, scene=scene)['data']
        else:
            aux = read.one_direct_predictor('ta', level=level, grid='ext', model=model, scene=scene)
            dates = aux['times']
            t = aux['data']
            td = read.one_direct_predictor('Td', level=level, grid='ext', model=model, scene=scene)['data']

    # Transform from Celcious to Kelvin
    t0 = 273.15
    if np.mean(t) < 100:
        t += t0
    if np.mean(td) < 100:
        td += t0

    e = Clausius_Clapeyron(td)
    es = Clausius_Clapeyron(t)
    r = 100 * e / es

    return {'data': r, 'times': times}


########################################################################################################################
def relative_humidity(level, model='reanalysis', scene='TESTING'):
    """get relative humidity directly or indirectly
    level: sfc or pressure level in mb
    """

    try:
        aux = aux_r_direct(level, model=model, scene=scene)
        r, times = aux['data'], aux['times']
    except:
        # print('relative humidity', level, 'not available. Retrieving it indirectly')
        try:
            aux = aux_r_from_q(level, model=model, scene=scene)
            r, times = aux['data'], aux['times']
        except:
            try:
                aux = aux_r_from_Td(level, model=model, scene=scene)
                r, times = aux['data'], aux['times']
            except:
                print('relative humidity', level, 'not available neither directly nor indirectly')
                exit()

    warnings.filterwarnings("ignore", message="invalid value encountered in greater")
    warnings.filterwarnings("ignore", message="invalid value encountered in less")
    r[r < 0] = 0
    r[r > 100] = 100

    return {'data': r, 'times': times}


########################################################################################################################
def aux_q_direct(level, model, scene):
    """get specific humidity directly
    level: sfc or pressure level in mb
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('huss', grid='ext', model=model, scene=scene)
            times = aux['times']
            q = aux['data']
        else:
            aux = read.one_direct_predictor('huss', grid='ext', model=model, scene=scene)
            times = aux['times']
            q = aux['data']
    else:
        if model == 'reanalysis':
            aux = read.one_direct_predictor('hus', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            q = aux['data']
        else:
            aux = read.one_direct_predictor('hus', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            q = aux['data']

    return {'data': q, 'times': times}


########################################################################################################################
def aux_q_from_r(level, model, scene):
    """get specific humidity indirectly from relative humidity
    level: sfc or pressure level in mb
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('tas', grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            r = read.one_direct_predictor('hurs', grid='ext', model=model, scene=scene)['data']
            p = read.one_direct_predictor('ps', grid='ext', model=model, scene=scene)['data']
        else:
            aux = read.one_direct_predictor('tas', grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            r = read.one_direct_predictor('hurs', grid='ext', model=model, scene=scene)['data']
            p = read.one_direct_predictor('ps', grid='ext', model=model, scene=scene)['data']
    else:
        if model == 'reanalysis':
            p = level
            aux = read.one_direct_predictor('ta', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            r = read.one_direct_predictor('hur', level=level, grid='ext', model=model, scene=scene)['data']
        else:
            p = level
            aux = read.one_direct_predictor('ta', level=level, grid='ext', model=model, scene=scene)
            dates = aux['times']
            t = aux['data']
            r = read.one_direct_predictor('hur', level=level, grid='ext', model=model, scene=scene)['data']

    es = Clausius_Clapeyron(t)
    e = r * es / 100
    q = e * 0.622 / (p - e * 0.378)

    return {'data': q, 'times': times}


########################################################################################################################
def aux_q_from_Td(level, model, scene):
    """get relative humidity indirectly from specific humidity
    level: sfc or pressure level in mb (if sfc, surface pressure will be converted to mb)
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('tdps', grid='ext', model=model, scene=scene)
            times = aux['times']
            td = aux['data']
            aux = read.one_direct_predictor('ps', grid='ext', model=model, scene=scene)
            units = aux['units']
            p = aux['data']
        else:
            aux = read.one_direct_predictor('tdps', grid='ext', model=model, scene=scene)
            times = aux['times']
            td = aux['data']
            aux = read.one_direct_predictor('ps', grid='ext', model=model, scene=scene)
            units = aux['units']
            p = aux['data']
        # Convert to hPa/mb
        if units == 'Pa':
            p /= 100
        elif units in ('hPa', 'mb'):
            pass
        else:
            print('Unknown units for surface pressure', units)
            exit()
    else:
        if model == 'reanalysis':
            aux = read.one_direct_predictor('Td', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            td = aux['data']
            p = level
        else:
            aux = read.one_direct_predictor('Td', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            td = aux['data']
            p = level

    L = 2.5 * 10 ** 6
    Rv = 461

    invalid = np.where(td == 0)
    td[invalid] = 1
    q = (0.622 * 6.11 / p) * np.exp((1 / 273.15 - 1 / td) / (Rv / L))
    q[invalid] = np.nan

    return {'data': q, 'times': times}


########################################################################################################################
def specific_humidity(level, model='reanalysis', scene='TESTING'):
    """get specific humidity directly or indirectly
    level: sfc or pressure level in mb
    """

    try:
        aux = aux_q_direct(level, model=model, scene=scene)
        q, times = aux['data'], aux['times']
    except:
        try:
            aux = aux_q_from_r(level, model=model, scene=scene)
            q, times = aux['data'], aux['times']
        except:
            try:
                aux = aux_q_from_Td(level, model=model, scene=scene)
                q, times = aux['data'], aux['times']
            except:
                print('specific humidity', level, 'not available neither directly nor indirectly')
                exit()

    warnings.filterwarnings("ignore", message="invalid value encountered in greater")
    warnings.filterwarnings("ignore", message="invalid value encountered in less")

    return {'data': q, 'times': times}


########################################################################################################################
def aux_Td_direct(level, model, scene):
    """get dew point directly
    level: sfc or pressure level in mb
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('tdps', grid='ext', model=model, scene=scene)
            times = aux['times']
            td = aux['data']
        else:
            aux = read.one_direct_predictor('tdps', grid='ext', model=model, scene=scene)
            dates = aux['times']
            td = aux['data']
    else:
        if model == 'reanalysis':
            aux = read.one_direct_predictor('Td', grid='ext', model=model, scene=scene)
            times = aux['times']
            td = aux['data']
        else:
            aux = read.one_direct_predictor('Td', grid='ext', model=model, scene=scene)
            times = aux['times']
            td = aux['data']

    return {'data': td, 'times': times}


########################################################################################################################
def aux_Td_from_q(level, model, scene):
    """get dew point indirectly from specific humidity
    level: sfc or pressure level in mb (if sfc, surface pressure will be converted to mb)
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('huss', grid='ext', model=model, scene=scene)
            times = aux['times']
            q = aux['data']
            aux = read.one_direct_predictor('ps', grid='ext', model=model, scene=scene)
            units = aux['units']
            p = aux['data']
        else:
            aux = read.one_direct_predictor('huss', grid='ext', model=model, scene=scene)
            times = aux['times']
            q = aux['data']
            aux = read.one_direct_predictor('ps', grid='ext', model=model, scene=scene)
            units = aux['units']
            p = aux['data']
        # Convert to hPa/mb
        if units == 'Pa':
            p /= 100
        elif units in ('hPa', 'mb'):
            pass
        else:
            print('Unknown units for surface pressure', units)
            exit()
    else:
        if model == 'reanalysis':
            p = level
            aux = read.one_direct_predictor('hus', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            q = aux['data']
        else:
            p = level
            aux = read.one_direct_predictor('hus', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            q = aux['data']

    L = 2.5 * 10 ** 6
    Rv = 461

    invalid = np.where(q == 0)
    q[invalid] = 1
    td = 1 / (1 / 273 - (Rv / L) * np.log(p * q / (0.622 * 6.11)))
    td[invalid] = np.nan

    return {'data': td, 'times': times}


########################################################################################################################
def aux_Td_from_r(level, model, scene):
    """get dew point indirectly from relative humidity
    level: sfc or pressure level in mb
    """

    if level == 'sfc':
        if model == 'reanalysis':
            aux = read.one_direct_predictor('tas', grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            r = read.one_direct_predictor('hurs', grid='ext', model=model, scene=scene)['data']
        else:
            aux = read.one_direct_predictor('tas', grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            r = read.one_direct_predictor('hurs', grid='ext', model=model, scene=scene)['data']
    else:
        if model == 'reanalysis':
            aux = read.one_direct_predictor('ta', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            r = read.one_direct_predictor('hur', level=level, grid='ext', model=model, scene=scene)['data']
        else:
            aux = read.one_direct_predictor('ta', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            t = aux['data']
            r = read.one_direct_predictor('hur', level=level, grid='ext', model=model, scene=scene)['data']

    es = Clausius_Clapeyron(t)
    e = r * es / 100
    td = Clausius_Clapeyron_inverse(e)

    return {'data': td, 'times': times}


########################################################################################################################
def dew_point(level, model='reanalysis', scene='TESTING'):
    """get dew point directly or indirectly
    level: sfc or pressure level in mb
    """

    try:
        aux = aux_Td_direct(level, model=model, scene=scene)
        td, times = aux['data'], aux['times']
    except:
        # print('dew point', level, 'not available. Retrieving it indirectly')
        try:
            aux = aux_Td_from_q(level, model=model, scene=scene)
            td, times = aux['data'], aux['times']
        except:
            try:
                aux = aux_Td_from_r(level, model=model, scene=scene)
                td, times = aux['data'], aux['times']
            except:
                print('dew point', level, 'not available neither directly nor indirectly')
                exit()

    warnings.filterwarnings("ignore", message="invalid value encountered in greater")
    warnings.filterwarnings("ignore", message="invalid value encountered in less")

    return {'data': td, 'times': times}


########################################################################################################################
def vtg(level0, level1, model='reanalysis', scene='TESTING'):
    """Gradient thermal vertical between level0 and level1 hPa"""

    # Read data
    if model == 'reanalysis':
        aux = read.one_direct_predictor('ta', level=level1, grid='ext', model=model, scene=scene)
        times = aux['times']
        t_level1 = aux['data']
        t_level0 = read.one_direct_predictor('ta', level=level0, grid='ext', model=model, scene=scene)['data']
    else:
        aux = read.one_direct_predictor('ta', level=level1, grid='ext', model=model, scene=scene)
        times = aux['times']
        t_level1 = aux['data']
        t_level0 = read.one_direct_predictor('ta', level=level0, grid='ext', model=model, scene=scene)['data']

    # Calculate GTV
    vtg = t_level0 - t_level1

    return {'data': vtg, 'times': times}


########################################################################################################################
def vorticity_and_divergence(model='reanalysis', scene='TESTING', level=None):
    """Vorticity and divergence"""

    # Read data
    if model == 'reanalysis':
        if level == 'sl':
            aux = geostrophic(model=model, scene=scene)
            times = aux['times']
            u = aux['data']['ugsl']
            v = aux['data']['vgsl']
        else:
            aux = read.one_direct_predictor('ua', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            u = aux['data']
            v = read.one_direct_predictor('va', level=level, grid='ext', model=model, scene=scene)['data']
    else:
        if level == 'sl':
            datesDefined = False
            for ipred in range(len(all_preds.keys())):
                try:
                    ncName = list(all_preds.keys())[ipred]
                    dates = read.one_direct_predictor(ncName, level=None, grid='ext', model=model, scene=scene)['times']
                    datesDefined = True
                    continue
                except:
                    pass
            aux = geostrophic(model=model, scene=scene)
            u, v = aux['data']['ugsl'], aux['data']['vgsl']
        else:
            aux = read.one_direct_predictor('ua', level=level, grid='ext', model=model, scene=scene)
            times = aux['times']
            u = aux['data']
            v = read.one_direct_predictor('va', level=level, grid='ext', model=model, scene=scene)['data']

    # Calculate wind gradients
    ndates = len(times)
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

    return {'data': {'vort': vort, 'div': div}, 'times': times}


########################################################################################################################
def psl_trend(model='reanalysis', scene='TESTING'):
    """psl trend from previous day"""

    # Read data
    if model == 'reanalysis':
        aux = read.one_direct_predictor('psl', level=None, grid='ext', model=model, scene=scene)
        times = aux['times']
        psl = aux['data']
    else:
        aux = read.one_direct_predictor('psl', level=None, grid='ext', model=model, scene=scene)
        times = aux['times']
        psl = aux['data']

    # Calculate psl_trend
    psl_dayBefore = np.copy(psl)
    psl_dayBefore[1:][:][:] = psl[:-1][:][:]
    psl_trend = psl - psl_dayBefore

    return {'data': psl_trend, 'times': times}


########################################################################################################################
def insolation(model='reanalysis', scene='TESTING'):
    """Theoretical insolation as sin function (between 0 and 1)"""
    pi = 3.14

    # Read data
    # if model == 'reanalysis':
    #     dates = calibration_dates
    # else:
    datesDefined = False
    for ipred in range(len(all_preds.keys())):
        try:
            ncName = list(all_preds.keys())[ipred]
            times = read.one_direct_predictor(ncName, level=None, grid='ext', model=model, scene=scene)['times']
            datesDefined = True
            continue
        except:
            pass
    if datesDefined == False:
        print('At least one direct predictors is needed, not only derived predictors.')
        exit()

    # Calculate ins
    ins = []
    for date in times:
        ndays = datetime.date(date.year, 12, 31).timetuple().tm_yday
        equinox = datetime.date(date.year, 3, 21).timetuple().tm_yday
        iday = date.timetuple().tm_yday
        ins.append(math.sin(2 * pi * (iday - equinox) / float(ndays)))

    ins = np.asarray(ins)
    ins = ins[:, np.newaxis, np.newaxis]
    ins = np.repeat(ins, ext_nlats, axis=1)
    ins = np.repeat(ins, ext_nlons, axis=2)

    return {'data': ins, 'times': times}


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
        aux = read.one_direct_predictor('ta', level=level, grid='ext', model=model, scene=scene)
        times = aux['times']
        t1000 = aux['data']
        psl = read.one_direct_predictor('psl', level=None, grid='ext', model=model, scene=scene)['data']
        tsl = t1000 / (100000 / psl) ** (R * alpha / g)
        denssl = psl / (R * tsl)
    else:
        aux = read.one_direct_predictor('psl', level=None, grid='ext', model=model, scene=scene)
        times = aux['times']
        psl = aux['data']
        denssl = 1.225  # Density

    ndates = len(times)

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
    grad = np.gradient(psl)
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

    return {'data': {'ugsl': ugsl, 'vgsl': vgsl}, 'times': times}

