showWelcomeMessage = True # For graphical mode


########################################       EXPERIMENT      #########################################################
# Select your current experiment type
# experiment = 'PRECONTROL'
experiment = 'EVALUATION'
# experiment = 'PROJECTIONS'

####################################################################################################################
#                                        METHODS                                                                   #
####################################################################################################################
methods = []

# # ---------------- TMAX --------------------------------
# methods.append({'var': 'tmax', 'methodName': 'RAW',     'family': 'RAW',    'mode': 'RAW',  'fields': 'var'})
# methods.append({'var': 'tmax', 'methodName': 'RAW-BIL',     'family': 'RAW',    'mode': 'RAW',  'fields': 'var'})
# methods.append({'var': 'tmax', 'methodName': 'QM',      'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmax', 'methodName': 'DQM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmax', 'methodName': 'QDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmax', 'methodName': 'PSDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmax', 'methodName': 'ANA-MLR', 'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'tmax', 'methodName': 'WT-MLR',  'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'tmax', 'methodName': 'MLR',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmax', 'methodName': 'SVM',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmax', 'methodName': 'LS-SVM',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmax', 'methodName': 'RF',          'family': 'TF',     'mode': 'PP',   'fields': 'pred+var'})
# methods.append({'var': 'tmax', 'methodName': 'XGB',          'family': 'TF',     'mode': 'PP',   'fields': 'pred+var'})
# methods.append({'var': 'tmax', 'methodName': 'ANN',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmax', 'methodName': 'CNN',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmax', 'methodName': 'WG-PDF',  'family': 'WG',     'mode': 'PP',   'fields': 'var'})
#
# # ---------------- TMIN --------------------------------
# methods.append({'var': 'tmin', 'methodName': 'RAW',     'family': 'RAW',    'mode': 'RAW',  'fields': 'var'})
# methods.append({'var': 'tmin', 'methodName': 'RAW-BIL',     'family': 'RAW',    'mode': 'RAW',  'fields': 'var'})
# methods.append({'var': 'tmin', 'methodName': 'QM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmin', 'methodName': 'DQM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmin', 'methodName': 'QDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmin', 'methodName': 'PSDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmin', 'methodName': 'ANA-MLR', 'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'tmin', 'methodName': 'WT-MLR',  'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'tmin', 'methodName': 'MLR',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmin', 'methodName': 'SVM',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmin', 'methodName': 'LS-SVM',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmin', 'methodName': 'RF',          'family': 'TF',     'mode': 'PP',   'fields': 'pred+var'})
# methods.append({'var': 'tmin', 'methodName': 'XGB',          'family': 'TF',     'mode': 'PP',   'fields': 'pred+var'})
# methods.append({'var': 'tmin', 'methodName': 'ANN',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmin', 'methodName': 'CNN',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmin', 'methodName': 'WG-PDF',  'family': 'WG',     'mode': 'PP',   'fields': 'var'})
#
# # # ---------------- PCP --------------------------------
# methods.append({'var': 'pcp', 'methodName': 'RAW',          'family': 'RAW',    'mode': 'RAW',  'fields': 'var'})
methods.append({'var': 'pcp', 'methodName': 'RAW-BIL',     'family': 'RAW',    'mode': 'RAW',  'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'QM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'DQM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'QDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
methods.append({'var': 'pcp', 'methodName': 'PSDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
methods.append({'var': 'pcp', 'methodName': 'PSDM-s',     'family': 'RAW',    'mode': 'RAW',  'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-SYN-1NN',    'family': 'ANA',    'mode': 'PP',   'fields': 'saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-SYN-kNN',    'family': 'ANA',    'mode': 'PP',   'fields': 'saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-SYN-rand',  'family': 'ANA',    'mode': 'PP',   'fields': 'saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-LOC-1NN',    'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-LOC-kNN',    'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-LOC-rand',  'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-VAR-1NN',    'family': 'ANA',    'mode': 'PP',   'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-VAR-kNN',    'family': 'ANA',    'mode': 'PP',   'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-VAR-rand',  'family': 'ANA',    'mode': 'PP',   'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'GLM-LIN',      'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'GLM-EXP',      'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'GLM-CUB',      'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'SVM',          'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'LS-SVM',          'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'RF',          'family': 'TF',     'mode': 'PP',   'fields': 'pred+var'})
methods.append({'var': 'pcp', 'methodName': 'XGB',          'family': 'TF',     'mode': 'PP',   'fields': 'pred+var'})
methods.append({'var': 'pcp', 'methodName': 'XGB+PSDM',          'family': 'TF',     'mode': 'PP',   'fields': 'pred+var'})
methods.append({'var': 'pcp', 'methodName': 'XGB+PSDM-s',          'family': 'TF',     'mode': 'PP',   'fields': 'pred+var'})
# methods.append({'var': 'pcp', 'methodName': 'ANN',          'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'CNN',          'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'CNN-SYN',          'family': 'TF',     'mode': 'PP',   'fields': 'saf'})
# methods.append({'var': 'pcp', 'methodName': 'WG-NMM',       'family': 'WG',     'mode': 'PP',   'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'WG-PDF',       'family': 'WG',     'mode': 'PP',   'fields': 'var'})


########################################       DATES      ##############################################################
# calibration_years corresponds to the longest period available by reanalysis and hres_data, which then can be split for
# training and testing
calibration_years = (1979, 2020)

single_split_testing_years = (2006, 2020)
fold1_testing_years = (1979, 1987)
fold2_testing_years = (1988, 1996)
fold3_testing_years = (1997, 2005)
fold4_testing_years = (2006, 2014)
fold5_testing_years = (2015, 2020)

# Activate one of the following training/testing split options
# split_mode  = 'all_training'
split_mode = 'all_testing'
# split_mode = 'single_split'
# split_mode = 'fold1'
# split_mode = 'fold2'
# split_mode = 'fold3'
# split_mode = 'fold4'
# split_mode = 'fold5' # This last fold will automatically join the 5 folds

# Reference: for standardization and future signal of change. The choice of the reference period is constrained by
# availability of reanalysis, historical GCMs and hres data.
reference_years = (1979, 2005)
# historical_years = (1950, 2014)
historical_years = (1979, 2005)
ssp_years = (2015, 2100)
# biasCorr_years = (1951, 2014)

hresPeriodFilename = {}
hresPeriodFilename.update({'t': '19510101-20201231'})
hresPeriodFilename.update({'p': '19510101-20201231'})
reanalysisName = 'ERA5'
reanalysisPeriodFilename = '19790101-20201231'
historicalPeriodFilename = '19500101-20141231'
sspPeriodFilename = '20150101-21001231'

#############################################  GRIDS  ##################################################################
grid_res = 1.5

# saf grid (for synoptic analogy). All files (reanalysis and models) need to contain at least this region plus one grid box border.
saf_lat_up, saf_lat_down =  49.0, 29.5
saf_lon_left, saf_lon_right = -18.0, 12.0


###########################################   PREDICTORS  ##############################################################
# IMPORTANT: do not change preds order. Otherwise, lib/read.lres_data would need to be adapted to follow the same order
# Define, for each predictor variable, reaName and modName (variable names for reanalysis and for CMIP models)

reaNames = {'u': 'u', 'v': 'v', 't': 't', 'z': 'z', 'q': 'q', 'r': 'r',
            'mslp': 'msl', 'u10': 'u10', 'v10': 'v10', 't2m': 't2m',
            'd2m': 'd2m', 'r2m': '-', 'q2m': '-', 'sp': 'sp', 'tcc': 'tcc',
            'tmax': 'mx2t', 'tmin': 'mn2t', 'pcp': 'tp'}

modNames = {'u': 'ua', 'v': 'va', 't': 'ta', 'z': 'zg', 'q': 'hus', 'r': 'hur',
            'mslp': 'psl', 'u10': 'uas', 'v10': 'vas', 't2m': 'tas',
            'd2m': 'tdps', 'r2m': 'hurs', 'q2m': 'huss', 'sp': 'ps', 'tcc': 'clt',
            'tmax': 'tasmax', 'tmin': 'tasmin', 'pcp': 'pr'}


preds_t_list = [
    # 'tmax',             # maximum daily temperature
    # 'tmin',             # minimum daily temperature
    # 'pcp',              # daily precipitation
    'mslp',             # mean sea level pressure
    # 'mslp_trend',       # mean sea level trend from last day (derivedFrom mslp)
    # 'ins',              # theoretical insolation (derived from dates)
    # 'u10',              # surface wind
    # 'v10',              # surface wind
    't2m',              # surface temperature
    # 'u1000',            # wind at pressure levels
    'u850',             # wind at pressure levels
    'u700',             # wind at pressure levels
    'u500',             # wind at pressure levels
    # 'u250',             # wind at pressure levels
    # 'v1000',            # wind at pressure levels
    'v850',             # wind at pressure levels
    'v700',             # wind at pressure levels
    'v500',             # wind at pressure levels
    # 'v250',             # wind at pressure levels
    # 't1000',            # temperature at pressure levels
    't850',             # temperature at pressure levels
    't700',             # temperature at pressure levels
    't500',             # temperature at pressure levels
    # 't250',             # temperature at pressure levels
    # 'z1000',            # geopotential at pressure levels
    # 'z850',             # geopotential at pressure levels
    # 'z700',             # geopotential at pressure levels
    # 'z500',             # geopotential at pressure levels
    # 'z250',             # geopotential at pressure levels
    # 'q1000',            # specifit humidity at pressure levels
    # 'q850',             # specifit humidity at pressure levels
    # 'q700',             # specifit humidity at pressure levels
    # 'q500',             # specifit humidity at pressure levels
    # 'q250',             # specifit humidity at pressure levels
    # 'r1000',            # relative humidity at pressure levels (derived from t and q)
    # 'r850',             # relative humidity at pressure levels (derived from t and q)
    # 'r700',             # relative humidity at pressure levels (derived from t and q)
    # 'r500',             # relative humidity at pressure levels (derived from t and q)
    # 'r250',             # relative humidity at pressure levels (derived from t and q)
    # 'td1000',           # dew point at pressure levels (derived from q)
    # 'td850',            # dew point at pressure levels (derived from q)
    # 'td700',            # dew point at pressure levels (derived from q)
    # 'td500',            # dew point at pressure levels (derived from q)
    # 'td250',            # dew point at pressure levels (derived from q)
    # 'Dtd1000',          # dew point depression at pressure levels (derived from t and q)
    # 'Dtd850',           # dew point depression at pressure levels (derived from t and q)
    # 'Dtd700',           # dew point depression at pressure levels (derived from t and q)
    # 'Dtd500',           # dew point depression at pressure levels (derived from t and q)
    # 'Dtd250',           # dew point depression at pressure levels (derived from t and q)
    # 'vort1000',         # vorticity at pressure levels (derived from u and v)
    # 'vort850',          # vorticity at pressure levels (derived from u and v)
    # 'vort700',          # vorticity at pressure levels (derived from u and v)
    # 'vort500',          # vorticity at pressure levels (derived from u and v)
    # 'vort250',          # vorticity at pressure levels (derived from u and v)
    # 'div1000',          # divergence at pressure levels (derived from u and v)
    # 'div850',           # divergence at pressure levels (derived from u and v)
    # 'div700',           # divergence at pressure levels (derived from u and v)
    # 'div500',           # divergence at pressure levels (derived from u and v)
    # 'div250',           # divergence at pressure levels (derived from u and v)
    # 'vtg_1000_850',     # vertical thermal gradient between 1000 and 850 hPa (derived from t)
    # 'vtg_850_700',      # vertical thermal gradient between 850 and 700 hPa (derived from t)
    # 'vtg_700_500',      # vertical thermal gradient between 700 and 500 hPa (derived from t)
    # 'ugsl',             # geostrophic wind at sea level (derived from t and mslp)
    # 'vgsl',             # geostrophic wind at sea level (derived from t and mslp)
    # 'vortgsl',          # vorticity of geostrophic wind at sea level (derived from t and mslp)
    # 'divgsl',           # divergence of geostrophic wind at sea level (derived from t and mslp)
    # 'K_index',          # K instability index
    # 'TT_index',         # Total Totals instability index
    # 'SSI_index',        # Showalter index
    # 'LI_index',         # Lifted index
]


preds_p_list = [
    # 'tmax',             # maximum daily temperature
    # 'tmin',             # minimum daily temperature
    # 'pcp',              # daily precipitation
    'mslp',             # mean sea level pressure
    # 'mslp_trend',       # mean sea level trend from last day (derivedFrom mslp)
    # 'ins',              # theoretical insolation (derived from dates)
    # 'u10',              # surface wind
    # 'v10',              # surface wind
    # 't2m',              # surface temperature
    # 'u1000',            # wind at pressure levels
    'u850',             # wind at pressure levels
    'u700',             # wind at pressure levels
    'u500',             # wind at pressure levels
    # 'u250',             # wind at pressure levels
    # 'v1000',            # wind at pressure levels
    'v850',             # wind at pressure levels
    'v700',             # wind at pressure levels
    'v500',             # wind at pressure levels
    # 'v250',             # wind at pressure levels
    # 't1000',            # temperature at pressure levels
    # 't850',             # temperature at pressure levels
    # 't700',             # temperature at pressure levels
    # 't500',             # temperature at pressure levels
    # 't250',             # temperature at pressure levels
    # 'z1000',            # geopotential at pressure levels
    'z850',             # geopotential at pressure levels
    'z700',             # geopotential at pressure levels
    'z500',             # geopotential at pressure levels
    # 'z250',             # geopotential at pressure levels
    # 'q1000',            # specifit humidity at pressure levels
    # 'q850',             # specifit humidity at pressure levels
    # 'q700',             # specifit humidity at pressure levels
    # 'q500',             # specifit humidity at pressure levels
    # 'q250',             # specifit humidity at pressure levels
    # 'r1000',            # relative humidity at pressure levels (derived from t and q)
    'r850',             # relative humidity at pressure levels (derived from t and q)
    'r700',             # relative humidity at pressure levels (derived from t and q)
    'r500',             # relative humidity at pressure levels (derived from t and q)
    # 'r250',             # relative humidity at pressure levels (derived from t and q)
    # 'td1000',           # dew point at pressure levels (derived from q)
    # 'td850',            # dew point at pressure levels (derived from q)
    # 'td700',            # dew point at pressure levels (derived from q)
    # 'td500',            # dew point at pressure levels (derived from q)
    # 'td250',            # dew point at pressure levels (derived from q)
    # 'Dtd1000',          # dew point depression at pressure levels (derived from t and q)
    # 'Dtd850',           # dew point depression at pressure levels (derived from t and q)
    # 'Dtd700',           # dew point depression at pressure levels (derived from t and q)
    # 'Dtd500',           # dew point depression at pressure levels (derived from t and q)
    # 'Dtd250',           # dew point depression at pressure levels (derived from t and q)
    # 'vort1000',         # vorticity at pressure levels (derived from u and v)
    # 'vort850',          # vorticity at pressure levels (derived from u and v)
    # 'vort700',          # vorticity at pressure levels (derived from u and v)
    # 'vort500',          # vorticity at pressure levels (derived from u and v)
    # 'vort250',          # vorticity at pressure levels (derived from u and v)
    # 'div1000',          # divergence at pressure levels (derived from u and v)
    # 'div850',           # divergence at pressure levels (derived from u and v)
    # 'div700',           # divergence at pressure levels (derived from u and v)
    # 'div500',           # divergence at pressure levels (derived from u and v)
    # 'div250',           # divergence at pressure levels (derived from u and v)
    # 'vtg_1000_850',     # vertical thermal gradient between 1000 and 850 hPa (derived from t)
    # 'vtg_850_700',      # vertical thermal gradient between 850 and 700 hPa (derived from t)
    # 'vtg_700_500',      # vertical thermal gradient between 700 and 500 hPa (derived from t)
    # 'ugsl',             # geostrophic wind at sea level (derived from t and mslp)
    # 'vgsl',             # geostrophic wind at sea level (derived from t and mslp)
    # 'vortgsl',          # vorticity of geostrophic wind at sea level (derived from t and mslp)
    # 'divgsl',           # divergence of geostrophic wind at sea level (derived from t and mslp)
    # 'K_index',          # K instability index
    # 'TT_index',         # Total Totals instability index
    # 'SSI_index',        # Showalter index
    # 'LI_index',         # Lifted index
]

###############################  SYNOPTIC ANALOGY FIELDS  ##############################################################

# Define synoptic analogy fields (saf)
saf_list = [
    # 'tmax',             # maximum daily temperature
    # 'tmin',             # minimum daily temperature
    # 'pcp',              # daily precipitation
    # 'mslp',             # mean sea level pressure
    # 'mslp_trend',       # mean sea level trend from last day (derivedFrom mslp)
    # 'ins',              # theoretical insolation (derived from dates)
    # 'u10',              # surface wind
    # 'v10',              # surface wind
    # 't2m',              # surface temperature
    # 'u1000',            # wind at pressure levels
    # 'u850',             # wind at pressure levels
    # 'u700',             # wind at pressure levels
    # 'u500',             # wind at pressure levels
    # 'u250',             # wind at pressure levels
    # 'v1000',            # wind at pressure levels
    # 'v850',             # wind at pressure levels
    # 'v700',             # wind at pressure levels
    # 'v500',             # wind at pressure levels
    # 'v250',             # wind at pressure levels
    # 't1000',            # temperature at pressure levels
    # 't850',             # temperature at pressure levels
    # 't700',             # temperature at pressure levels
    # 't500',             # temperature at pressure levels
    # 't250',             # temperature at pressure levels
    # 'z1000',            # geopotential at pressure levels
    # 'z850',             # geopotential at pressure levels
    # 'z700',             # geopotential at pressure levels
    'z500',             # geopotential at pressure levels
    # 'z250',             # geopotential at pressure levels
    # 'q1000',            # specifit humidity at pressure levels
    # 'q850',             # specifit humidity at pressure levels
    # 'q700',             # specifit humidity at pressure levels
    # 'q500',             # specifit humidity at pressure levels
    # 'q250',             # specifit humidity at pressure levels
    # 'r1000',            # relative humidity at pressure levels (derived from t and q)
    # 'r850',             # relative humidity at pressure levels (derived from t and q)
    # 'r700',             # relative humidity at pressure levels (derived from t and q)
    # 'r500',             # relative humidity at pressure levels (derived from t and q)
    # 'r250',             # relative humidity at pressure levels (derived from t and q)
    # 'td1000',           # dew point at pressure levels (derived from q)
    # 'td850',            # dew point at pressure levels (derived from q)
    # 'td700',            # dew point at pressure levels (derived from q)
    # 'td500',            # dew point at pressure levels (derived from q)
    # 'td250',            # dew point at pressure levels (derived from q)
    # 'Dtd1000',          # dew point depression at pressure levels (derived from t and q)
    # 'Dtd850',           # dew point depression at pressure levels (derived from t and q)
    # 'Dtd700',           # dew point depression at pressure levels (derived from t and q)
    # 'Dtd500',           # dew point depression at pressure levels (derived from t and q)
    # 'Dtd250',           # dew point depression at pressure levels (derived from t and q)
    # 'vort1000',         # vorticity at pressure levels (derived from u and v)
    # 'vort850',          # vorticity at pressure levels (derived from u and v)
    # 'vort700',          # vorticity at pressure levels (derived from u and v)
    # 'vort500',          # vorticity at pressure levels (derived from u and v)
    # 'vort250',          # vorticity at pressure levels (derived from u and v)
    # 'div1000',          # divergence at pressure levels (derived from u and v)
    # 'div850',           # divergence at pressure levels (derived from u and v)
    # 'div700',           # divergence at pressure levels (derived from u and v)
    # 'div500',           # divergence at pressure levels (derived from u and v)
    # 'div250',           # divergence at pressure levels (derived from u and v)
    # 'vtg_1000_850',     # vertical thermal gradient between 1000 and 850 hPa (derived from t)
    # 'vtg_850_700',      # vertical thermal gradient between 850 and 700 hPa (derived from t)
    # 'vtg_700_500',      # vertical thermal gradient between 700 and 500 hPa (derived from t)
    # 'ugsl',             # geostrophic wind at sea level (derived from t and mslp)
    # 'vgsl',             # geostrophic wind at sea level (derived from t and mslp)
    # 'vortgsl',          # vorticity of geostrophic wind at sea level (derived from t and mslp)
    # 'divgsl',           # divergence of geostrophic wind at sea level (derived from t and mslp)
    # 'K_index',          # K instability index
    # 'TT_index',         # Total Totals instability index
    # 'SSI_index',        # Showalter index
    # 'LI_index',         # Lifted index
]



###################################     SCENES AND MODELS    ###########################################################

# Define scenes and models. scene names for titles at figures. In filenames they are historical, ssp126, etc.
scene_names_list = [
    'HISTORICAL',
    # 'SSP1-1.9',
    # 'SSP1-2.6',
    # 'SSP2-4.5',
    # 'SSP3-7.0',
    'SSP5-8.5',
]

model_names_list = (
    'ACCESS-CM2_r1i1p1f1',
    'CanESM5_r1i1p1f1',
    'EC-Earth3_r1i1p1f1',
    'INM-CM4-8_r1i1p1f1',
    'INM-CM5-0_r1i1p1f1',
    'IPSL-CM6A-LR_r1i1p1f1',
    'MIROC6_r1i1p1f1',
    'MPI-ESM1-2-HR_r1i1p1f1',
    'MPI-ESM1-2-LR_r1i1p1f1',
    'MRI-ESM2-0_r1i1p1f1',
)


# model_names_list = (
#     'ACCESS-CM2_r1i1p1f1',
#     'ACCESS-ESM1-5_r1i1p1f1',
#     # 'AWI-CM-1-1-MR_r1i1p1f1',
#     # 'AWI-CM-1-1-LR_r1i1p1f1',
#     # 'BCC-CSM2-MR_r1i1p1f1',
#     # 'BCC-ESM1_r1i1p1f1',
#     # 'CAMS-CSM1-0_r1i1p1f1',
#     'CanESM5_r1i1p1f1',
#     # 'CanESM5-CanOE_r1i1p1f1',
#     # 'CESM2_r1i1p1f1',
#     # 'CESM2-FV2_r1i1p1f1',
#     # 'CESM2-WACCM_r1i1p1f1',
#     # 'CESM2-WACCM-FV2_r1i1p1f1',
#     # 'CIESM_r1i1p1f1',
#     # 'CMCC-CM2-HR4_r1i1p1f1',
#     # 'CMCC-CM2-SR5_r1i1p1f1',
#     # 'CMCC-ESM2_r1i1p1f1',
#     # 'CNRM-CM6-1_r1i1p1f1',
#     # 'CNRM-CM6-1-HR_r1i1p1f1',
#     'CMCC-ESM2_r1i1p1f1',
#     # 'E3SM-1-0_r1i1p1f1',
#     # 'E3SM-1-1_r1i1p1f1',
#     # 'E3SM-1-1-ECA_r1i1p1f1',
#     'EC-Earth3_r1i1p1f1',
#     # 'EC-Earth3-AerChem_r1i1p1f1',
#     # 'EC-Earth3-CC_r1i1p1f1',
#     'EC-Earth3-Veg_r1i1p1f1',
#     'EC-Earth3-Veg-LR_r1i1p1f1',
#     # 'FGOALS-f3-L_r1i1p1f1',
#     'FGOALS-g3_r1i1p1f1',
#     # 'FIO-ESM-2-0_r1i1p1f1',
#     # 'GFDL-CM4_r1i1p1f1',
#     'GFDL-ESM4_r1i1p1f1',
#     # 'GISS-E2-1-G_r1i1p1f1',
#     # 'GISS-E2-1-H_r1i1p1f1',
#     # 'HadGEM3-GC31-LL_r1i1p1f1',
#     # 'HadGEM3-GC31-MM_r1i1p1f1',
#     # 'IITM-ESM_r1i1p1f1',
#     'INM-CM4-8_r1i1p1f1',
#     'INM-CM5-0_r1i1p1f1',
#     # 'IPSL-CM5A2-INCA_r1i1p1f1',
#     'IPSL-CM6A-LR_r1i1p1f1',
#     'KACE-1-0-G_r1i1p1f1',
#     # 'KIOST-ESM_r1i1p1f1',
#     # 'MCM-UA-1-0_r1i1p1f1',
#     # 'MIROC-ES2H_r1i1p1f1',
#     # 'MIROC-ES2L_r1i1p1f1',
#     'MIROC6_r1i1p1f1',
#     # 'MPI-ESM-1-2-HAM_r1i1p1f1',
#     'MPI-ESM1-2-HR_r1i1p1f1',
#     'MPI-ESM1-2-LR_r1i1p1f1',
#     'MRI-ESM2-0_r1i1p1f1',
#     # 'NESM3_r1i1p1f1',
#     # 'NorCPM1_r1i1p1f1',
#     'NorESM2-LM_r1i1p1f1',
#     'NorESM2-MM_r1i1p1f1',
#     # 'SAM0-UNICON_r1i1p1f1',
#     'TaiESM1_r1i1p1f1',
#     # 'UKESM1-0-LL_r1i1p1f1',
#
#     'CanESM5_r1i1p2f1',
#     'CNRM-CM6-1_r1i1p1f2',
#     'CNRM-ESM2-1_r1i1p1f2',
#     'MIROC-ES2L_r1i1p1f2',
#     'UKESM1-0-LL_r1i1p1f2',
# )


###################################     CLIMDEX    #####################################################################

# Define climdex to be used (see https://www.climdex.org/learn/indices/)
climdex_names = {
    'tmax': (
        'TXm',
        # 'TX90p',
        # 'TX10p',
        'TXx',
        'TXn',
        # 'p99',
        # 'p95',
        # 'p5',
        # 'p1',
        # 'SU',
        # 'ID',
        # 'WSDI',
    ),
    'tmin': (
        'TNm',
        # 'TN90p',
        # 'TN10p',
        'TNx',
        'TNn',
        # 'p99',
        # 'p95',
        # 'p5',
        # 'p1',
        # 'FD',
        # 'TR',
        # 'CSDI',
    ),
    'pcp': (
        # 'Pm',
        'PRCPTOT',
        'R01',
        'SDII',
        # 'Rx1day',
        # 'Rx5day',
        # 'R10mm',
        # 'R20mm',
        # 'CDD',
        # 'p95',
        'R95p',
        # 'R95pFRAC',
        # 'p99',
        # 'R99p',
        # 'R99pFRAC',
        # 'CWD',
    )}


###################################     Bias correction   #################################################
apply_bc = False # Apply bias correction after downscaling
apply_bc_bySeason = False # Apply bias correction customized for each season after downcaling

