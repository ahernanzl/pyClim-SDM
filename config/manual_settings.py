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

# ---------------- TMAX --------------------------------
# methods.append({'var': 'tmax', 'methodName': 'RAW',     'family': 'RAW',    'mode': 'RAW',  'fields': 'var'})
methods.append({'var': 'tmax', 'methodName': 'QM',      'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmax', 'methodName': 'DQM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmax', 'methodName': 'QDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmax', 'methodName': 'PSDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmax', 'methodName': 'ANA-MLR', 'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'tmax', 'methodName': 'WT-MLR',  'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
methods.append({'var': 'tmax', 'methodName': 'MLR',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmax', 'methodName': 'ANN',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmax', 'methodName': 'SVM',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmax', 'methodName': 'LS-SVM',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmax', 'methodName': 'WG-PDF',  'family': 'WG',     'mode': 'PP',   'fields': 'var'})

# # ---------------- TMIN --------------------------------
# methods.append({'var': 'tmin', 'methodName': 'RAW',     'family': 'RAW',    'mode': 'RAW',  'fields': 'var'})
# methods.append({'var': 'tmin', 'methodName': 'QM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmin', 'methodName': 'DQM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmin', 'methodName': 'QDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'tmin', 'methodName': 'PSDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
methods.append({'var': 'tmin', 'methodName': 'ANA-MLR', 'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'tmin', 'methodName': 'WT-MLR',  'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'tmin', 'methodName': 'MLR',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmin', 'methodName': 'ANN',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmin', 'methodName': 'SVM',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'tmin', 'methodName': 'LS-SVM',     'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
methods.append({'var': 'tmin', 'methodName': 'WG-PDF',  'family': 'WG',     'mode': 'PP',   'fields': 'var'})

# ---------------- PCP --------------------------------
# methods.append({'var': 'pcp', 'methodName': 'RAW',          'family': 'RAW',    'mode': 'RAW',  'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'QM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'DQM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'QDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
methods.append({'var': 'pcp', 'methodName': 'PSDM',     'family': 'BC',     'mode': 'MOS',  'fields': 'var'})
methods.append({'var': 'pcp', 'methodName': 'ANA-SYN-1NN',    'family': 'ANA',    'mode': 'PP',   'fields': 'saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-SYN-kNN',    'family': 'ANA',    'mode': 'PP',   'fields': 'saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-SYN-rand',  'family': 'ANA',    'mode': 'PP',   'fields': 'saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-LOC-1NN',    'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-LOC-kNN',    'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-LOC-rand',  'family': 'ANA',    'mode': 'PP',   'fields': 'pred+saf'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-PCP-1NN',    'family': 'ANA',    'mode': 'PP',   'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-PCP-kNN',    'family': 'ANA',    'mode': 'PP',   'fields': 'var'})
# methods.append({'var': 'pcp', 'methodName': 'ANA-PCP-rand',  'family': 'ANA',    'mode': 'PP',   'fields': 'var'})
methods.append({'var': 'pcp', 'methodName': 'GLM-LIN',      'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'GLM-EXP',      'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'GLM-CUB',      'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'ANN',          'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'SVM',          'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
# methods.append({'var': 'pcp', 'methodName': 'LS-SVM',          'family': 'TF',     'mode': 'PP',   'fields': 'pred'})
methods.append({'var': 'pcp', 'methodName': 'WG-NMM',       'family': 'WG',     'mode': 'PP',   'fields': 'var'})
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
# split_mode = 'all_testing'
split_mode = 'single_split'
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
biasCorr_years = (1979, 2005)
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

reaNames = {'u': 'u', 'v': 'v', 't': 't', 'z': 'z', 'q': 'q', 'mslp': 'msl', 'u10': 'u10', 'v10': 'v10',
            't2m': 't2m', 'tmax': 'mx2t', 'tmin': 'mn2t', 'pcp': 'tp'}

modNames = {'u': 'ua', 'v': 'va', 't': 'ta', 'z': 'zg', 'q': 'hus', 'mslp': 'psl', 'u10': 'uas', 'v10': 'vas',
            't2m': 'tas', 'tmax': 'tasmax', 'tmin': 'tasmin', 'pcp': 'pr'}


preds_t_list = [
    # 'tmax',             # maximum daily temperature
    # 'tmin',             # minimum daily temperature
    # 'pcp',              # daily predicitation
    'mslp',             # mean sea level pressure
    # 'mslp_trend',       # mean sea level trend from last day (derivedFrom mslp)
    # 'ins',              # theoretical insolation (derived from dates)
    # 'u10',              # surface wind
    # 'v10',              # surface wind
    't2m',              # surface temperature
    # 'u1000',            # wind at pressure levels
    'u850',             # wind at pressure levels
    # 'u700',             # wind at pressure levels
    'u500',             # wind at pressure levels
    # 'u250',             # wind at pressure levels
    # 'v1000',            # wind at pressure levels
    'v850',             # wind at pressure levels
    # 'v700',             # wind at pressure levels
    'v500',             # wind at pressure levels
    # 'v250',             # wind at pressure levels
    # 't1000',            # temperature at pressure levels
    't850',             # temperature at pressure levels
    # 't700',             # temperature at pressure levels
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
]


preds_p_list = [
    # 'tmax',             # maximum daily temperature
    # 'tmin',             # minimum daily temperature
    # 'pcp',              # daily predicitation
    'mslp',             # mean sea level pressure
    # 'mslp_trend',       # mean sea level trend from last day (derivedFrom mslp)
    # 'ins',              # theoretical insolation (derived from dates)
    # 'u10',              # surface wind
    # 'v10',              # surface wind
    # 't2m',              # surface temperature
    # 'u1000',            # wind at pressure levels
    'u850',             # wind at pressure levels
    # 'u700',             # wind at pressure levels
    'u500',             # wind at pressure levels
    # 'u250',             # wind at pressure levels
    # 'v1000',            # wind at pressure levels
    'v850',             # wind at pressure levels
    # 'v700',             # wind at pressure levels
    'v500',             # wind at pressure levels
    # 'v250',             # wind at pressure levels
    # 't1000',            # temperature at pressure levels
    # 't850',             # temperature at pressure levels
    # 't700',             # temperature at pressure levels
    # 't500',             # temperature at pressure levels
    # 't250',             # temperature at pressure levels
    # 'z1000',            # geopotential at pressure levels
    'z850',             # geopotential at pressure levels
    # 'z700',             # geopotential at pressure levels
    'z500',             # geopotential at pressure levels
    # 'z250',             # geopotential at pressure levels
    # 'q1000',            # specifit humidity at pressure levels
    # 'q850',             # specifit humidity at pressure levels
    # 'q700',             # specifit humidity at pressure levels
    # 'q500',             # specifit humidity at pressure levels
    # 'q250',             # specifit humidity at pressure levels
    # 'r1000',            # relative humidity at pressure levels (derived from t and q)
    'r850',             # relative humidity at pressure levels (derived from t and q)
    # 'r700',             # relative humidity at pressure levels (derived from t and q)
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
]

###############################  SYNOPTIC ANALOGY FIELDS  ##############################################################

# Define synoptic analogy fields (saf)
saf_list = [
    # 'tmax',             # maximum daily temperature
    # 'tmin',             # minimum daily temperature
    # 'pcp',              # daily predicitation
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
    'ACCESS-CM2',
    # 'ACCESS-ESM1-5',
    # 'AWI-CM-1-1-MR',
    # 'BCC-CSM2-MR',
    # 'CAMS-CSM1-0',
    'CanESM5',
    # 'CESM2',
    # 'CESM2-WACCM',
    # 'CMCC-CM2-SR5',
    # 'CMCC-ESM2',
    # 'CNRM-CM6-1',
    # 'CNRM-CM6-1-HR',
    # 'CNRM-ESM2-1',
    'EC-Earth3',
    # 'EC-Earth3-AerChem',
    # 'EC-Earth3-CC',
    # 'EC-Earth3-Veg',
    # 'EC-Earth3-Veg-LR',
    # 'FGOALS-g3',
    # 'GFDL-CM4',
    # 'GFDL-ESM4',
    # 'GISS-E2-1-G',
    # 'HadGEM3-GC31-LL',
    # 'HadGEM3-GC31-MM',
    # 'IITM-ESM',
    'INM-CM4-8',
    'INM-CM5-0',
    # 'IPSL-CM5A2-INCA',
    'IPSL-CM6A-LR',
    # 'KACE-1-0-G',
    # 'KIOST-ESM',
    # 'MIROC-ES2L',
    'MIROC6',
    # 'MPI-ESM-1-2-HAM',
    'MPI-ESM1-2-HR',
    'MPI-ESM1-2-LR',
    'MRI-ESM2-0',
    # 'NESM3',
    # 'NorESM2-LM',
    # 'NorESM2-MM',
    # 'TaiESM1',
    # 'UKESM1-0-LL',
)


modelRealizationFilename = 'r1i1p1f1'


###################################     CLIMDEX    #####################################################################

# Define climdex to be used (see https://www.climdex.org/learn/indices/)
climdex_names = {
    'tmax': (
        'TXm',
        # 'TX90p',
        # 'TX10p',
        # 'TXx',
        # 'TXn',
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
        # 'TNx',
        # 'TNn',
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
        # 'SDII',
        # 'Rx1day',
        # 'Rx5day',
        # 'R10mm',
        # 'R20mm',
        # 'CDD',
        # 'p95',
        # 'R95p',
        # 'R95pFRAC',
        # 'p99',
        # 'R99p',
        # 'R99pFRAC',
        # 'CWD',
    )}


###################################     Bias correction projections    #################################################
bc_method = None
# bc_method = 'QM'
# bc_method = 'DQM'
# bc_method