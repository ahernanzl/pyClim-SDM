showWelcomeMessage = True # For graphical mode


########################################       EXPERIMENT      #########################################################
# Select your current experiment type
# experiment = 'PRECONTROL'
experiment = 'EVALUATION'
# experiment = 'PROJECTIONS'

####################################################################################################################
#                                        targetVars                                                                   #
####################################################################################################################
# Activate/deactivate targetVars
targetVars = [
    'tasmax',
    'tasmin',
    # 'tas',
    'pr',
    # 'uas',
    # 'vas',
    # 'sfcWind',
    # 'hurs',
    # 'huss',
    # 'clt',
    # 'rsds',
    # 'rlds',
    # 'evspsbl',
    # 'evspsblpot',
    # 'psl',
    # 'ps',
    # 'mrro',
    # 'mrso',
    # 'myTargetVar',
]

methods = {
    'tasmax': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        # 'QDM',
        'PSDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
        'WG-PDF',
    ],
    'tasmin': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        # 'QDM',
        'PSDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
        'WG-PDF',
    ],
    'tas': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        # 'QDM',
        'PSDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
        'WG-PDF',
    ],
    'pr': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        # 'QDM',
        # 'PSDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'GLM-LIN',
        # 'GLM-EXP',
        # 'GLM-CUB',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
        # 'WG-PDF',
        # 'WG-NMM',
    ],
    'uas': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        # 'QDM',
        'PSDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
        'WG-PDF',
    ],
    'vas': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        # 'QDM',
        'PSDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
        'WG-PDF',
    ],
    'sfcWind': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'hurs': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        # 'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'huss': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'clt': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        # 'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'rsds': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'rlds': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'evspsbl': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'evspsblpot': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'psl': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'ps': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'mrro': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'mrso': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
    ],
    'myTargetVar': [
        'RAW',
        # 'RAW-BIL',
        # 'QM',
        # 'DQM',
        'QDM',
        # 'PSDM',
        'ANA-SYN-1NN',
        # 'ANA-SYN-kNN',
        # 'ANA-SYN-rand',
        # 'ANA-LOC-1NN',
        # 'ANA-LOC-kNN',
        # 'ANA-LOC-rand',
        # 'ANA-VAR-1NN',
        # 'ANA-VAR-kNN',
        # 'ANA-VAR-rand',
        'MLR',
        # 'MLR-ANA',
        # 'MLR-WT',
        # 'SVM',
        # 'LS-SVM',
        # 'RF',
        # 'XGB',
        # 'ANN',
        # 'CNN',
        # 'WG-PDF',
    ],
}



########################################       DATES      ##############################################################
# calibration_years corresponds to the longest period available by reanalysis and hres_data, which then can be split for
# training and testing
calibration_years = (1979, 2020)
single_split_testing_years = (2006, 2020)

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
reanalysisName = 'ERA5'

#############################################  GRIDS  ##################################################################
# saf grid (for synoptic analogy). All files (reanalysis and models) need to contain at least this region plus one grid box border.
saf_lat_up, saf_lat_down =  49.0, 29.5
saf_lon_left, saf_lon_right = -18.0, 12.0


###########################################   PREDICTORS  ##############################################################
# IMPORTANT: do not change preds order. Otherwise, lib/read.lres_data would need to be adapted to follow the same order
# Define, for each predictor variable, reaName and modName (variable names for reanalysis and for CMIP models)


reaNames = {'ua': 'u', 'va': 'v', 'ta': 't', 'zg': 'z', 'hus': 'q', 'hur': 'r', 'td': '-',
            'psl': 'msl', 'tdps': 'd2m', 'ps': 'sp',
            'tasmax': 'mx2t', 'tasmin': 'mn2t', 'tas': 't2m',
            'pr': 'tp', 'uas': 'u10', 'vas': 'v10', 'sfcWind': 'ws',
            'hurs': '-', 'huss': '-', 'clt': 'tcc',
            'rsds': 'issrd', 'rlds': 'istrd',
            'evspsbl': 'e', 'evspsblpot': '-',
            'mrro': 'ro', 'mrso': 'swvl1',
            'myTargetVar': 'fwi'}

modNames = {'ua': 'ua', 'va': 'va', 'ta': 'ta', 'zg': 'zg', 'hus': 'hus', 'hur': 'hur', 'td': '-',
            'psl': 'psl', 'tdps': 'tdps', 'ps': 'ps',
            'tasmax': 'tasmax', 'tasmin': 'tasmin', 'tas': 'tas',
            'pr': 'pr', 'uas': 'uas', 'vas': 'vas', 'sfcWind': 'sfcWind',
            'hurs': 'hurs', 'huss': 'huss', 'clt': 'clt',
            'rsds': 'rsds', 'rlds': 'rlds',
            'evspsbl': 'evspsbl', 'evspsblpot': 'evspsblpot',
            'mrro': 'mrro', 'mrso': 'mrso',
            'myTargetVar': 'fwi'}


preds_targetVars_dict = {
    'tasmax': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
        # 'K',          # K instability index
        # 'TT',         # Total Totals instability index
        # 'SSI',        # Showalter index
        # 'LI',         # Lifted index
    ],
    'tasmin': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        'tas',              # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        'ua850',             # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        'ua500',             # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        'va850',             # wind at pressure levels
        # 'va700',             # wind at pressure levels
        'va500',             # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        'ta850',             # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        'ta500',             # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
        # 'K',          # K instability index
        # 'TT',         # Total Totals instability index
        # 'SSI',        # Showalter index
        # 'LI',         # Lifted index
    ],
    'tas': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        'tas',              # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        'ua850',             # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        'ua500',             # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        'va850',             # wind at pressure levels
        # 'va700',             # wind at pressure levels
        'va500',             # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        'ta850',             # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        'ta500',             # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
        # 'K',          # K instability index
        # 'TT',         # Total Totals instability index
        # 'SSI',        # Showalter index
        # 'LI',         # Lifted index
    ],
    'pr': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',              # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        'ua850',             # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        'ua500',             # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        'va850',             # wind at pressure levels
        # 'va700',             # wind at pressure levels
        'va500',             # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',             # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',             # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
        'K',          # K instability index
        'TT',         # Total Totals instability index
        # 'SSI',        # Showalter index
        # 'LI',         # Lifted index
    ],
    'uas': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        'uas',              # surface wind
        'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',              # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        'ua850',             # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        'ua500',             # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        'va850',             # wind at pressure levels
        # 'va700',             # wind at pressure levels
        'va500',             # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        'ta850',             # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        'ta500',             # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
        # 'K',          # K instability index
        # 'TT_index',         # Total Totals instability index
        # 'SSI_index',        # Showalter index
        # 'LI_index',         # Lifted index
    ],
    'vas': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        'uas',              # surface wind
        'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',              # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        'ua850',             # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        'ua500',             # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        'va850',             # wind at pressure levels
        # 'va700',             # wind at pressure levels
        'va500',             # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        'ta850',             # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        'ta500',             # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'sfcWind': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        'uas',              # surface wind
        'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',              # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        'ua850',             # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        'ua500',             # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        'va850',             # wind at pressure levels
        # 'va700',             # wind at pressure levels
        'va500',             # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        'ta850',             # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        'ta500',             # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'hurs': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        'tas',              # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',             # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',             # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',             # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',             # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        'ta850',             # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        'ta500',             # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'huss': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'clt': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',              # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',             # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',             # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',             # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',             # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',             # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',             # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'rsds': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'rlds': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'evspsbl': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'evspsblpot': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'psl': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'ps': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'mrro': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'mrso': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        # 'psl',             # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        # 'uas',              # surface wind
        # 'vas',              # surface wind
        # 'sfcWind',              # surface wind speed
        # 'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        # 'hurs',              # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        # 'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        # 'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        # 'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        # 'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        # 'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        # 'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        # 'hur850',             # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        # 'hur500',             # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
    'myTargetVar': [
        # 'tasmax',             # maximum daily temperature
        # 'tasmin',             # minimum daily temperature
        # 'pr',              # daily precipitation
        'psl',  # mean sea level pressure
        # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
        # 'ps',             # surface pressure
        # 'ins',              # theoretical insolation (derived from dates)
        # 'clt',              # total cloud cover
        'uas',  # surface wind
        'vas',  # surface wind
        'sfcWind',  # surface wind speed
        'tas',  # surface temperature
        # 'tdps',              # surface dew point
        # 'huss',              # surface specific humidity
        'hurs',  # surface relative humidity
        # 'ua1000',            # wind at pressure levels
        'ua850',  # wind at pressure levels
        # 'ua700',             # wind at pressure levels
        'ua500',  # wind at pressure levels
        # 'ua250',             # wind at pressure levels
        # 'va1000',            # wind at pressure levels
        'va850',  # wind at pressure levels
        # 'va700',             # wind at pressure levels
        'va500',  # wind at pressure levels
        # 'va250',             # wind at pressure levels
        # 'ta1000',            # temperature at pressure levels
        'ta850',  # temperature at pressure levels
        # 'ta700',             # temperature at pressure levels
        'ta500',  # temperature at pressure levels
        # 'ta250',             # temperature at pressure levels
        # 'zg1000',            # geopotential at pressure levels
        # 'zg850',             # geopotential at pressure levels
        # 'zg700',             # geopotential at pressure levels
        # 'zg500',             # geopotential at pressure levels
        # 'zg250',             # geopotential at pressure levels
        # 'hus1000',            # specifit humidity at pressure levels
        # 'hus850',             # specifit humidity at pressure levels
        # 'hus700',             # specifit humidity at pressure levels
        # 'hus500',             # specifit humidity at pressure levels
        # 'hus250',             # specifit humidity at pressure levels
        # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
        'hur850',  # relative humidity at pressure levels (derived from t and q)
        # 'hur700',             # relative humidity at pressure levels (derived from t and q)
        'hur500',  # relative humidity at pressure levels (derived from t and q)
        # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    ],
}


###############################  SYNOPTIC ANALOGY FIELDS  ##############################################################

# Define synoptic analogy fields (saf)
saf_list = [
    # 'tasmax',             # maximum daily temperature
    # 'tasmin',             # minimum daily temperature
    # 'pr',              # daily precipitation
    # 'psl',             # mean sea level pressure
    # 'psl_trend',       # mean sea level trend from last day (derivedFrom mslp)
    # 'ps',             # surface pressure
    # 'ins',              # theoretical insolation (derived from dates)
    # 'clt',              # total cloud cover
    # 'uas',              # surface wind
    # 'vas',              # surface wind
    # 'sfcWind',              # surface wind speed
    # 'tas',              # surface temperature
    # 'tdps',              # surface dew point
    # 'huss',              # surface specific humidity
    # 'hurs',              # surface relative humidity
    # 'ua1000',            # wind at pressure levels
    # 'ua850',             # wind at pressure levels
    # 'ua700',             # wind at pressure levels
    # 'ua500',             # wind at pressure levels
    # 'ua250',             # wind at pressure levels
    # 'va1000',            # wind at pressure levels
    # 'va850',             # wind at pressure levels
    # 'va700',             # wind at pressure levels
    # 'va500',             # wind at pressure levels
    # 'va250',             # wind at pressure levels
    # 'ta1000',            # temperature at pressure levels
    # 'ta850',             # temperature at pressure levels
    # 'ta700',             # temperature at pressure levels
    # 'ta500',             # temperature at pressure levels
    # 'ta250',             # temperature at pressure levels
    # 'zg1000',            # geopotential at pressure levels
    # 'zg850',             # geopotential at pressure levels
    # 'zg700',             # geopotential at pressure levels
    'zg500',             # geopotential at pressure levels
    # 'zg250',             # geopotential at pressure levels
    # 'hus1000',            # specifit humidity at pressure levels
    # 'hus850',             # specifit humidity at pressure levels
    # 'hus700',             # specifit humidity at pressure levels
    # 'hus500',             # specifit humidity at pressure levels
    # 'hus250',             # specifit humidity at pressure levels
    # 'hur1000',            # relative humidity at pressure levels (derived from t and q)
    # 'hur850',             # relative humidity at pressure levels (derived from t and q)
    # 'hur700',             # relative humidity at pressure levels (derived from t and q)
    # 'hur500',             # relative humidity at pressure levels (derived from t and q)
    # 'hur250',             # relative humidity at pressure levels (derived from t and q)
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
    'tasmax': (
        'TXm',
        'TXx',
        'TXn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
        # 'TX99p',
        # 'TX95p',
        # 'TX90p',
        # 'TX10p',
        # 'TX5p',
        # 'TX1p',
        # 'SU',
        # 'ID',
        # 'WSDI',
    ),
    'tasmin': (
        'TNm',
        'TNx',
        'TNn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
        # 'TN99p',
        # 'TN95p',
        # 'TN90p',
        # 'TN10p',
        # 'TN5p',
        # 'TN1p',
        # 'FD',
        # 'TR',
        # 'CSDI',
    ),
    'tas': (
        'Tm',
        'Tx',
        'Tn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
        # 'T99p',
        # 'T95p',
        # 'T90p',
        # 'T10p',
        # 'T5p',
        # 'T1p',
    ),
    'pr': (
        'Pm',
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
    ),
    'uas': (
        'Um',
        'Ux',
    ),
    'vas': (
        'Vm',
        'Vx',
    ),
    'sfcWind':(
        'SFCWINDm',
        'SFCWINDx',
    ),
    'hurs': (
        'HRm',
    ),
    'huss': (
        'HUSSm',
        'HUSSx',
        'HUSSn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
    ),
    'clt': (
        'CLTm',
    ),
    'rsds': (
        'RSDSm',
        'RSDSx',
        'RSDSn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
    ),
    'rlds': (
        'RLDSm',
        'RLDSx',
        'RLDSn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
    ),
    'evspsbl': (
        'Em',
        'Ex',
        'En',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
    ),
    'evspsblpot': (
        'EPm',
        'EPx',
        'EPn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
    ),
    'psl': (
        'PSLm',
        'PSLx',
        'PSLn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
    ),
    'ps': (
        'PSm',
        'PSx',
        'PSn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
    ),
    'mrro': (
        'RUNOFFm',
        'RUNOFFx',
        'RUNOFFn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
    ),
    'mrso': (
        'SOILMOISTUREm',
        'SOILMOISTUREx',
        'SOILMOISTUREn',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
    ),
    'myTargetVar': (
        'm',
        'x',
        'n',
        # 'p99',
        # 'p95',
        # 'p90',
        # 'p10',
        # 'p5',
        # 'p1',
        # '99p_days',
        # '95p_days',
        # '90p_days',
        # '10p_days',
        # '5p_days',
        # '1p_days',
    ),
    }


###################################     Seasons           #################################################
# Season values can be adapted, but once they have been set do not change, because they are used both for filenames
# and for titles in figures. Never change keys of dictionary, that is what the program uses internally. Just change
# the values of the dictionary
# inverse_seasonNames = ['ANUAL',
#                        'INVIERNO', 'INVIERNO', 'PRIMAVERA',
#                        'PRIMAVERA', 'PRIMAVERA', 'VERANO',
#                        'VERANO', 'VERANO', 'OTOÑO',
#                        'OTOÑO',  'OTOÑO',  'INVIERNO']
inverse_seasonNames = ['ANNUAL',
                       'DJF', 'DJF', 'MAM',
                       'MAM', 'MAM', 'JJA',
                       'JJA', 'JJA', 'SON',
                       'SON',  'SON',  'DJF']
# inverse_seasonNames = ['ANNUAL',
#                        'DRY', 'DRY', 'DRY', 'DRY',
#                        'dry2wet',
#                        'WET', 'WET', 'WET', 'WET', 'WET',
#                        'wet2dry',  'wet2dry']


###################################     Bias correction   #################################################
apply_bc = False    # Apply bias correction after downscaling
apply_bc_bySeason = False # Apply bias correction customized for each season after downcaling

bc_method = 'QM'
# bc_method = 'DQM'
# bc_method = 'QDM'
# bc_method = 'PSDM'



###################################     myTargetVar           #################################################
myTargetVarName = 'fwi'

# Define myTargetVar min and max allowed values
myTargetVarMinAllowed = 0
myTargetVarMaxAllowed = None
myTargetVarUnits = ''

# # Define whether myTargetVar can be treated as gaussian
# myTargetVarIsGaussian = False

# Define whether myTargetVar should be treated as an additive of multiplicative variable
myTargetVarIsAdditive = False

# # Define whether apply bias correction as for precipitation (multiplicative correction)
# treatAsAdditiveBy_DQM_and_QDM = False
