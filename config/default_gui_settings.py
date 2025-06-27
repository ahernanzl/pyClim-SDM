
########################################       EXPERIMENT      #########################################################
# Select your current experiment type
experiment = 'EVALUATION'

####################################################################################################################
#                                        targetVars                                                                   #
####################################################################################################################
# Activate/deactivate targetVars
targetVars = [
    'tasmax',
    'tasmin',
    'pr',
]

methods = {
    'tasmax': [
        'RAW-BIL',
        'MLR',
        'WG-PDF',
    ],
    'tasmin': [
        'RAW-BIL',
        'MLR',
        'WG-PDF',
    ],
    'pr': [
        'RAW-BIL',
        'ANA-SYN-1NN',
        'GLM-LIN',
    ],
}


########################################       DATES      ##############################################################
# calibration_years corresponds to the longest period available by reanalysis and hres_data, which then can be split for
# training and testing
calibration_years = (1979, 2020)
single_split_testing_years = (2006, 2020)

# Reference: for standardization and future signal of change. The choice of the reference period is constrained by
# availability of reanalysis, historical GCMs and hres data.
reference_years = (1979, 2005)
reanalysisName = 'ERA5'

#############################################  GRIDS  ##################################################################
# saf grid (for synoptic analogy). All files (reanalysis and models) need to contain at least this region plus one grid box border.
saf_lat_up, saf_lat_down =  46.0, 34
saf_lon_left, saf_lon_right = -15.0, 9.0


###########################################   PREDICTORS  ##############################################################
# IMPORTANT: do not change preds order. Otherwise, lib/read.lres_data would need to be adapted to follow the same order
# Define, for each predictor variable, reaName and modName (variable names for reanalysis and for CMIP models)

reaNames = {'ua': 'u', 'va': 'v', 'ta': 't', 'zg': 'z', 'hus': 'q', 'hur': 'r', 'td': '-',
            'psl': 'msl', 'tdps': 'd2m', 'ps': 'sp',
            'tasmax': 'mx2t', 'tasmin': 'mn2t', 'tas': 't2m',
            'pr': 'tp', 'uas': 'u10', 'vas': 'v10', 'sfcWind': 'ws',
            'hurs': '-', 'huss': '-', 'clt': 'tcc',
            'rsds': 'ssrd', 'rlds': 'strd',
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
    'tasmax': ['tas', ],
    'tasmin': ['tas',  ],
    'pr': ['zg500', ],
}


###############################  SYNOPTIC ANALOGY FIELDS  ##############################################################

# Define synoptic analogy fields (saf)
saf_list = [
        'zg500',             # geopotential at pressure levels
]



###################################     SCENES AND MODELS    ###########################################################

# Define scenes and models. scene names for titles at figures. In filenames they are historical, ssp126, etc.
scene_names_list = [
    'HISTORICAL',
    'SSP5-8.5',
]

model_names_list = (
    'ACCESS-CM2_r1i1p1f1',
    'EC-Earth3_r1i1p1f1',
    'MIROC6_r1i1p1f1',
)



###################################     CLIMDEX    #####################################################################

# Define climdex to be used (see https://www.climdex.org/learn/indices/)
climdex_names = {
    'tasmax': (
        'TXm',
        'TXx',
        'TXn',
    ),
    'tasmin': (
        'TNm',
        'TNx',
        'TNn',
    ),
    'pr': (
        'Pm',
        'PRCPTOT',
        'R01',
        'SDII',
        'R95p',
    ),
    }

##################################     bias correction           #################################################
apply_bc = True

##################################     myTargetVar           #################################################
myTargetVarName = 'fwi'

# Define myTargetVar min and max allowed values
myTargetVarMinAllowed = 0
myTargetVarMaxAllowed = None
myTargetVarUnits = ''

# # Define whether myTargetVar can be treated as gaussian
# myTargetVarIsGaussian = False

# Define whether myTargetVar should be treated as an additive of multiplicative variable
myTargetVarIsAdditive = False
