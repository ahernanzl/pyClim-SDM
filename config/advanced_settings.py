import sys
sys.path.append('../config/')
from imports import *
from settings import *


# ########################################      WARNING       ##########################################################
# DO NOT CHANGE ANY ADVANCED SETTINGS UNLESS YOU ARE SURE OF WHAT YOU ARE DOING !!!

# ########################################  RUNNING OPTIONS   ##########################################################
# Predictands preprarataion. When working with stations, files with all stations have to be previously prepared.
# All stations should have the same number of data and have to be between the ranges indicated below.
# When building these files, no-data must be set to "special_value". Note thant pcp and temp scpecial_values differ.
# If min/max range is surpased, some adaptations must be made (change uint16 for uin32 for pcp, for example, which
# needs more memory

# This dictionary controls the interpolation used for each family
interp_dict = {'RAW': 'nearest', 'PP': 'bilinear', 'MOS': 'bilinear', 'WG': 'bilinear'}

# Predictands have to be between min/max. Use uint16/uint32 for precipitation depending on your data
predictands_codification = {
    'tmax': {'type': 'int16', 'min_valid': -327.68, 'max_valid': 327.66, 'special_value': 327.67},
    'tmin': {'type': 'int16', 'min_valid': -327.68, 'max_valid': 327.66, 'special_value': 327.67},
    'pcp': {'type': 'uint16', 'min_valid': 0, 'max_valid': 655.34, 'special_value': 655.35},
    # 'pcp': {'type': 'uint32', 'min_valid': 0, 'max_valid': 42949672.94, 'special_value': 42949672.95},
}

###################################     PSEUDOREALITY    ###########################################################

# Select a GCM/RCM combination if using pseudo reality and do nothing otherwise
pseudoreality, GCM_shortName, GCM_longName, RCM = False, None, None, None
# pseudoreality, GCM_shortName, GCM_longName, RCM = True, 'CNRM-CM5', 'CNRM-CERFACS-CNRM-CM5', 'CNRM-ALADIN63'
# pseudoreality, GCM_shortName, GCM_longName, RCM = True, 'CNRM-CM5', 'CNRM-CERFACS-CNRM-CM5', 'DMI-HIRHAM5'
# pseudoreality, GCM_shortName, GCM_longName, RCM = True, 'CNRM-CM5', 'CNRM-CERFACS-CNRM-CM5', 'KNMI-RACMO22E'
# pseudoreality, GCM_shortName, GCM_longName, RCM = True, 'IPSL-CM5A-MR', 'IPSL-IPSL-CM5A-MR', 'DMI-HIRHAM5'
# pseudoreality, GCM_shortName, GCM_longName, RCM = True, 'IPSL-CM5A-MR', 'IPSL-IPSL-CM5A-MR', 'KNMI-RACMO22E'
# pseudoreality, GCM_shortName, GCM_longName, RCM = True, 'MPI-ESM-LR', 'MPI-M-MPI-ESM-LR', 'CNRM-ALADIN63'
# pseudoreality, GCM_shortName, GCM_longName, RCM = True, 'MPI-ESM-LR', 'MPI-M-MPI-ESM-LR', 'KNMI-RACMO22E'

# Definition of paths (do not change)
if pseudoreality == True:
    experiment = 'PSEUDOREALITY'
    pathHres = '../input_data/OBS_PSEUDO/hres_' + GCM_longName + '_' + RCM + '/'
    pathAux = '../aux_RCM/'
else:
    pathHres = '../input_data/hres/'
    pathAux = '../aux/'
# Set to False only if generating figures for different subregions so they will be saved in subdirectories.
pathFigures = '../results/Figures/'
degree_sign = u'\N{DEGREE SIGN}C'

# Set the following boolean parameters
plot_hyperparameters = False # Plots hyperparameters when training SVR or LS-SVM. Important to see hyperparameters range
mean_and_std_from_GCM = True # Set to False to use them from reanalysis
get_reg_and_clf_scores = False # Computes and save R2 and accuracy at training

exp_var_ratio_th = .95 # threshold for PCA of SAFs
k_clusters = 250   # set to None first time, and when weather_types.set_number_of_clusters ends see elbow curve and
                    # set k_clusters properly
anal_pcp_corr_th = 0.2 # correlation threshold for analogs pcp significant predictors
min_days_corr = 30 # for analogs pcp significant predictors
wetDry_th = 0.1 # mm. It is used for classifiers (for climdex the threshold is 1 mm)
n_analogs_preselection = 150 # for analogs
kNN = 5 # for analogs
max_perc_missing_predictands_allowed = 20 # maximum percentage of missing predictands allowed
thresholds_WG_NMM = [0, 1, 2, 5, 10, 20, ]
aggregation_pcp_WG_PDF = 1
# aggregation_pcp_WG_PDF = 3

if experiment == 'EVALUATION':
    classifier_mode = 'deterministic' # clf.predict. Recommended if validating daily data
elif experiment in ('PROJECTIONS', 'PSEUDOREALITY'):
    classifier_mode = 'probabilistic' # clf.predict_proba. Recommended for out of range (extrapolation) classifications.
    # It slows down training.

# When a point and day has missing predictors, all days (for that point) will be recalibrated if True.
# If False, all days (for that point) will be calculated normally, and that particular day and point will be set to Nan
recalibrating_when_missing_preds = False

# Certain climdex make use of a reference period which can correspond to observations or to the proper method/model.
# That is the case of TX10p, R95p, etc. When evaluating ESD methods, set to True, but when studying change on the
# climdex (projections), set to False. This parameter is used in postporcess.get_climdex_oneModel
if experiment in ('EVALUATION', 'PSEUDOREALITY'):
    reference_climatology_from_observations = True
elif experiment == 'PROJECTIONS':
    reference_climatology_from_observations = False


########################################       DATES      ##############################################################
# Definition of testing_years and historical_years depending on the experiment (do not change)

if split_mode == 'all_training':
    testing_years = (calibration_years[1]+1, calibration_years[1]+2)
elif split_mode == 'all_testing':
    testing_years = calibration_years
elif split_mode == 'single_split':
    testing_years = single_split_testing_years
elif split_mode == 'fold1':
    testing_years = fold1_testing_years
elif split_mode == 'fold2':
    testing_years = fold2_testing_years
elif split_mode == 'fold3':
    testing_years = fold3_testing_years
elif split_mode == 'fold4':
    testing_years = fold4_testing_years
elif split_mode == 'fold5':
    testing_years = fold5_testing_years


if experiment == 'PSEUDOREALITY':
    calibration_years = (1961, 2005)
    testing_years  = (1986, 2005)
    historical_years = (1986, 2005)
    ssp_years = (2081, 2100)
shortTerm_years = (2040, 2069)
longTerm_years = (2071, 2100)

# Season values can be adapted, but once they have been set do not change, because they are used both for filenames
# and for titles in figures. Never change keys of dictionary, that is what the program uses internally. Just change
# the values of the dictionary
# season_dict = {'ANNUAL': 'ANUAL', 'DJF': 'INVIERNO', 'MAM': 'PRIMAVERA', 'JJA': 'VERANO', 'SON': 'OTOÑO'}
season_dict = {'ANNUAL': 'ANNUAL', 'DJF': 'DJF', 'MAM': 'MAM', 'JJA': 'JJA', 'SON': 'SON'}

# Hereafter different dates will be defined (do not change)
# Calibration (this will be separated later into training and testing)
calibration_first_date = datetime.datetime(calibration_years[0], 1, 1, 12, 0)
calibration_last_date = datetime.datetime(calibration_years[1], 12, 31, 12, 0)
calibration_ndates = (calibration_last_date-calibration_first_date).days+1
calibration_dates = [calibration_first_date + datetime.timedelta(days=i) for i in range(calibration_ndates)]
if ((pseudoreality==True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    calibration_dates = [x for x in calibration_dates if not ((x.month == 2) and (x.day == 29))]
    calibration_ndates = len(calibration_dates)
# Testing period
testing_first_date = datetime.datetime(testing_years[0], 1, 1, 12, 0)
testing_last_date = datetime.datetime(testing_years[1], 12, 31, 12, 0)
testing_ndates = (testing_last_date-testing_first_date).days+1
testing_dates = [testing_first_date + datetime.timedelta(days=i) for i in range(testing_ndates)]
if ((pseudoreality==True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    testing_dates = [x for x in testing_dates if not ((x.month == 2) and (x.day == 29))]
    testing_ndates = len(testing_dates)
# Training
training_dates = [x for x in calibration_dates if x not in testing_dates]
training_ndates = len(training_dates)
# Reference
reference_first_date = datetime.datetime(reference_years[0], 1, 1, 12, 0)
reference_last_date = datetime.datetime(reference_years[1], 12, 31, 12, 0)
reference_ndates = (reference_last_date-reference_first_date).days+1
reference_dates = [reference_first_date + datetime.timedelta(days=i) for i in range(reference_ndates)]
if ((pseudoreality==True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    reference_dates = [x for x in reference_dates if not ((x.month == 2) and (x.day == 29))]
    reference_ndates = len(reference_dates)
# BiasCorrection
biasCorr_first_date = datetime.datetime(biasCorr_years[0], 1, 1, 12, 0)
biasCorr_last_date = datetime.datetime(biasCorr_years[1], 12, 31, 12, 0)
biasCorr_ndates = (biasCorr_last_date-biasCorr_first_date).days+1
biasCorr_dates = [biasCorr_first_date + datetime.timedelta(days=i) for i in range(biasCorr_ndates)]
if ((pseudoreality==True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    biasCorr_dates = [x for x in biasCorr_dates if not ((x.month == 2) and (x.day == 29))]
    biasCorr_ndates = len(biasCorr_dates)
# historical scene
historical_first_date = datetime.datetime(historical_years[0], 1, 1, 12, 0)
historical_last_date = datetime.datetime(historical_years[1], 12, 31, 12, 0)
historical_ndates = (historical_last_date-historical_first_date).days+1
historical_dates = [historical_first_date + datetime.timedelta(days=i) for i in range(historical_ndates)]
if ((pseudoreality==True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    historical_dates = [x for x in historical_dates if not ((x.month == 2) and (x.day == 29))]
    historical_ndates = len(historical_dates)
# RCP scene
ssp_first_date = datetime.datetime(ssp_years[0], 1, 1, 12, 0)
ssp_last_date = datetime.datetime(ssp_years[1], 12, 31, 12, 0)
ssp_ndates = (ssp_last_date-ssp_first_date).days+1
ssp_dates = [ssp_first_date + datetime.timedelta(days=i) for i in range(ssp_ndates)]
if ((pseudoreality==True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    ssp_dates = [x for x in ssp_dates if not ((x.month == 2) and (x.day == 29))]
    ssp_ndates = len(ssp_dates)
# Short and long term
shortTermPeriodFilename = str(shortTerm_years[0]) + '-' + str(shortTerm_years[1])
longTermPeriodFilename = str(longTerm_years[0]) + '-' + str(longTerm_years[1])


#############################################  GRIDS  ##################################################################
aux_hres_metadata = np.loadtxt(pathHres + 'hres_metadata.txt')
hres_npoints = aux_hres_metadata.shape[0]
target_type = 'gridded_data'
# target_type = 'stations'

# ext
# ext_lat_up, ext_lat_down  = 55., 23.5
# ext_lon_left, ext_lon_right = -27, 21
ext_lat_up, ext_lat_down  = saf_lat_up+grid_res, saf_lat_down-grid_res
ext_lon_left, ext_lon_right = saf_lon_left-grid_res, saf_lon_right+grid_res
ext_nlats = int(1 + (ext_lat_up - ext_lat_down) / grid_res)
ext_nlons = int(1 + (ext_lon_right - ext_lon_left) / grid_res)
ext_lats = np.linspace(ext_lat_up, ext_lat_down, ext_nlats)
ext_lons = np.linspace(ext_lon_left, ext_lon_right, ext_nlons)

# saf
saf_nlats = int(((saf_lat_up - saf_lat_down) / 1.5 ) + 1)
saf_nlons = int(((saf_lon_right - saf_lon_left) / 1.5 ) + 1)
saf_lats = np.linspace(saf_lat_up, saf_lat_down, saf_nlats)
saf_lons = np.linspace(saf_lon_left, saf_lon_right, saf_nlons)
saf_ilats = [i for i in range(ext_nlats) if ext_lats[i] in saf_lats]
saf_ilons = [i for i in range(ext_nlons) if ext_lons[i] in saf_lons]

# pred grid (for predictors). Smaller area which covers, at least, the target region.
hres_lats, hres_lons = aux_hres_metadata[:, 2], aux_hres_metadata[:, 1]
pred_lat_up = np.min(saf_lats[saf_lats >= np.max(hres_lats)])
pred_lat_down = np.max(saf_lats[saf_lats <= np.min(hres_lats)])
pred_lon_right = np.min(saf_lons[saf_lons >= np.max(hres_lons)])
pred_lon_left = np.max(saf_lons[saf_lons <= np.min(hres_lons)])
pred_nlats = int(1 + (pred_lat_up - pred_lat_down) / grid_res)
pred_nlons = int(1 + (pred_lon_right - pred_lon_left) / grid_res)
pred_lats = np.linspace(pred_lat_up, pred_lat_down, pred_nlats)
pred_lons = np.linspace(pred_lon_left, pred_lon_right, pred_nlons)
pred_ilats = [i for i in range(ext_nlats) if ext_lats[i] in pred_lats]
pred_ilons = [i for i in range(ext_nlons) if ext_lons[i] in pred_lons]

###############################  SYNOPTIC ANALOGY FIELDS  ##############################################################

# Build saf_dict
saf_dict = {}
for pred in saf_list:
    key = pred.replace('1000', '').replace('850', '').replace('700', '').replace('500', '').replace('250', '')
    if key in reaNames:
        reaName = reaNames[key]
        modName = modNames[key]
    else:
        reaName = None
        modName = None
    saf_dict.update({pred: {'reaName': reaName, 'modName': modName, 'w': 1}})

nsaf = len(saf_dict)

# Create W_saf (weights for synoptic analogy fields).Do not change
W_saf = np.ones((nsaf, saf_nlats, saf_nlons))
i = 0
for saf in saf_dict:
    W_saf[i] = saf_dict[saf]['w'] * np.ones((saf_nlats, saf_nlons))
    i += 1
W_saf = W_saf.flatten()


###########################################   PREDICTORS  ###############################################################

# Build preds_t
preds_t = {}
for pred in preds_t_list:
    key = pred.replace('1000', '').replace('850', '').replace('700', '').replace('500', '').replace('250', '')
    if key in reaNames:
        reaName = reaNames[key]
        modName = modNames[key]
    else:
        reaName = None
        modName = None
    preds_t.update({pred: {'reaName': reaName, 'modName': modName}})

# Build preds_p
preds_p = {}
for pred in preds_p_list:
    key = pred.replace('1000', '').replace('850', '').replace('700', '').replace('500', '').replace('250', '')
    if key in reaNames:
        reaName = reaNames[key]
        modName = modNames[key]
    else:
        reaName = None
        modName = None
    preds_p.update({pred: {'reaName': reaName, 'modName': modName}})


n_preds_t, n_preds_p = len(preds_t.keys()), len(preds_p.keys())

all_preds = {**preds_t, **preds_p, **saf_dict}

preds_levels = []
for level in [1000, 850, 700, 500, 250]:
    for pred in all_preds:
        if str(level) in pred:
            preds_levels.append(level)
preds_levels = list(dict.fromkeys(preds_levels))

if 'pcp' in all_preds.keys():
    print('---------------------------------------------------------------')
    print('CAUTION: predictors will be standardized, and precipitation should not be mixed with other predictors and used that way.')
    print('---------------------------------------------------------------')

###################################     SCENES AND MODELS    ###########################################################
scene_names_dict = {}
for scene in scene_names_list:
    scene_names_dict.update({scene.replace('.', '').replace('-', '').lower(): scene})

# Define scene_list and model_list depending on experiment type (do not change)
if experiment == 'EVALUATION':
    scene_list = ('TESTING', )
    model_list = ('reanalysis', )
elif experiment == 'PROJECTIONS':
    scene_list = list(scene_names_dict.keys())
    model_list = model_names_list
elif experiment == 'PSEUDOREALITY':
    scene_list = ('historical', 'RCP85')
    model_list = (GCM_shortName, )
elif experiment == 'GCM_EVAL':
    scene_list = list(scene_names_dict.keys())
    model_list = model_names_list

###################################     SUBREGIONS    #####################################################################

# This program allows to analyse results by subregions. Region types and names are handled by grids.subregions()
# At the moment only EspañaPB is implemented, but adaptation to other regions (and their shapefiles) is prepared to be easy
typeCompleteRegion = 'SPAIN' # It would make more sense to call it "COUNTRY", but it was designed as "SPAIN" for the website
nameCompleteRegion = 'EspañaPB' # Chose between EspañaPB or Canarias (Canarias to be implemented)
divideByRegions = False # Set to True if a division by regions will be done, and to False if no regions will be used
plotAllRegions = False  # Set to False so only the complete region will be plotted and to True so all regions will be plotted

#
# PROV_list = ['A Coruña', 'Alacant/Alicante', 'Albacete', 'Almería', 'Araba/Álava', 'Asturias', 'Badajoz', 'Barcelona',
#          'Bizkaia', 'Burgos', 'Cantabria', 'Castelló/Castellón', 'Ciudad Real', 'Cuenca', 'Cáceres', 'Cádiz', 'Córdoba',
#          'Gipuzkoa', 'Girona', 'Granada', 'Guadalajara', 'Huelva', 'Huesca', 'Illes Balears', 'Jaén', 'La Rioja',
#          'León', 'Lleida', 'Lugo', 'Madrid', 'Melilla', 'Murcia', 'Málaga', 'Navarra', 'Ourense', 'Palencia',
#          'Pontevedra', 'Salamanca', 'Segovia', 'Sevilla', 'Soria', 'Tarragona', 'Teruel', 'Toledo', 'Valladolid',
#          'València/Valencia', 'Zamora', 'Zaragoza', 'Ávila']
#
# CCAA_list = ['Andalucía', 'Aragón', 'Cantabria', 'Castilla y León', 'Castilla-La Mancha', 'Cataluña/Catalunya',
#          'Ciudad Autónoma de Melilla', 'Comunidad Foral de Navarra', 'Comunidad de Madrid', 'Comunitat Valenciana',
#          'Extremadura', 'Galicia', 'Illes Balears', 'La Rioja', 'País Vasco/Euskadi', 'Principado de Asturias',
#          'Región de Murcia']
#
# CCHH_list = ['CANTÁBRICO OCCIDENTAL', 'CANTÁBRICO ORIENTAL', 'CEUTA', 'CUENCAS INTERNAS DE CATALUÑA',
#            'CUENCAS MEDITERRÁNEAS ANDALUZAS', 'DUERO', 'EBRO', 'GALICIA-COSTA', 'GUADALETE Y BARBATE', 'GUADALQUIVIR',
#            'GUADIANA', 'ISLAS BALEARES', 'JÚCAR', 'MELILLA', 'MIÑO-SIL', 'SEGURA', 'TAJO', 'TINTO, ODIEL Y PIEDRAS']


# ####################  HIGH PERFORMANCE COMPUTER (HPC) OPTIONS    #####################################################
# Ir running in a HPC, define partition to lauch jobs here or in a private_settings.py file
user = os.popen('whoami').read().split('\n')[0]
max_nJobs = 5 # Max number of jobs
if os.path.isfile('../private/private_settings.py'):
    sys.path.append(('../private/'))
    from private_settings import *
else:
    running_at_HPC, HPC_partition = False, 'enterPartitionName'

# ####################  COLORS AND STYLES    #####################################################
t_methods_colors = {
    'RAW': 'lightgray',
    'QM': 'orange',
    'DQM': 'orange',
    'QDM': 'orange',
    'PSDM': 'orange',
    'ANA-MLR': 'r',
    'WT-MLR': 'r',
    'MLR': 'cyan',
    'ANN': 'b',
    'SVM': 'b',
    'LS-SVM': 'b',
    'WG-PDF': 'g',
}

t_methods_linestyles = {
    'RAW': '-',
    'QM': '-',
    'DQM': '--',
    'QDM': ':',
    'PSDM': '-.',
    'ANA-MLR': '-',
    'WT-MLR': '--',
    'MLR': '-',
    'ANN': '-',
    'SVM': '--',
    'LS-SVM': ':',
    'WG-PDF': '-',
}

p_methods_colors = {
    'RAW': 'lightgray',
    'QM': 'orange',
    'DQM': 'orange',
    'QDM': 'orange',
    'PSDM': 'orange',
    'ANA-SYN-1NN': 'r',
    'ANA-SYN-kNN': 'r',
    'ANA-SYN-rand': 'r',
    'ANA-LOC-1NN': 'lightcoral',
    'ANA-LOC-kNN': 'lightcoral',
    'ANA-LOC-rand': 'lightcoral',
    'ANA-PCP-1NN': 'darkred',
    'ANA-PCP-kNN': 'darkred',
    'ANA-PCP-rand': 'darkred',
    'GLM-LIN': 'cyan',
    'GLM-EXP': 'cyan',
    'GLM-CUB': 'cyan',
    'ANN': 'b',
    'SVM': 'b',
    'LS-SVM': 'b',
    'WG-NMM': 'g',
    'WG-PDF': 'g',
}

p_methods_linestyles = {
    'RAW': '-',
    'QM': '-',
    'DQM': '--',
    'QDM': ':',
    'PSDM': '-.',
    'ANA-SYN-1NN': '-',
    'ANA-SYN-kNN': '--',
    'ANA-SYN-rand': ':',
    'ANA-LOC-1NN': '-',
    'ANA-LOC-kNN': '--',
    'ANA-LOC-rand': ':',
    'ANA-PCP-1NN': '-',
    'ANA-PCP-kNN': '--',
    'ANA-PCP-rand': ':',
    'GLM-LIN': '-',
    'GLM-EXP': '--',
    'GLM-CUB': ':',
    'ANN': '-',
    'SVM': '--',
    'LS-SVM': ':',
    'WG-NMM': '-',
    'WG-PDF': '--',
}
