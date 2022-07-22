import sys

sys.path.append('../config/')
from imports import *
from settings import *

# ########################################      WARNING       ##########################################################
# DO NOT CHANGE ANY ADVANCED SETTINGS UNLESS YOU ARE SURE OF WHAT YOU ARE DOING !!!

# ####################  HIGH PERFORMANCE COMPUTER (HPC) OPTIONS    #####################################################
# Ir running in a HPC, define partition to lauch jobs here or in a private_settings.py file
user = os.popen('whoami').read().split('\n')[0]
max_nJobs = 5  # Max number of jobs
if os.path.isfile('../private/private_settings.py'):
    sys.path.append(('../private/'))
    from private_settings import *
else:
    running_at_HPC, HPC_partition = False, 'enterPartitionName'
if running_at_HPC == True:
    from mpi4py import MPI
if running_at_HPC == False:
    from mpl_toolkits.basemap import Basemap

# ########################################  RUNNING OPTIONS   ##########################################################
# Predictands preprarataion. When working with stations, files with all stations have to be previously prepared.
# All stations should have the same number of data and have to be between the ranges indicated below.
# When building these files, no-data must be set to "special_value". Note thant pcp and temp scpecial_values differ.
# If min/max range is surpased, some adaptations must be made (change uint16 for uin32 for pcp, for example, which
# needs more memory

# This parameter controls the interpolation used for predictors
# interp_mode = 'nearest'
interp_mode = 'bilinear'

###################################     myTargetVar           #################################################
if 'myTargetVar' not in locals():
    myTargetVar = 'None'

###################################     predictands           #################################################
# Predictands have to be between min/max as inputs. Use uint16/uint32 for precipitation depending on your data
predictands_codification = {
    'tasmax': {'type': 'int16', 'min_valid': -327.68, 'max_valid': 327.66, 'special_value': 327.67},
    'tasmin': {'type': 'int16', 'min_valid': -327.68, 'max_valid': 327.66, 'special_value': 327.67},
    'tas': {'type': 'int16', 'min_valid': -327.68, 'max_valid': 327.66, 'special_value': 327.67},
    # 'pr': {'type': 'uint16', 'min_valid': 0, 'max_valid': 655.34, 'special_value': 655.35},
    'pr': {'type': 'uint32', 'min_valid': 0, 'max_valid': 42949672.94, 'special_value': 42949672.95},
    'uas': {'type': 'int16', 'min_valid': -327.68, 'max_valid': 327.66, 'special_value': 327.67},
    'vas': {'type': 'int16', 'min_valid': -327.68, 'max_valid': 327.66, 'special_value': 327.67},
    'sfcWind': {'type': 'uint16', 'min_valid': 0, 'max_valid': 655.34, 'special_value': 655.35},
    'hurs': {'type': 'int16', 'min_valid': -327.68, 'max_valid': 327.66, 'special_value': 327.67},
    'clt': {'type': 'int16', 'min_valid': -327.68, 'max_valid': 327.66, 'special_value': 327.67},
}
if myTargetVar != 'None':
    predictands_codification.update(
        {myTargetVar:
             {'type': 'int32', 'min_valid': -21474836.48, 'max_valid': 21474836.46, 'special_value': 21474836.47}}
    )

# Predictands have to be between min/max theoretically
predictands_range = {
    'tasmax': {'min': None, 'max': None},
    'tasmin': {'min': None, 'max': None},
    'tas': {'min': None, 'max': None},
    'pr': {'min': 0, 'max': None},
    'uas': {'min': None, 'max': None},
    'vas': {'min': None, 'max': None},
    'sfcWind': {'min': 0, 'max': None},
    'hurs': {'min': 0, 'max': 100},
    'clt': {'min': 0, 'max': 100},
}
if myTargetVar != 'None':
    predictands_range.update({myTargetVar: {'min': myTargetVarMinAllowed, 'max': myTargetVarMaxAllowed}})

# Predictands have to be between min/max. Use uint16/uint32 for precipitation depending on your data
degree_sign = u'\N{DEGREE SIGN}C'
predictands_units = {
    'tasmax': degree_sign,
    'tasmin': degree_sign,
    'tas': degree_sign,
    'pr': 'mm',
    'uas': 'm/s',
    'vas': 'm/s',
    'sfcWind': 'm/s',
    'hurs': '%',
    'clt': '%',
}
if myTargetVar != 'None':
    predictands_units.update({myTargetVar: myTargetVarUnits})

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

# Plots hyperparameters, epochs, nEstimators and featureImportances for Machine Learning methods (and trains only one out of 500 points)
# This is used to establish the hyperparameters range, but once that has been done, set to False to train the methods.
# If set to True, figures will be storaged at aux/TRAINED_MODELS
plot_hyperparameters_epochs_nEstimators_featureImportances = False

# When standarizing predictors from a GCM, use its own mean and std or take them from the reanalysis
mean_and_std_from_GCM = True

exp_var_ratio_th = .95  # threshold for PCA of SAFs
k_clusters = 250  # set to None first time, and when weather_types.set_number_of_clusters ends see elbow curve and
# set k_clusters properly
anal_corr_th_dict = {
    'tasmax': 0.7,
    'tasmin': 0.7,
    'tas': 0.7,
    'precipitation': 0.2,
    'uas': 0.7,
    'vas': 0.7,
    'sfcWind': 0.7,
    'hurs': 0.7,
    'clt': 0.7,
}
if myTargetVar != 'None':
    anal_corr_th_dict.update({myTargetVar: .5})

min_days_corr = 30  # for analogs pcp significant predictors
wetDry_th = 0.1  # mm. It is used for classifiers (for climdex the threshold is 1 mm)
n_analogs_preselection = 150  # for analogs
kNN = 5  # for analogs
max_perc_missing_predictands_allowed = 20  # maximum percentage of missing predictands allowed
thresholds_WG_NMM = [0, 1, 2, 5, 10, 20, ]
aggregation_pcp_WG_PDF = 1
# aggregation_pcp_WG_PDF = 3

if experiment == 'EVALUATION':
    classifier_mode = 'deterministic'  # clf.predict. Recommended if validating daily data
elif experiment in ('PROJECTIONS', 'PSEUDOREALITY'):
    classifier_mode = 'probabilistic'  # clf.predict_proba. Recommended for out of range (extrapolation) classifications.
    # It slows down training.

# When a point and day has missing predictors, all days (for that point) will be recalibrated if True.
# If False, all days (for that point) will be calculated normally, and that particular day and point will be set to Nan
recalibrating_when_missing_preds = False

# Transfer function methods can use local predictors (nearest neightbour / bilinear) or predictors from the whole grid
# When using the whole grid, they have more information as inputs, but that consumes more memory
# Furthermore, when using the whole grid, a missing value affects all points, so more problems related to missing data
# will arise.
methods_using_preds_from_whole_grid = ['CNN', ]

# Certain climdex make use of a reference period which can correspond to observations or to the proper method/model.
# That is the case of TX10p, R95p, etc. When evaluating ESD methods, set to True, but when studying change on the
# climdex (projections), set to False. This parameter is used in postporcess.get_climdex_oneModel
if experiment in ('EVALUATION', 'PSEUDOREALITY'):
    reference_climatology_from_observations = True
elif experiment == 'PROJECTIONS':
    reference_climatology_from_observations = False


# Controls the path name for bias corrected outputs
if apply_bc == False:
    bc_sufix = ''
else:
    bc_sufix = '-BC-' + bc_method
    if apply_bc_bySeason == True:
        bc_sufix += '-s'

########################################       DATES      ##############################################################
# Definition of testing_years and historical_years depending on the experiment (do not change)

if split_mode == 'all_training':
    testing_years = (calibration_years[1] + 1, calibration_years[1] + 2)
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

biasCorr_years = reference_years

if experiment == 'PSEUDOREALITY':
    calibration_years = (1961, 2005)
    testing_years = (1986, 2005)
    historical_years = (1986, 2005)
    ssp_years = (2081, 2100)
shortTerm_years = (2041, 2070)
longTerm_years = (2071, 2100)

# Hereafter different dates will be defined (do not change)
# Calibration (this will be separated later into training and testing)
calibration_first_date = datetime.datetime(calibration_years[0], 1, 1, 12, 0)
calibration_last_date = datetime.datetime(calibration_years[1], 12, 31, 12, 0)
calibration_ndates = (calibration_last_date - calibration_first_date).days + 1
calibration_dates = [calibration_first_date + datetime.timedelta(days=i) for i in range(calibration_ndates)]
if ((pseudoreality == True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    calibration_dates = [x for x in calibration_dates if not ((x.month == 2) and (x.day == 29))]
    calibration_ndates = len(calibration_dates)
# Testing period
testing_first_date = datetime.datetime(testing_years[0], 1, 1, 12, 0)
testing_last_date = datetime.datetime(testing_years[1], 12, 31, 12, 0)
testing_ndates = (testing_last_date - testing_first_date).days + 1
testing_dates = [testing_first_date + datetime.timedelta(days=i) for i in range(testing_ndates)]
if ((pseudoreality == True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    testing_dates = [x for x in testing_dates if not ((x.month == 2) and (x.day == 29))]
    testing_ndates = len(testing_dates)
# Training
training_dates = [x for x in calibration_dates if x not in testing_dates]
training_ndates = len(training_dates)
# Reference
reference_first_date = datetime.datetime(reference_years[0], 1, 1, 12, 0)
reference_last_date = datetime.datetime(reference_years[1], 12, 31, 12, 0)
reference_ndates = (reference_last_date - reference_first_date).days + 1
reference_dates = [reference_first_date + datetime.timedelta(days=i) for i in range(reference_ndates)]
if ((pseudoreality == True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    reference_dates = [x for x in reference_dates if not ((x.month == 2) and (x.day == 29))]
    reference_ndates = len(reference_dates)
# BiasCorrection
biasCorr_first_date = datetime.datetime(biasCorr_years[0], 1, 1, 12, 0)
biasCorr_last_date = datetime.datetime(biasCorr_years[1], 12, 31, 12, 0)
biasCorr_ndates = (biasCorr_last_date - biasCorr_first_date).days + 1
biasCorr_dates = [biasCorr_first_date + datetime.timedelta(days=i) for i in range(biasCorr_ndates)]
if ((pseudoreality == True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    biasCorr_dates = [x for x in biasCorr_dates if not ((x.month == 2) and (x.day == 29))]
    biasCorr_ndates = len(biasCorr_dates)
# historical scene
historical_first_date = datetime.datetime(historical_years[0], 1, 1, 12, 0)
historical_last_date = datetime.datetime(historical_years[1], 12, 31, 12, 0)
historical_ndates = (historical_last_date - historical_first_date).days + 1
historical_dates = [historical_first_date + datetime.timedelta(days=i) for i in range(historical_ndates)]
if ((pseudoreality == True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    historical_dates = [x for x in historical_dates if not ((x.month == 2) and (x.day == 29))]
    historical_ndates = len(historical_dates)
# RCP scene
ssp_first_date = datetime.datetime(ssp_years[0], 1, 1, 12, 0)
ssp_last_date = datetime.datetime(ssp_years[1], 12, 31, 12, 0)
ssp_ndates = (ssp_last_date - ssp_first_date).days + 1
ssp_dates = [ssp_first_date + datetime.timedelta(days=i) for i in range(ssp_ndates)]
if ((pseudoreality == True) and (GCM_shortName == 'IPSL-CM5A-MR')):
    ssp_dates = [x for x in ssp_dates if not ((x.month == 2) and (x.day == 29))]
    ssp_ndates = len(ssp_dates)
# Short and long term
shortTermPeriodFilename = str(shortTerm_years[0]) + '-' + str(shortTerm_years[1])
longTermPeriodFilename = str(longTerm_years[0]) + '-' + str(longTerm_years[1])

# ####################  METHODS    #####################################################
# Family, mode and fields for each method is defined here. Field var stands for the target variable it self (at low res).
# Field pred stands for predictors (the ones selected for each target variable). And field saf stands for synoptic
# analogy fields
families_modes_and_fields = {
    'RAW': ['RAW', 'RAW', 'var'],
    'RAW-BIL': ['RAW', 'RAW', 'var'],
    'QM': ['MOS', 'MOS', 'var'],
    'DQM': ['MOS', 'MOS', 'var'],
    'QDM': ['MOS', 'MOS', 'var'],
    'PSDM': ['MOS', 'MOS', 'var'],
    'ANA-SYN-1NN': ['ANA', 'PP', 'saf'],
    'ANA-SYN-kNN': ['ANA', 'PP', 'saf'],
    'ANA-SYN-rand': ['ANA', 'PP', 'saf'],
    'ANA-LOC-1NN': ['ANA', 'PP', 'pred+saf'],
    'ANA-LOC-kNN': ['ANA', 'PP', 'pred+saf'],
    'ANA-LOC-rand': ['ANA', 'PP', 'pred+saf'],
    'ANA-VAR-1NN': ['ANA', 'PP', 'var'],
    'ANA-VAR-kNN': ['ANA', 'PP', 'var'],
    'ANA-VAR-rand': ['ANA', 'PP', 'var'],
    'MLR': ['TF', 'PP', 'pred'],
    'MLR-ANA': ['ANA', 'PP', 'pred+saf'],
    'MLR-WT': ['ANA', 'PP', 'pred+saf'],
    'GLM-LIN': ['TF', 'PP', 'pred'],
    'GLM-EXP': ['TF', 'PP', 'pred'],
    'GLM-CUB': ['TF', 'PP', 'pred'],
    'SVM': ['TF', 'PP', 'pred'],
    'LS-SVM': ['TF', 'PP', 'pred'],
    'RF': ['TF', 'PP', 'pred+var'],
    'XGB': ['TF', 'PP', 'pred+var'],
    'ANN': ['TF', 'PP', 'pred'],
    'CNN': ['TF', 'PP', 'pred'],
    'WG-PDF': ['WG', 'PP', 'var'],
    'WG-NMM': ['WG', 'PP', 'var'],
}

methods_list = []
for targetVar in methods:
    for methodName in methods[targetVar]:
        methods_list.append({'var': targetVar, 'methodName': methodName,
                             'family': families_modes_and_fields[methodName][0],
                             'mode': families_modes_and_fields[methodName][1],
                             'fields': families_modes_and_fields[methodName][2], })

methods = methods_list
del methods_list

if myTargetVar != 'None':
    myTargetVar_methods = [x['methodName'] for x in methods if x['var'] == myTargetVar]
    if myTargetVarIsGaussian == False:
        if 'PSDM' in myTargetVar_methods:
            print('PSDM not allowed when using a non gaussian myTargetVar')
            exit()
        if 'WG-PDF' in myTargetVar_methods:
            print('WG-PDF not allowed when using a non gaussian myTargetVar')
            exit()
        if bc_method == 'PSDM':
            print('PSDM not allowed when using a non gaussian myTargetVar')
            exit()

###################################     Seasons           #################################################
seasonNames = []
for seasonName in inverse_seasonNames:
    if seasonName not in seasonNames:
        seasonNames.append(seasonName)
season_dict = {}
for seasonName in seasonNames:
    months = []
    for month in range(1, 13):
        try:
            if inverse_seasonNames[month] == seasonName:
                months.append(month)
        except:
            print('ERROR: seasons have not been defined properly')
            print('Define season name for month', month)
            exit()
    season_dict.update({seasonName: months})
season_dict.update({inverse_seasonNames[0]: range(1, 13)})
annualName = inverse_seasonNames[0]

###############################  SYNOPTIC ANALOGY FIELDS  ##############################################################
all_levels = [1000, 850, 700, 500, 250]

# Force to define at least one synoptic analogy field
if len(saf_list) == 0:
    print('-----------------------------------------------')
    print('At least one field must be selected for Synoptic Analogy Fields')
    print('-----------------------------------------------')
    exit()

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


###########################################   PREDICTORS  ###############################################################


# Detect targetVars depending on selected methods
all_possible_targetVars = ['tasmax', 'tasmin', 'tas', 'pr', 'uas', 'vas', 'sfcWind', 'hurs', 'clt', ]
try:
    all_possible_targetVars.append(myTargetVar)
except:
    pass

targetVars = []
for var in all_possible_targetVars:
    for method in methods:
        # if method['var'] == var and 'var' in method['fields'] and var not in targetVars:
        if method['var'] == var and var not in targetVars:
            targetVars.append(var)


preds_dict = {}
for targetVar in targetVars:
    preds_dict.update({targetVar: {}})
    for pred in preds_targetVars_dict[targetVar]:
        key = pred.replace('1000', '').replace('850', '').replace('700', '').replace('500', '').replace('250', '')
        if key in reaNames:
            reaName = reaNames[key]
            modName = modNames[key]
        else:
            reaName = None
            modName = None
        preds_dict[targetVar].update({pred: {'reaName': reaName, 'modName': modName}})

n_preds_dict = {}
for targetVar in targetVars:
    n_preds_dict.update({targetVar: len(preds_dict[targetVar].keys())})

all_preds = {}
for pred in saf_list:
    if pred not in all_preds:
        all_preds.update({pred: {'reaName': reaName, 'modName': modName}})
for targetVar in targetVars:
    for pred in preds_targetVars_dict[targetVar]:
        if pred not in all_preds:
            all_preds.update({pred: {'reaName': reaName, 'modName': modName}})

preds_levels = []
for level in all_levels:
    for pred in all_preds:
        if str(level) in pred:
            preds_levels.append(level)
preds_levels = list(dict.fromkeys(preds_levels))

if 'pr' in all_preds.keys():
    print('---------------------------------------------------------------')
    print(
        'CAUTION: predictors will be standardized, and precipitation should not be mixed with other predictors and used that way.')
    print(
        'Standardizing precipitation will lead to zeros being represented by different values depending on the model.')
    print('This is not advisable and can lead to poor performance of Transfer Function methods.')
    print('---------------------------------------------------------------')

# Check for consistency between predictors and methods
for var in all_possible_targetVars:
    if (var in targetVars) and (len(preds_targetVars_dict[var]) == 0) and (experiment != 'PRECONTROL'):
        print('-----------------------------------------------')
        print('Inconsistency found between preditors and methods selection.')
        print('Your selection includes some methods for ' + var + ' but no predictor has been selected')
        print('-----------------------------------------------')
        exit()

# Force at least one predictor
for targetVar in targetVars:
    if len(preds_targetVars_dict[targetVar]) == 0:
        print('-----------------------------------------------')
        print('At least one predictor must be selected for', targetVar)
        print('-----------------------------------------------')
        exit()

#############################################  GRIDS  ##################################################################

target_type = 'gridded_data'
# target_type = 'stations'

hres_npoints, hres_lats, hres_lons = {}, {}, {}
for targetVar in targetVars:
    if not os.path.isfile(pathHres + targetVar + '_hres_metadata.txt'):
        print('----------------------------------------------------------------------------------------')
        print('Make sure your input_data directory is prepared as indicated in the input_data_template.')
        print('Missing hres/' + targetVar + '_hres_metadata.txt file.')
        print('----------------------------------------------------------------------------------------')
        exit()
    aux_hres_metadata = np.loadtxt(pathHres + targetVar + '_hres_metadata.txt')
    hres_npoints.update({targetVar: aux_hres_metadata.shape[0]})
    hres_lats.update({targetVar: aux_hres_metadata[:, 2]})
    hres_lons.update({targetVar: aux_hres_metadata[:, 1]})

hres_lats_all = []
for targetVar in targetVars:
    for i in list(hres_lats[targetVar]):
        hres_lats_all.append(i)
hres_lats_all = np.asarray(hres_lats_all)
hres_lons_all = []
for targetVar in targetVars:
    for i in list(hres_lons[targetVar]):
        hres_lons_all.append(i)
hres_lons_all = np.asarray(hres_lons_all)

# Modify saf_lat_up, saf_lat_down, saf_lon_left and saf_lon_right forcing to exist in the netCDF files
for targetVar in targetVars:
    try:
        nc = Dataset('../input_data/reanalysis/'+reaNames[targetVar]+'_'+reanalysisName+'_'+reanalysisPeriodFilename+'.nc')
        break
    except:
        pass

if 'lat' in nc.variables:
    lat_name, lon_name = 'lat', 'lon'
elif 'latitude' in nc.variables:
    lat_name, lon_name = 'latitude', 'longitude'
lats = nc.variables[lat_name][:]
lons = nc.variables[lon_name][:]
lons[lons > 180] -= 360
saf_lat_down = np.min(lats[lats >= saf_lat_down])
saf_lat_up = np.max(lats[lats <= saf_lat_up])
saf_lon_left = np.min(lons[lons >= saf_lon_left])
saf_lon_right = np.max(lons[lons <= saf_lon_right])

# ext
ext_lat_up, ext_lat_down = saf_lat_up + grid_res, saf_lat_down - grid_res
ext_lon_left, ext_lon_right = saf_lon_left - grid_res, saf_lon_right + grid_res
ext_nlats = int(1 + (ext_lat_up - ext_lat_down) / grid_res)
ext_nlons = int(1 + (ext_lon_right - ext_lon_left) / grid_res)
ext_lats = np.linspace(ext_lat_up, ext_lat_down, ext_nlats)
ext_lons = np.linspace(ext_lon_left, ext_lon_right, ext_nlons)
ext_ilats = [i for i in range(ext_nlats)]
ext_ilons = [i for i in range(ext_nlons)]

# saf
saf_nlats = int(((saf_lat_up - saf_lat_down) / grid_res) + 1)
saf_nlons = int(((saf_lon_right - saf_lon_left) / grid_res) + 1)
saf_lats = np.linspace(saf_lat_up, saf_lat_down, saf_nlats)
saf_lons = np.linspace(saf_lon_left, saf_lon_right, saf_nlons)
saf_ilats = [i for i in range(ext_nlats) if ext_lats[i] in saf_lats]
saf_ilons = [i for i in range(ext_nlons) if ext_lons[i] in saf_lons]

# Check that hres_points are fully contained in the defined domain
if saf_lats[saf_lats >= np.max(hres_lats_all)].size == 0 or saf_lats[saf_lats <= np.min(hres_lats_all)].size == 0 or \
        saf_lons[saf_lons >= np.max(hres_lons_all)].size == 0 or saf_lons[saf_lons <= np.min(hres_lons_all)].size == 0:
    print('hres_points are not fully contained inside the domain defined for Synoptic Analogy Fields')
    print('Please, define a larger domain')
    exit()

# pred grid (for predictors). Smaller area which covers, at least, the target region.
pred_lat_up = np.min(saf_lats[saf_lats >= np.max(hres_lats_all)])
pred_lat_down = np.max(saf_lats[saf_lats <= np.min(hres_lats_all)])
pred_lon_right = np.min(saf_lons[saf_lons >= np.max(hres_lons_all)])
pred_lon_left = np.max(saf_lons[saf_lons <= np.min(hres_lons_all)])
pred_nlats = int(1 + (pred_lat_up - pred_lat_down) / grid_res)
pred_nlons = int(1 + (pred_lon_right - pred_lon_left) / grid_res)
pred_lats = np.linspace(pred_lat_up, pred_lat_down, pred_nlats)
pred_lons = np.linspace(pred_lon_left, pred_lon_right, pred_nlons)
pred_ilats = [i for i in range(ext_nlats) if ext_lats[i] in pred_lats]
pred_ilons = [i for i in range(ext_nlons) if ext_lons[i] in pred_lons]

##################################      SAF WEIGHTS         ############################################################
# Create W_saf (weights for synoptic analogy fields).Do not change
W_saf = np.ones((nsaf, saf_nlats, saf_nlons))
i = 0
for saf in saf_dict:
    W_saf[i] = saf_dict[saf]['w'] * np.ones((saf_nlats, saf_nlons))
    i += 1
W_saf = W_saf.flatten()

###################################     SCENES AND MODELS    ###########################################################
scene_names_dict = {}
for scene in scene_names_list:
    scene_names_dict.update({scene.replace('.', '').replace('-', '').lower(): scene})

# Define scene_list and model_list depending on experiment type (do not change)
if experiment == 'EVALUATION':
    scene_list = ('TESTING',)
    model_list = ('reanalysis',)
elif experiment == 'PROJECTIONS':
    scene_list = list(scene_names_dict.keys())
    model_list = model_names_list
elif experiment == 'PSEUDOREALITY':
    scene_list = ('historical', 'RCP85')
    model_list = (GCM_shortName,)
elif experiment == 'PRECONTROL':
    scene_list = list(scene_names_dict.keys())
    model_list = model_names_list

###################################     SUBREGIONS    #####################################################################

# This program allows to analyse results by subregions. Region types and names are handled by grids.subregions()
# # At the moment only EspañaPB is implemented, but adaptation to other regions (and their shapefiles) is prepared to be easy
# typeCompleteRegion = 'SPAIN' # It would make more sense to call it "COUNTRY", but it was designed as "SPAIN" for the website
typeCompleteRegion = 'COMPLETE'  # It would make more sense to call it "COUNTRY", but it was designed as "SPAIN" for the website
# nameCompleteRegion = 'EspañaPB' # Chose between EspañaPB or Canarias (Canarias to be implemented)
nameCompleteRegion = 'myRegionName'  # Chose between EspañaPB or Canarias (Canarias to be implemented)
divideByRegions = False  # Set to True if a division by regions will be done, and to False if no regions will be used
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


# ####################  CLIMDEX AND BIAS MODE    #####################################################
units_and_biasMode_climdex = {
    'tasmax_TXm': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmax_TX90p': {'units': 'days', 'biasMode': 'rel'},
    'tasmax_TX99p': {'units': 'days', 'biasMode': 'rel'},
    'tasmax_TX95p': {'units': 'days', 'biasMode': 'rel'},
    'tasmax_TX10p': {'units': 'days', 'biasMode': 'rel'},
    'tasmax_TX1p': {'units': 'days', 'biasMode': 'rel'},
    'tasmax_TX5p': {'units': 'days', 'biasMode': 'rel'},
    'tasmax_TXx': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmax_TXn': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmax_p99': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmax_p95': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmax_p90': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmax_p10': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmax_p5': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmax_p1': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmax_SU': {'units': 'days', 'biasMode': 'rel'},
    'tasmax_ID': {'units': 'days', 'biasMode': 'rel'},
    'tasmax_WSDI': {'units': 'days', 'biasMode': 'rel'},

    'tasmin_TNm': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmin_TN90p': {'units': 'days', 'biasMode': 'rel'},
    'tasmin_TN99p': {'units': 'days', 'biasMode': 'rel'},
    'tasmin_TN95p': {'units': 'days', 'biasMode': 'rel'},
    'tasmin_TN10p': {'units': 'days', 'biasMode': 'rel'},
    'tasmin_TN5p': {'units': 'days', 'biasMode': 'rel'},
    'tasmin_TN1p': {'units': 'days', 'biasMode': 'rel'},
    'tasmin_TNx': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmin_TNn': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmin_p99': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmin_p95': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmin_p90': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmin_p10': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmin_p5': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmin_p1': {'units': degree_sign, 'biasMode': 'abs'},
    'tasmin_FD': {'units': 'days', 'biasMode': 'rel'},
    'tasmin_TR': {'units': 'days', 'biasMode': 'rel'},
    'tasmin_CSDI': {'units': 'days', 'biasMode': 'rel'},

    'tas_Tm': {'units': degree_sign, 'biasMode': 'abs'},
    'tas_T90p': {'units': 'days', 'biasMode': 'rel'},
    'tas_T99p': {'units': 'days', 'biasMode': 'rel'},
    'tas_T95p': {'units': 'days', 'biasMode': 'rel'},
    'tas_T10p': {'units': 'days', 'biasMode': 'rel'},
    'tas_T5p': {'units': 'days', 'biasMode': 'rel'},
    'tas_T1p': {'units': 'days', 'biasMode': 'rel'},
    'tas_Tx': {'units': degree_sign, 'biasMode': 'abs'},
    'tas_Tn': {'units': degree_sign, 'biasMode': 'abs'},
    'tas_p99': {'units': degree_sign, 'biasMode': 'abs'},
    'tas_p95': {'units': degree_sign, 'biasMode': 'abs'},
    'tas_p90': {'units': degree_sign, 'biasMode': 'abs'},
    'tas_p10': {'units': degree_sign, 'biasMode': 'abs'},
    'tas_p5': {'units': degree_sign, 'biasMode': 'abs'},
    'tas_p1': {'units': degree_sign, 'biasMode': 'abs'},

    'pr_Pm': {'units': 'mm', 'biasMode': 'rel'},
    'pr_PRCPTOT': {'units': 'mm', 'biasMode': 'rel'},
    'pr_R01': {'units': 'days', 'biasMode': 'rel'},
    'pr_SDII': {'units': 'mm', 'biasMode': 'rel'},
    'pr_Rx1day': {'units': 'mm', 'biasMode': 'rel'},
    'pr_Rx5day': {'units': 'mm', 'biasMode': 'rel'},
    'pr_R10mm': {'units': 'days', 'biasMode': 'rel'},
    'pr_R20mm': {'units': 'days', 'biasMode': 'rel'},
    'pr_CDD': {'units': 'days', 'biasMode': 'rel'},
    'pr_p95': {'units': 'mm', 'biasMode': 'rel'},
    'pr_R95p': {'units': 'mm', 'biasMode': 'rel'},
    'pr_R95pFRAC': {'units': 'mm', 'biasMode': 'rel'},
    'pr_p99': {'units': 'mm', 'biasMode': 'rel'},
    'pr_R99p': {'units': 'mm', 'biasMode': 'rel'},
    'pr_R99pFRAC': {'units': 'mm', 'biasMode': 'rel'},
    'pr_CWD': {'units': 'days', 'biasMode': 'rel'},

    'uas_Um': {'units': 'm/s', 'biasMode': 'abs'},
    'uas_Ux': {'units': 'm/s', 'biasMode': 'abs'},

    'vas_Vm': {'units': 'm/s', 'biasMode': 'abs'},
    'vas_Vx': {'units': 'm/s', 'biasMode': 'abs'},

    'sfcWind_SFCWINDm': {'units': 'm/s', 'biasMode': 'abs'},
    'sfcWind_SFCWINDx': {'units': 'm/s', 'biasMode': 'abs'},

    'hurs_HRm': {'units': '%', 'biasMode': 'abs'},
    'hurs_p99': {'units': '%', 'biasMode': 'abs'},
    'hurs_p95': {'units': '%', 'biasMode': 'abs'},
    'hurs_p90': {'units': '%', 'biasMode': 'abs'},
    'hurs_p10': {'units': '%', 'biasMode': 'abs'},
    'hurs_p5': {'units': '%', 'biasMode': 'abs'},
    'hurs_p1': {'units': '%', 'biasMode': 'abs'},

    'clt_CLTm': {'units': '%', 'biasMode': 'abs'},
    'clt_p99': {'units': '%', 'biasMode': 'abs'},
    'clt_p95': {'units': '%', 'biasMode': 'abs'},
    'clt_p90': {'units': '%', 'biasMode': 'abs'},
    'clt_p10': {'units': '%', 'biasMode': 'abs'},
    'clt_p5': {'units': '%', 'biasMode': 'abs'},
    'clt_p1': {'units': '%', 'biasMode': 'abs'},
}
if myTargetVar != 'None':
    if myTargetVarIsAdditive == True:
        biasMode = 'abs'
        units = myTargetVarUnits
    else:
        biasMode = 'rel'
        units = '%'
    for climdex in climdex_names[myTargetVar]:
        units_and_biasMode_climdex.update({myTargetVar + '_' + climdex: {'units': units, 'biasMode': biasMode}})

# ####################  COLORS AND STYLES    #####################################################

methods_colors = {
    'RAW': 'lightgray',
    'RAW-BIL': 'lightgray',
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
    'ANA-VAR-1NN': 'darkred',
    'ANA-VAR-kNN': 'darkred',
    'ANA-VAR-rand': 'darkred',
    'MLR': 'cyan',
    'MLR-ANA': 'cyan',
    'MLR-WT': 'cyan',
    'GLM-LIN': 'cyan',
    'GLM-EXP': 'cyan',
    'GLM-CUB': 'cyan',
    'SVM': 'b',
    'LS-SVM': 'b',
    'RF': 'purple',
    'XGB': 'purple',
    'ANN-sklearn': 'magenta',
    'ANN': 'magenta',
    'CNN': 'magenta',
    'WG-PDF': 'g',
    'WG-NMM': 'g',
}

methods_linestyles = {
    'RAW': '-',
    'RAW-BIL': '--',
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
    'ANA-VAR-1NN': '-',
    'ANA-VAR-kNN': '--',
    'ANA-VAR-rand': ':',
    'MLR': '-',
    'MLR-ANA': '--',
    'MLR-WT': ':',
    'GLM-LIN': '-',
    'GLM-EXP': '--',
    'GLM-CUB': ':',
    'SVM': '-',
    'LS-SVM': '--',
    'RF': '-',
    'XGB': '--',
    'ANN-sklearn': '-',
    'ANN': '--',
    'CNN': ':',
    'WG-PDF': '-',
    'WG-NMM': '--',
}
