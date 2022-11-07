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

# ########################################  MULTIPROCESSING   ##########################################################
nCPUs_multiprocessing = 1
if running_at_HPC == True or nCPUs_multiprocessing == 1:
    runInParallel_multiprocessing = False
else:
    runInParallel_multiprocessing = True


# ########################################  RUNNING OPTIONS   ##########################################################
# Predictands preprarataion. When working with stations, files with all stations have to be previously prepared.
# All stations should have the same number of data and have to be between the ranges indicated below.
# When building these files, no-data must be set to "special_value". Note thant pcp and temp scpecial_values differ.
# If min/max range is surpased, some adaptations must be made (change uint16 for uin32 for pcp, for example, which
# needs more memory

# This parameter controls the interpolation used for predictors
# interp_mode = 'nearest'
interp_mode = 'bilinear'


###################################     predictands           #################################################
all_possible_targetVars = [
    'tasmax', 'tasmin', 'tas',
    'pr',
    'uas', 'vas', 'sfcWind',
    'hurs', 'huss',
    'clt', 'rsds', 'rlds',
    'evspsbl', 'evspsblpot',
    'psl', 'ps',
    'mrro', 'mrso',
    ]

###################################     myTargetVar           #################################################
if 'myTargetVar' in targetVars:
    myTargetVar = myTargetVarName

    targetVars.remove('myTargetVar')
    targetVars.append(myTargetVar)

    # Define myTargetVar reaNames and modNames
    try:
        reaNames.update({myTargetVar: myTargetVar})
    except:
        pass
    try:
        modNames.update({myTargetVar: myTargetVar})
    except:
        pass
    try:
        methods[myTargetVar] = methods.pop('myTargetVar')
    except:
        pass
    try:
        preds_targetVars_dict[myTargetVar] = preds_targetVars_dict.pop('myTargetVar')
    except:
        pass
    try:
        climdex_names[myTargetVar] = climdex_names.pop('myTargetVar')
    except:
        pass
    # climdex_names[myTargetVar] = [x.replace('MYTARGETVAR', myTargetVar.upper()) for x in climdex_names[myTargetVar]]

    # Define whether myTargetVar can be treated as gaussian
    myTargetVarIsGaussian = False
    # Define whether apply bias correction as for precipitation (multiplicative correction)
    treatAsAdditiveBy_DQM_and_QDM = myTargetVarIsAdditive
else:
    myTargetVar = 'None'


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
    'huss': {'type': 'uint16', 'min_valid': 0, 'max_valid': 655.34, 'special_value': 655.35},
    'clt': {'type': 'uint16', 'min_valid': 0, 'max_valid': 655.34, 'special_value': 655.35},
    'rsds': {'type': 'uint32', 'min_valid': 0, 'max_valid': 42949672.94, 'special_value': 42949672.95},
    'rlds': {'type': 'uint32', 'min_valid': 0, 'max_valid': 42949672.94, 'special_value': 42949672.95},
    'evspsbl': {'type': 'uint16', 'min_valid': 0, 'max_valid': 655.34, 'special_value': 655.35},
    'evspsblpot': {'type': 'uint16', 'min_valid': 0, 'max_valid': 655.34, 'special_value': 655.35},
    'psl': {'type': 'uint32', 'min_valid': 0, 'max_valid': 42949672.94, 'special_value': 42949672.95},
    'ps': {'type': 'uint32', 'min_valid': 0, 'max_valid': 42949672.94, 'special_value': 42949672.95},
    'mrro': {'type': 'uint16', 'min_valid': 0, 'max_valid': 655.34, 'special_value': 655.35},
    'mrso': {'type': 'uint16', 'min_valid': 0, 'max_valid': 655.34, 'special_value': 655.35},
}
if myTargetVar in targetVars:
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
    'huss': {'min': 0, 'max': None},
    'clt': {'min': 0, 'max': 100},
    'rsds': {'min': 0, 'max': None},
    'rlds': {'min': 0, 'max': None},
    'evspsbl': {'min': 0, 'max': None},
    'evspsblpot': {'min': 0, 'max': None},
    'psl': {'min': 0, 'max': None},
    'ps': {'min': 0, 'max': None},
    'mrro': {'min': 0, 'max': None},
    'mrso': {'min': 0, 'max': None},
}
if myTargetVar in targetVars:
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
    'huss': '1',
    'clt': '%',
    'rsds': 'W/m2',
    'rlds': 'W/m2',
    'evspsbl': 'kg m-2 s-1',
    'evspsblpot': 'kg m-2 s-1',
    'psl': 'Pa',
    'ps': 'Pa',
    'mrro': 'kg m-2 s-1',
    'mrso': 'kg m-2',
}
if myTargetVar in targetVars:
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

# Force calculations even if files already exist
force_downscaling = False
force_climdex_calculation = False
force_bias_correction = False

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
    'huss': 0.7,
    'clt': 0.7,
    'rsds': 0.7,
    'rlds': 0.7,
    'evspsbl': 0.7,
    'evspsblpot': 0.7,
    'psl': 0.7,
    'ps': 0.7,
    'mrro': 0.7,
    'mrso': 0.7,
}
if myTargetVar in targetVars:
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

# The following Transfer Function methods will be replaced by a MLR where predictos lie out of the training range
# This is done for all targetVars except for precipitation
methods_to_extrapolate_with_MLR = ['RF', 'XGB', ]
for methodName in methods_using_preds_from_whole_grid:
    if methodName in methods_to_extrapolate_with_MLR:
        print('Remove', methodName, 'from methods_using_preds_from_whole_grid or from'
            'methods_to_extrapolate_with_MLR at advanced_settings.\nBoth options are not compatible.')
        exit()

# Certain climdex make use of a reference period which can correspond to observations or to the proper method/model.
# That is the case of TX10p, R95p, etc. When evaluating ESD methods, set to True, but when studying change on the
# climdex (projections), set to False. This parameter is used in postporcess.get_climdex_oneModel
if experiment in ('EVALUATION', 'PSEUDOREALITY'):
    reference_climatology_from_observations = True
elif experiment == 'PROJECTIONS':
    reference_climatology_from_observations = False


# Controls the path name for bias corrected outputs
if apply_bc_bySeason == True:
    apply_bc = True
if apply_bc == False:
    bc_sufix = ''
else:
    bc_sufix = '-BC-' + bc_method
    if apply_bc_bySeason == True:
        bc_sufix += '-s'

########################################       DATES      ##############################################################
# Definition of testing_years and historical_years depending on the experiment (do not change)
nyears = calibration_years[1]-calibration_years[0]+1
block = nyears//5
rest = nyears%5
blocks = [block, block, block, block, block, ]
for i in range(rest):
    blocks[i] += 1
first_years = [calibration_years[0],]
for i in range(4):
    first_years.append(first_years[i]+blocks[i])

fold1_testing_years = (first_years[0], first_years[0]+blocks[0]-1)
fold2_testing_years = (first_years[1], first_years[1]+blocks[1]-1)
fold3_testing_years = (first_years[2], first_years[2]+blocks[2]-1)
fold4_testing_years = (first_years[3], first_years[3]+blocks[3]-1)
fold5_testing_years = (first_years[4], first_years[4]+blocks[4]-1)

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

# Detect reanalysisPeriodFilename
reanalysisPeriodFilenames = []
for file in os.listdir('../input_data/reanalysis/'):
    if file.endswith(".nc"):
        aux = file.split('_')[-1].split('.')[0]
        if aux[4:8] != '0101' or aux[-4:] != '1231':
            print('Invalid filename at input_data/reanalysis/: ' + file)
            print('Please, modify filename')
            exit()
        else:
            if aux not in reanalysisPeriodFilenames:
                reanalysisPeriodFilenames.append(aux)
if len(reanalysisPeriodFilenames) > 1:
    print('Different periods detected at input_data/reanalysis/: ' + reanalysisPeriodFilenames)
    print('Please, modify filenames so all files contain the same period.')
    exit()
elif len(reanalysisPeriodFilenames) == 1:
    reanalysisPeriodFilename = reanalysisPeriodFilenames[0]


# Detect historicalPeriodFilename, sspPeriodFilename, historical_years  and ssp_years
historicalPeriodFilenames = []
sspPeriodFilenames = []
if os.path.exists('../input_data/models/'):
    for file in os.listdir('../input_data/models/'):
        if file.endswith(".nc"):
            aux = file.split('_')[-1].split('.')[0]
            if aux[4:8] != '0101' or aux[-4:] != '1231':
                print('Invalid filename at input_data/models/: ' + file)
                print('Please, modify filename')
                exit()
            else:
                if 'historical' in file:
                    if aux not in historicalPeriodFilenames:
                        historicalPeriodFilenames.append(aux)
                else:
                    if aux not in sspPeriodFilenames:
                        sspPeriodFilenames.append(aux)
if len(historicalPeriodFilenames) > 1:
    print('Different periods detected at input_data/models/: ' + historicalPeriodFilenames)
    print('Please, modify filenames so all files contain the same period.')
    exit()
elif len(historicalPeriodFilenames) == 1:
    historicalPeriodFilename = historicalPeriodFilenames[0]
    historical_years = (int(historicalPeriodFilenames[0][:4]), int(historicalPeriodFilenames[0][9:13]))
elif len(historicalPeriodFilenames) == 0:
    historicalPeriodFilename = ''
    historical_years = (1950, 2014)

if len(sspPeriodFilenames) > 1:
    print('Different periods detected at input_data/models/: ' + sspPeriodFilenames)
    print('Please, modify filenames so all files contain the same period.')
    exit()
elif len(sspPeriodFilenames) == 1:
    sspPeriodFilename = sspPeriodFilenames[0]
    ssp_years = (int(sspPeriodFilenames[0][:4]), int(sspPeriodFilenames[0][9:13]))
elif len(sspPeriodFilenames) == 0:
    sspPeriodFilename = ''
    ssp_years = (2015, 2100)

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
methods = [x for x in methods if x['var'] in targetVars]
del methods_list

if myTargetVar in targetVars:
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

# # Check for consistency between predictors and methods
# for var in all_possible_targetVars:
#     if (var in targetVars) and (len(preds_targetVars_dict[var]) == 0) and (experiment != 'PRECONTROL'):
#         print('-----------------------------------------------')
#         print('Inconsistency found between preditors and methods selection.')
#         print('Your selection includes some methods for ' + var + ' but no predictor has been selected')
#         print('-----------------------------------------------')
#         exit()

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

aux = []
hresPeriodFilename = {}
for targetVar in targetVars:
    if os.path.isfile(pathHres + targetVar + '_hres_metadata.txt'):
        files_with_data = []
        for file in os.listdir(pathHres):
            if file.endswith(".txt") and file.startswith(targetVar) and file!=targetVar + '_hres_metadata.txt':
                newHresPeriodFilename = file.replace(targetVar, '').replace('_', '').replace('.txt', '')
                hresPeriodFilename.update({targetVar: newHresPeriodFilename})
                files_with_data.append(file)
        if len(files_with_data) > 1:
            print('------------------------------------------------------------------------------------------------')
            print('ERROR:', len(files_with_data),'files have been found at input_data/hres/ containing data for:', targetVar)
            print('Please, remove all files except one from the following list:', files_with_data)
            print('------------------------------------------------------------------------------------------------')
            exit()
        if targetVar not in aux:
            aux.append(targetVar)
targetVars = aux

for targetVar in targetVars:
    aux_hres_metadata = np.loadtxt(pathHres + targetVar + '_hres_metadata.txt')
    hres_npoints.update({targetVar: aux_hres_metadata.shape[0]})
    hres_lats.update({targetVar: aux_hres_metadata[:, 2]})
    hres_lons.update({targetVar: aux_hres_metadata[:, 1]})


hres_lats_all, hres_lons_all = [], []
for targetVar in all_possible_targetVars:
    try:
        aux_hres_metadata = np.loadtxt(pathHres + targetVar + '_hres_metadata.txt')
        for i in range(len(aux_hres_metadata[:, 2])):
            hres_lats_all.append(aux_hres_metadata[i, 2])
            hres_lons_all.append(aux_hres_metadata[i, 1])
    except:
        pass
hres_lats_all = np.asarray(hres_lats_all)
hres_lons_all = np.asarray(hres_lons_all)

# Modify saf_lat_up, saf_lat_down, saf_lon_left and saf_lon_right forcing to exist in the netCDF files
for file in os.listdir('../input_data/reanalysis/'):
    if file.endswith(".nc"):
        try:
            nc = Dataset('../input_data/reanalysis/'+file)
            break
        except:
            pass
if 'nc' not in locals():
    print('No netCDF file detected to extract lat/lon')
    print('Check the input_data/reanalysis/ folder')
    exit()

if 'lat' in nc.variables:
    lat_name, lon_name = 'lat', 'lon'
elif 'latitude' in nc.variables:
    lat_name, lon_name = 'latitude', 'longitude'
lats = nc.variables[lat_name][:]
grid_res = abs(lats[0]-lats[1])
lons = nc.variables[lon_name][:]
lons[lons > 180] -= 360
if len(hres_lats_all) == 0:
    print('Make sure there are files at input_data/hres/')
    exit()
hres_max_lat, hres_min_lat, hres_max_lon, hres_min_lon = np.max(hres_lats_all), np.min(hres_lats_all), np.max(hres_lons_all), np.min(hres_lons_all)
lres_max_lat, lres_min_lat, lres_max_lon, lres_min_lon = np.max(lats), np.min(lats), np.max(lons), np.min(lons)

# Check if hres domain is fully contained by the low resolution grid from the netCDFs
if lres_max_lat < hres_max_lat or lres_min_lat > hres_min_lat or lres_max_lon < hres_max_lon or lres_min_lon > hres_min_lon:
    print('Domain at netCDF files do not fully contain domain at hres files. Please check your input files.')
    exit()
# Check if hres domain is fully contained by the low resolution grid from the netCDFs, after saving a one gridbox border
elif lres_max_lat-grid_res < hres_max_lat or lres_min_lat+grid_res > hres_min_lat or lres_max_lon-grid_res < hres_max_lon or lres_min_lon+grid_res > hres_min_lon:
    print('Domain at netCDF files not large enough. At least one-gridbox border is needed around the domain at hres files.')
    print('Prepare yout netCDF files covering a larger spatial domain')
    exit()
# Force synoptic domain to be compatible with spatial domains at hres and netCDF files
else:
    possible_saf_lat_up_list = [x for x in lats[1:] if x >= hres_max_lat]
    possible_saf_lat_down_list = [x for x in lats[:-1] if x <= hres_min_lat]
    possible_saf_lon_left_list = [x for x in lons[1:] if x <= hres_min_lon]
    possible_saf_lon_right_list = [x for x in lons[:-1] if x >= hres_max_lon]
    if saf_lat_up not in possible_saf_lat_up_list:
        old = saf_lat_up
        saf_lat_up = min(possible_saf_lat_up_list, key=lambda x:abs(x-saf_lat_up))
        print('Synoptic domain incompatible with coordinates in input files. saf_lat_up forced from', old, 'to', saf_lat_up)
        time.sleep(1)
    if saf_lat_down not in possible_saf_lat_down_list:
        old = saf_lat_down
        saf_lat_down = min(possible_saf_lat_down_list, key=lambda x:abs(x-saf_lat_down))
        print('Synoptic domain incompatible with coordinates in input files. saf_lat_down forced from', old, 'to', saf_lat_down)
        time.sleep(1)
    if saf_lon_left not in possible_saf_lon_left_list:
        old = saf_lon_left
        saf_lon_left = min(possible_saf_lon_left_list, key=lambda x:abs(x-saf_lon_left))
        print('Synoptic domain incompatible with coordinates in input files. saf_lon_left forced from', old, 'to', saf_lon_left)
        time.sleep(1)
    if saf_lon_right not in possible_saf_lon_right_list:
        old = saf_lon_right
        saf_lon_right = min(possible_saf_lon_right_list, key=lambda x:abs(x-saf_lon_right))
        print('Synoptic domain incompatible with coordinates in input files. saf_lon_right forced from', old, 'to', saf_lon_right)
        time.sleep(1)

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

    'huss_HUSSm': {'units': '%', 'biasMode': 'rel'},
    'huss_HUSSx': {'units': '%', 'biasMode': 'rel'},
    'huss_HUSSn': {'units': '%', 'biasMode': 'rel'},
    'huss_p99': {'units': '%', 'biasMode': 'rel'},
    'huss_p95': {'units': '%', 'biasMode': 'rel'},
    'huss_p90': {'units': '%', 'biasMode': 'rel'},
    'huss_p10': {'units': '%', 'biasMode': 'rel'},
    'huss_p5': {'units': '%', 'biasMode': 'rel'},
    'huss_p1': {'units': '%', 'biasMode': 'rel'},

    'clt_CLTm': {'units': '%', 'biasMode': 'abs'},
    'clt_p99': {'units': '%', 'biasMode': 'abs'},
    'clt_p95': {'units': '%', 'biasMode': 'abs'},
    'clt_p90': {'units': '%', 'biasMode': 'abs'},
    'clt_p10': {'units': '%', 'biasMode': 'abs'},
    'clt_p5': {'units': '%', 'biasMode': 'abs'},
    'clt_p1': {'units': '%', 'biasMode': 'abs'},

    'rsds_RSDSm': {'units': '%', 'biasMode': 'rel'},
    'rsds_RSDSx': {'units': '%', 'biasMode': 'rel'},
    'rsds_RSDSn': {'units': '%', 'biasMode': 'rel'},
    'rsds_p99': {'units': '%', 'biasMode': 'rel'},
    'rsds_p95': {'units': '%', 'biasMode': 'rel'},
    'rsds_p90': {'units': '%', 'biasMode': 'rel'},
    'rsds_p10': {'units': '%', 'biasMode': 'rel'},
    'rsds_p5': {'units': '%', 'biasMode': 'rel'},
    'rsds_p1': {'units': '%', 'biasMode': 'rel'},

    'rlds_RLDSm': {'units': '%', 'biasMode': 'rel'},
    'rlds_RLDSx': {'units': '%', 'biasMode': 'rel'},
    'rlds_RLDSn': {'units': '%', 'biasMode': 'rel'},
    'rlds_p99': {'units': '%', 'biasMode': 'rel'},
    'rlds_p95': {'units': '%', 'biasMode': 'rel'},
    'rlds_p90': {'units': '%', 'biasMode': 'rel'},
    'rlds_p10': {'units': '%', 'biasMode': 'rel'},
    'rlds_p5': {'units': '%', 'biasMode': 'rel'},
    'rlds_p1': {'units': '%', 'biasMode': 'rel'},

    'e_Em': {'units': '%', 'biasMode': 'rel'},
    'e_Ex': {'units': '%', 'biasMode': 'rel'},
    'e_En': {'units': '%', 'biasMode': 'rel'},
    'e_p99': {'units': '%', 'biasMode': 'rel'},
    'e_p95': {'units': '%', 'biasMode': 'rel'},
    'e_p90': {'units': '%', 'biasMode': 'rel'},
    'e_p10': {'units': '%', 'biasMode': 'rel'},
    'e_p5': {'units': '%', 'biasMode': 'rel'},
    'e_p1': {'units': '%', 'biasMode': 'rel'},

    'ep_EPm': {'units': '%', 'biasMode': 'rel'},
    'ep_EPx': {'units': '%', 'biasMode': 'rel'},
    'ep_EPn': {'units': '%', 'biasMode': 'rel'},
    'ep_p99': {'units': '%', 'biasMode': 'rel'},
    'ep_p95': {'units': '%', 'biasMode': 'rel'},
    'ep_p90': {'units': '%', 'biasMode': 'rel'},
    'ep_p10': {'units': '%', 'biasMode': 'rel'},
    'ep_p5': {'units': '%', 'biasMode': 'rel'},
    'ep_p1': {'units': '%', 'biasMode': 'rel'},

    'psl_PSLm': {'units': '%', 'biasMode': 'rel'},
    'psl_PSLx': {'units': '%', 'biasMode': 'rel'},
    'psl_PSLn': {'units': '%', 'biasMode': 'rel'},
    'psl_p99': {'units': '%', 'biasMode': 'rel'},
    'psl_p95': {'units': '%', 'biasMode': 'rel'},
    'psl_p90': {'units': '%', 'biasMode': 'rel'},
    'psl_p10': {'units': '%', 'biasMode': 'rel'},
    'psl_p5': {'units': '%', 'biasMode': 'rel'},
    'psl_p1': {'units': '%', 'biasMode': 'rel'},

    'ps_PSm': {'units': '%', 'biasMode': 'rel'},
    'ps_PSx': {'units': '%', 'biasMode': 'rel'},
    'ps_PSn': {'units': '%', 'biasMode': 'rel'},
    'ps_p99': {'units': '%', 'biasMode': 'rel'},
    'ps_p95': {'units': '%', 'biasMode': 'rel'},
    'ps_p90': {'units': '%', 'biasMode': 'rel'},
    'ps_p10': {'units': '%', 'biasMode': 'rel'},
    'ps_p5': {'units': '%', 'biasMode': 'rel'},
    'ps_p1': {'units': '%', 'biasMode': 'rel'},

    'mrro_RUNOFFm': {'units': '%', 'biasMode': 'rel'},
    'mrro_RUNOFFx': {'units': '%', 'biasMode': 'rel'},
    'mrro_RUNOFFn': {'units': '%', 'biasMode': 'rel'},
    'mrro_p99': {'units': '%', 'biasMode': 'rel'},
    'mrro_p95': {'units': '%', 'biasMode': 'rel'},
    'mrro_p90': {'units': '%', 'biasMode': 'rel'},
    'mrro_p10': {'units': '%', 'biasMode': 'rel'},
    'mrro_p5': {'units': '%', 'biasMode': 'rel'},
    'mrro_p1': {'units': '%', 'biasMode': 'rel'},

    'mrso_SOILMOISTUREm': {'units': '%', 'biasMode': 'rel'},
    'mrso_SOILMOISTUREx': {'units': '%', 'biasMode': 'rel'},
    'mrso_SOILMOISTUREn': {'units': '%', 'biasMode': 'rel'},
    'mrso_p99': {'units': '%', 'biasMode': 'rel'},
    'mrso_p95': {'units': '%', 'biasMode': 'rel'},
    'mrso_p90': {'units': '%', 'biasMode': 'rel'},
    'mrso_p10': {'units': '%', 'biasMode': 'rel'},
    'mrso_p5': {'units': '%', 'biasMode': 'rel'},
    'mrso_p1': {'units': '%', 'biasMode': 'rel'},
}
if myTargetVar in targetVars:
    if myTargetVarIsAdditive == True:
        biasMode = 'abs'
        units = myTargetVarUnits
    else:
        biasMode = 'rel'
        units = '%'
    for climdex in climdex_names[myTargetVar]:
        if climdex[-1] == 'p':
            newUnits = 'days'
        else:
            newUnits = units
        units_and_biasMode_climdex.update({myTargetVar + '_' + climdex: {'units': newUnits, 'biasMode': biasMode}})

######################################## BIAS CORRECTION MODE ##########################################################
# Two bias correction / MOS methods, DQM and QDM, can be applied as an additive or multiplicative correction
# The following parameter controls how to apply them for each targetVar ('abs' / 'rel')
bc_mode_dict = {
    'tasmax': 'abs',
    'tasmin': 'abs',
    'tas': 'abs',
    'pr': 'abs',
    'uas': 'abs',
    'vas': 'abs',
    'sfcWind': 'abs',
    'hurs': 'abs',
    'huss': 'abs',
    'clt': 'abs',
    'rsds': 'abs',
    'rlds': 'abs',
    'evspsbl': 'abs',
    'evspsblpot': 'abs',
    'psl': 'abs',
    'ps': 'abs',
    'mrro': 'abs',
    'mrso': 'abs',
}
if myTargetVar in targetVars:
    bc_mode_dict.update({myTargetVar: 'abs'})


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

