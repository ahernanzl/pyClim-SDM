showWelcomeMessage = True
experiment = 'EVALUATION'
steps = []
methods = [{'var': 'tmax', 'methodName': 'QM', 'family': 'BC', 'mode': 'MOS', 'fields': 'var'}, {'var': 'tmax', 'methodName': 'ANA-MLR', 'family': 'ANA', 'mode': 'PP', 'fields': 'pred+saf'}, {'var': 'tmax', 'methodName': 'MLR', 'family': 'TF', 'mode': 'PP', 'fields': 'pred'}, {'var': 'tmax', 'methodName': 'WG-PDF', 'family': 'WG', 'mode': 'WG', 'fields': 'var'}, {'var': 'pcp', 'methodName': 'PSDM', 'family': 'BC', 'mode': 'MOS', 'fields': 'var'}, {'var': 'pcp', 'methodName': 'ANA-SYN-1NN', 'family': 'ANA', 'mode': 'PP', 'fields': 'saf'}, {'var': 'pcp', 'methodName': 'GLM-LIN', 'family': 'TF', 'mode': 'PP', 'fields': 'pred'}, {'var': 'pcp', 'methodName': 'WG-NMM', 'family': 'WG', 'mode': 'WG', 'fields': 'var'}]
reaNames = {'u': 'u', 'v': 'v', 't': 't', 'z': 'z', 'q': 'q', 'mslp': 'msl', 'u10': 'u10', 'v10': 'v10', 't2m': 't2m', 'tmax': 'mx2t', 'tmin': 'mn2t', 'pcp': 'tp'}
modNames = {'u': 'ua', 'v': 'va', 't': 'ta', 'z': 'zg', 'q': 'hus', 'mslp': 'psl', 'u10': 'uas', 'v10': 'vas', 't2m': 'tas', 'tmax': 'tasmax', 'tmin': 'tasmin', 'pcp': 'pr'}
preds_t_list = ['tmax']
preds_p_list = ['mslp']
saf_list = ['mslp']
calibration_years = (1979, 2020)
reference_years = (1979, 2005)
historical_years = (1979, 2005)
ssp_years = (2015, 2100)
biasCorr_years = (1979, 2005)
bc_method = None
single_split_testing_years = (2006, 2020)
fold1_testing_years = (1979, 1987)
fold2_testing_years = (1988, 1996)
fold3_testing_years = (1997, 2005)
fold4_testing_years = (2006, 2014)
fold5_testing_years = (2015, 2020)
hresPeriodFilename = '19510101-20201231'
reanalysisName = 'ERA5'
reanalysisPeriodFilename = '19790101-20201231'
historicalPeriodFilename = '19500101-20141231'
rcpPeriodFilename = '20150101-21001231'
split_mode = 'single_split'
grid_res = 1.5
saf_lat_up = 49.0
saf_lon_left = -18.0
saf_lon_right = 12.0
saf_lat_down = 29.5
model_names_list = ['ACCESS-CM2', 'CanESM5', 'EC-Earth3', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0']
scene_names_list = ['HISTORICAL', 'SSP5-8.5']
modelRealizationFilename = 'r1i1p1f1'
climdex_names = {'tmax': ['TXm', 'TXx', 'TXn'], 'tmin': ['TNm', 'TNx', 'TNn'], 'pcp': ['PRCPTOT', 'R01', 'SDII', 'R95p']}
