import sys
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import BC_lib
import down_scene_ANA
import down_scene_BC
import down_scene_RAW
import down_scene_TF
import down_scene_WG
import down_day
import down_point
import evaluate_methods
import grids
import gui_lib
import launch_jobs
import plot
import postpro_lib
import postprocess
import derived_predictors
import precontrol
import preprocess
import process
import read
import standardization
import TF_lib
import val_lib
import WG_lib
import write

#########################################################################################################################
def unitplot(input_var):
    """
    Assigning units to variables
    """
    if input_var in ['tmax','tmin']:
        unit = input_var + ' ' + '(°C)'
    elif input_var in ['t2m','t1000','t850','t700','t500','t250','td1000','td850','td700','td500','td250','Dtd1000','Dtd850',
                       'Dtd700','Dtd500','Dtd250','Dvtg_1000_850','Dvtg_850_700','Dvtg_700_500']:
        unit = input_var + ' ' + '(K)'
    elif input_var in ['u1000','u850','u700','u500','u250','v1000','v850','v700','v500','v250','u10','v10','ugsl','vgsl']:
        unit = input_var + ' ' + '(m/s)'
    elif input_var in ['pcp']:
        unit = input_var + ' ' + '(mm)'
    elif input_var in ['mslp','mslp_trend','z1000','z850','z700','z500','z250']:
        unit = input_var + ' ' + '(Pa)'
    elif input_var in ['ins']:
        unit = input_var + ' ' + '( )'
    elif input_var in ['vort1000','vort850','vort700','vort500','vort250','div1000','div850','div700','div500','div250',
                       'vortgsl','divgsl']:
        unit = input_var + ' ' + '(s-1)'
    elif input_var in ['r1000','r850','r700','r500','r250']:
        unit = input_var + ' ' + '(%)'
    elif input_var in ['q1000','q850','q700','q500','q250']:
        unit = input_var + ' ' + '(kg*kg-1)' 
    return (unit)

########################################################################################################################
def missing_data_check():
    """
    Check for missing data in predictors by GCMs. It can be used to discard some predictors/levels.
    """

    print('missing_data_check...')

    # Go through all target variables
    for var0 in target_vars0:

        # Define pathTmp
        pathTmp = '../results/' + experiment + '/missing_data_check/' + var0.upper() + '/'
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        pathOut = pathFigures
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Define preds
        preds = preds_dict[var0]
        npreds = len(preds)
        nscenes = len(scene_list)
        nmodels = len(model_list)
        nlats = pred_nlats
        nlons = pred_nlons
        PERC_NAN = np.zeros((npreds, nscenes, nmodels, nlats, nlons))
        
        
        # Go through all predictors, scenes and models
        for ipred in range(npreds):
            predName = list(preds.keys())[ipred]
            iscene = 0
            for sceneName in scene_list:
                imodel = 0
                for model in model_list:

                    # Read data
                    data = read.lres_data(var0, 'pred', model=model, scene=sceneName, predName=predName)['data']

                    # Calculate percentaje of nans
                    perc_nan = 100 * np.count_nonzero(np.isnan(data), axis=0)[0] / data.shape[0]
                    PERC_NAN[ipred, iscene, imodel] = perc_nan
                    print(var0, predName, sceneName, model, 'perc_nan', np.max(perc_nan))

                    if np.max(perc_nan) != 0:
                        # Plot map
                        filename = '_'.join((experiment, 'nansMap', var0, predName, model+'-'+sceneName, 'None'))
                        title = ' '.join((predName, model, sceneName, 'pertentage of NANs'))
                        plot.map(var0, perc_nan, 'perc_nan', grid='pred', path=pathOut, filename=filename, title=title)

                    imodel += 1
                iscene += 1

        # Save results
        np.save(pathTmp+'PERC_NAN', PERC_NAN)
        PERC_NAN = np.load(pathTmp+'PERC_NAN.npy')

        # Plot heatmaps
        nscenes = len(scene_names_list)
        predNames = [list(preds.keys())[i] for i in range(npreds)]
        modelNames = model_names_list

        # Define colors and units for heatmap
        cmap = 'RdYlGn_r'
        bounds = [0, .01, .1, 1, 2, 5, 10, 20, 50, 100]
        units = '%'
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        cmap = plt.get_cmap(cmap)

        # Go through all scenes
        for iscene in range(nscenes):
            sceneName = scene_names_list[iscene]
            matrix = np.mean(PERC_NAN[:, iscene, :].reshape(npreds, nmodels, -1), axis=2).T

            xticklabels = predNames
            g = sns.heatmap(matrix, annot=True, vmin=0, vmax=100, fmt='.2f',
                            cmap=cmap, norm=norm, cbar_kws={'label': units, 'ticks': bounds,
                                                            # 'format':'%.2f%%'
                                                            },
                            xticklabels=xticklabels, yticklabels=True, square=True)
            g.tick_params(left=False, bottom=False)
            g.yaxis.set_label_position("right")
            g.set_yticklabels(modelNames, rotation=0, fontsize=5)
            g.set_xticklabels(predNames, rotation=90)
            plt.title(sceneName + ' pertentage of NANs')
            # plt.show()
            # exit()
            filename = '_'.join((experiment, 'nansMatrix', var0, 'None', sceneName, 'None.png'))
            plt.savefig(pathOut + filename)
            plt.close()



########################################################################################################################
def predictors_correlation():
    """
    Test the correlation of the predictors/predictand relationships. It can be used to select the most relevant predictors
    for the downscaling.
    """

    print('predictors_correlation...')


    # Go through all target variables
    for var in ('tmax', 'tmin', 'pcp'):
        if var[0] in target_vars0:
            # Define pathTmp
            pathTmp = '../results/' + experiment + '/predictors_correlation/' + var.upper() + '/'
            if not os.path.exists(pathTmp):
                os.makedirs(pathTmp)
            pathOut = pathFigures
            if not os.path.exists(pathOut):
                os.makedirs(pathOut)

            # For interpolation
            interp_mode = 'bilinear'
            i_4nn = np.load(pathAux + 'ASSOCIATION/' + var[0].upper() + '_' + interp_mode + '/i_4nn.npy')
            j_4nn = np.load(pathAux + 'ASSOCIATION/' + var[0].upper() + '_' + interp_mode + '/j_4nn.npy')
            w_4nn = np.load(pathAux + 'ASSOCIATION/' + var[0].upper() + '_' + interp_mode + '/w_4nn.npy')

            # Read data predictand
            obs = read.hres_data(var, period='calibration')['data']

            # Define preds
            preds = preds_dict[var[0]]
            npreds = len(preds)

            # Go through all seasons
            for season in season_dict.values():

                R = np.zeros((npreds, hres_npoints[var[0]]))

                # Calculate correlations for each predictor
                for ipred in range(npreds):
                    predName = list(preds.keys())[ipred]

                    # Read data predictor
                    data = read.lres_data(var, 'pred', predName=predName)['data']

                    # Select season
                    data_season = postpro_lib.get_season(data, calibration_dates, season)['data']
                    obs_season = postpro_lib.get_season(obs, calibration_dates, season)['data']

                    # Go through all points
                    for ipoint in range(hres_npoints[var[0]]):

                        # Interpolate to one point
                        X = grids.interpolate_predictors(data_season, i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_mode)[:, 0]
                        y = obs_season[:, ipoint]

                        # Calculate correlation
                        if var == 'pcp':
                            R[ipred, ipoint] = spearmanr(X, y)[0]
                        else:
                            R[ipred, ipoint] = pearsonr(X, y)[0]

                    # Save results
                    print(var, predName, 'correlation', season, np.mean(abs(R[ipred])))
                    np.save(pathTmp + '_'.join((predName, 'correlation', season)), R[ipred])

                # Plot correlation maps and boxplots
                for ipred in range(npreds):
                    predName = list(preds.keys())[ipred]

                    # Load correlation
                    R[ipred] = np.load(pathTmp + '_'.join((predName, 'correlation', season+'.npy')))
                    print(var, predName, 'correlation', season, np.mean(abs(R[ipred])))

                    # Plot map
                    title = ' '.join((var.upper(), predName, 'correlation', season))
                    filename = '_'.join((experiment, 'correlationMap', var, predName, 'None', season))
                    plot.map(var[0], abs(R[ipred]), 'correlation', path=pathOut, filename=filename, title=title)

                # Boxplot
                fig, ax = plt.subplots()
                ax.boxplot(abs(R.T), showfliers=False)
                ax.set_xticklabels(list(preds.keys()), rotation=90)
                plt.ylim((0, 1))
                plt.title(' '.join((var.upper(), 'correlation', season)))
                # plt.show()
                # exit()
                filename = '_'.join((experiment, 'correlationBoxplot', var, 'None', 'None', season))
                plt.savefig(pathOut + filename)






########################################################################################################################
def GCMs_evaluation_historical():
    """
    Test the reliability of GCMs in a historical period comparing them with a reanalysis, analysing all predictors,
    models, synoptic analogy fields... It can be used to discard models/predictors and to detect outliers.
    The comparison is performed in the reference period, when both reanalysis and models exist.
    """
    
    print('GCMs_evaluation_historical...')
    sceneName = 'historical'

    # Go through all target variables
    for var0 in ('t', 'p', ):

        # Define pathTmp
        pathTmp = '../results/' + experiment + '/GCMs_evaluation_historical/' + var0.upper() + '/'
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        pathOut = pathFigures
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Define preds
        preds = preds_dict[var0]
        npreds = len(preds)
        nmodels = len(model_list)
        nlats = pred_nlats
        nlons = pred_nlons
        
        
        
        # Go through all seasons
        for season in season_dict.values():

            # Go through all predictors
            for ipred in range(npreds):
                predName = list(preds.keys())[ipred]
                print(var0, season, predName)

                # Read reanalysis
                rea = read.lres_data(var0, 'pred', predName=predName)['data']
                time_first, time_last = calibration_dates.index(reference_first_date), calibration_dates.index(
                    reference_last_date) + 1
                rea = rea[time_first:time_last]
                rea_dates = read.lres_data(var0, 'pred', predName=predName)['times']
                rea_dates = rea_dates[time_first:time_last]
                
                # Go through all models
                lst_sceneData_mean_season = []
                
                for imodel in range(nmodels):
                    model = model_list[imodel]

                    # Read model
                    if var0 == 'p':
                        ncVar = modNames['pcp']
                    else:
                        ncVar = modNames['tmax']
                        
                    modelName, modelRun = model.split('_')[0], model.split('_')[1]
                    calendar = read.netCDF('../input_data/models/',
                                           ncVar + '_' + modelName + '_' + sceneName + '_' + modelRun + '_' + historicalPeriodFilename + '.nc',
                                           ncVar)['calendar']
                    aux = read.lres_data(var0, 'pred', model=model, scene=sceneName, predName=predName)
                    scene_dates = aux['times']
                    if calendar == '360':
                        time_first, time_last = scene_dates.index(reference_first_date), reference_dates[-2]
                    else:
                        time_first, time_last = scene_dates.index(reference_first_date), scene_dates.index(
                            reference_last_date) + 1
                    sceneData = aux['data']
                    sceneData = sceneData[time_first:time_last]
                    scene_dates = scene_dates[time_first:time_last]

                    # Loading mean and std for standardizating predictors (models & reanalysis)
                    path_standard = '../aux/STANDARDIZATION/PRED/' + var0.upper() + '/' 
                    mean_refperiod = np.mean(np.load(path_standard + model + '_mean.npy')[ipred,:,:])
                    std_refperiod = np.mean(np.load(path_standard + model + '_std.npy')[ipred,:,:])
                     
                    rea_mean_refperiod = np.mean(np.load(path_standard + 'reanalysis' + '_mean.npy')[ipred,:,:])
                    rea_std_refperiod = np.mean(np.load(path_standard + 'reanalysis' + '_std.npy')[ipred,:,:])
                    
                    # Calculate annual cycle
                    datevec = scene_dates
                    if predName in ['tmax','tmin','pcp']:
                        varcy = sceneData
                    else:
                        varcy = (sceneData - mean_refperiod)/ std_refperiod
                    lst_months = []
                    for mon in range(1,13):
                        kkmon = [ii for ii,val in enumerate(datevec) if val.month == mon]
                        if predName != 'pcp':
                            monmean = varcy[kkmon,:,:].mean(axis=0)
                            lst_months.append(monmean.mean())
                        else:
                            nyears = datevec[len(datevec) - 1].year - datevec[0].year
                            nyears_copy = nyears
                            monsum = varcy[kkmon,:,:].sum(axis=0)/nyears
                            lst_months.append(monsum.mean())
                    cycle = lst_months
                    
                    # Calculate annual values (used in spaghetti plots) 
                    datevec = postpro_lib.get_season(sceneData, scene_dates, season)['times']
                    if predName in ['tmax','tmin','pcp']:
                        varcy = postpro_lib.get_season(sceneData, scene_dates, season)['data']
                    else:
                        varcy = (postpro_lib.get_season(sceneData, scene_dates, season)['data'] - mean_refperiod)/ std_refperiod
                    
                    lst_years = []
                    years = [i for i in range(datevec[0].year, datevec[len(datevec) - 1].year)]
                    for iyear in years:
                        kkyear = [ii for ii,val in enumerate(datevec) if val.year == iyear]
                        if predName != 'pcp':
                            yearmean = varcy[kkyear,:,:].mean(axis=0)
                            lst_years.append(yearmean.mean())
                        else:
                            yearsum = varcy[kkyear,:,:].sum(axis=0)
                            lst_years.append(yearsum.mean())
                    spaghetti = lst_years
                    
                    # Add rea data to annual cycle
                    datevec = rea_dates
                    if predName in ['tmax','tmin','pcp']:
                        varcy = rea
                    else:
                        varcy = (rea - rea_mean_refperiod)/ rea_std_refperiod
                    lst_months = []
                    for mon in range(1,13):
                        kkmon = [ii for ii,val in enumerate(datevec) if val.month == mon]
                        if predName != 'pcp':
                            monmean = varcy[kkmon,:,:].mean(axis=0)
                            lst_months.append(monmean.mean())
                        else:
                            nyears = datevec[len(datevec)- 1].year - datevec[0].year
                            monsum = varcy[kkmon,:,:].sum(axis=0)/nyears
                            lst_months.append(monsum.mean())
                    cycle_rea = lst_months
                    
                    # Calculate rea annual values (used in spaghetti plots)
                    datevec = postpro_lib.get_season(sceneData, scene_dates, season)['times']
                    if predName in ['tmax','tmin','pcp']:
                        varcy = postpro_lib.get_season(sceneData, scene_dates, season)['data']
                    else:
                        varcy = (postpro_lib.get_season(sceneData, scene_dates, season)['data'] - mean_refperiod)/ std_refperiod
                    
                    lst_years = []
                    for iyear in years:
                        kkyear = [ii for ii,val in enumerate(datevec) if val.year == iyear]
                        if predName != 'pcp':
                            yearmean = varcy[kkyear,:,:].mean(axis=0)
                            lst_years.append(yearmean.mean())
                        else:
                            yearsum = varcy[kkyear,:,:].sum(axis=0)
                            lst_years.append(yearsum.mean())
                    spaghetti_rea = lst_years
                    
                    
                    # Select season data
                    rea_season = postpro_lib.get_season(rea, reference_dates, season)['data']
                    sceneData_season = postpro_lib.get_season(sceneData, scene_dates, season)['data']
                    

                    # Calculate mean values and absolute bias
                    rea_mean_season = np.nanmean(rea_season[:, 0, :, :], axis=0)
                    sceneData_mean_season = np.nanmean(sceneData_season[:, 0, :, :], axis=0)
                    bias = sceneData_mean_season - rea_mean_season
                    
                    # Calculate relative bias
                    if predName == 'pcp':
                        bias = 100*bias/rea_mean_season
                    '''
                    # Calculate annual accumulated precipitation 
                    if predName == 'pcp':
                        years = [i for i in range(scene_dates[0].year, scene_dates[-1].year)]
                        for iyear in years:
                            kkyear = [ii for ii,val in enumerate(scene_dates) if val.year == iyear]
                            yearsum = sceneData[kkyear,:,:].nansum(axis=0)/nyears
                    '''        
                    # Appending sceneData_mean_season of selected model in a list
                    if predName in ['tmax','tmin','pcp']:
                        lst_sceneData_mean_season.append(sceneData_mean_season)
                    
                    # Save results
                    np.save(pathTmp + '_'.join((var0, predName, model, sceneName, season, 'bias')), bias)
                    np.save(pathTmp + '_'.join((var0, predName, model, sceneName, 'ANNUAL', 'cycle')), cycle)
                    np.save(pathTmp + '_'.join((var0, predName, 'Reanalysis', 'ANNUAL', 'cycle_rea')), cycle_rea)
                    np.save(pathTmp + '_'.join((var0, predName, model, sceneName, season, 'spaghetti')), spaghetti)
                    np.save(pathTmp + '_'.join((var0, predName, 'Reanalysis', sceneName, season, 'spaghetti_rea')), spaghetti_rea)
                    print(var0, predName, model, sceneName, season)
                    
                    
                    # Q-Q plot 
                    fig, ax = plt.subplots()
                    a = rea_mean_season
                    b = sceneData_mean_season
                    percs = np.linspace(0,100,21)
                    qn_a = np.percentile(a, percs)
                    qn_b = np.percentile(b, percs)
                    ax.plot(qn_a,qn_b, ls="", marker="o")#plt
                    x = np.linspace(np.min((qn_a.min(),qn_b.min())), np.max((qn_a.max(),qn_b.max())))
                    ax.plot(x,x, color="k", ls="--")#plt
                    ax.set_xlabel('Reanalysis' + ' ' + unitplot(predName))
                    ax.set_ylabel(model + ' ' + unitplot(predName))
                    plt.title(' '.join(('qqPlot', predName, model, sceneName, season)))
                    filename = '_'.join((experiment, 'qqPlot', var0, predName, model, season))
                    #plt.show()
                    # exit()
                    plt.savefig(pathOut + filename)
                    plt.close()
                    

                # Saving multi-model statistics
                if predName in ['tmax','tmin','pcp']:
                    historical_multimodel_mean = np.mean(lst_sceneData_mean_season, axis=0)
                    historical_multimodel_std = np.std(lst_sceneData_mean_season, axis=0)
                    historical_multimodel_per25 = np.percentile(lst_sceneData_mean_season, 25, axis=0)
                    historical_multimodel_per50 = np.percentile(lst_sceneData_mean_season, 50, axis=0)
                    historical_multimodel_per75 = np.percentile(lst_sceneData_mean_season, 75, axis=0)
                    
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, season, 'multimodel_mean')), historical_multimodel_mean)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, season, 'multimodel_std')), historical_multimodel_std)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, season, 'multimodel_per25')), historical_multimodel_per25)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, season, 'multimodel_per50')), historical_multimodel_per50)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, season, 'multimodel_per75')), historical_multimodel_per75)
                    
                # Go through all models
                matrix = np.zeros((nmodels, nlats, nlons))
                for imodel in range(nmodels):
                    model = model_list[imodel]
                    matrix[imodel] = np.load(pathTmp + '_'.join((var0, predName, model, sceneName, season, 'bias.npy')))     
                matrix = matrix.reshape(nmodels, -1)
                

                # Boxplot
                fig, ax = plt.subplots()
                ax.boxplot(matrix.T, showfliers=False)
                ax.set_xticklabels(model_list, rotation=45, fontsize=5)
                plt.axhline(y=0, ls='--', c='grey')
                plt.title(' '.join(('bias', predName, season)))
                if predName != 'pcp':
                    ax.set_ylabel(unitplot(predName))
                else:
                    ax.set_ylabel('pcp relative bias (%)')
                # plt.show()
                # exit()
                filename = '_'.join((experiment, 'biasBoxplot', var0, predName, 'None', season))
                plt.savefig(pathOut + filename)
                plt.close()
                
                    
                # Annual cycle plot
                fig, ax = plt.subplots()
                for model in model_list:
                    cycle_load = np.load(pathTmp + '_'.join((var0, predName, model, sceneName, 'ANNUAL', 'cycle.npy')))
                    x_months = [i for i in ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
                    y_cycle = [i for i in cycle_load]
                    ax.plot(x_months, y_cycle, label= model )
                cycle_rea_load = np.load(pathTmp + '_'.join((var0, predName, 'Reanalysis', 'ANNUAL', 'cycle_rea.npy')))
                x_months = [i for i in ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
                y_cycle_rea = [i for i in cycle_rea_load]
                ax.plot(x_months, cycle_rea, label= 'Reanalysis' )    
                ax.set_xlabel('month')
                if predName in ['tmax','tmin', 'pcp']:
                    ax.set_ylabel(unitplot(predName))
                else: 
                    ax.set_ylabel('standardized ' + predName)
                plt.title(' '.join(('annual cycle', predName, sceneName)))
                ax.legend()
                filename = '_'.join((experiment, 'annualCycle', var0, predName, sceneName, 'None'))
                # plt.show()
                # exit()
                plt.savefig(pathOut + filename)
                plt.close()
                
                # Evolution Spaghetti plot in reference period 
                fig, ax = plt.subplots() 
                for model in model_list:
                    spaghetti_load = np.load(pathTmp + '_'.join((var0, predName, model, sceneName, season, 'spaghetti.npy')))
                    y_spaghetti = [i for i in spaghetti_load]
                    y_spaghetti = gaussian_filter1d(y_spaghetti, 5)
                    ax.plot(years, y_spaghetti, label= model)
                spaghetti_rea_load = np.load(pathTmp + '_'.join((var0, predName, 'Reanalysis', sceneName, season, 'spaghetti_rea.npy')))
                spaghetti_rea_load = gaussian_filter1d(spaghetti_rea_load, 5)
                ax.plot(years, spaghetti_rea_load, label= 'Reanalysis' )
                plt.title(' '.join((predName, sceneName, season)))
                if predName in ['tmax','tmin', 'pcp']:
                    ax.set_ylabel(unitplot(predName))
                else: 
                    ax.set_ylabel('standardized ' + predName)
                plt.legend()
                # plt.show()
                # exit()
                filename = '_'.join((experiment, 'evolSpaghetti', var0, predName, sceneName, season))
                plt.savefig(pathOut + filename)
                plt.close()
                
########################################################################################################################
def GCMs_evaluation_future():
    """
    Test the uncertainty in GCMs in the future, analysing all predictors, models, synoptic analogy fields...
    It can be used to discard models/predictors and to detect outliers.
    """

    print('GCMs_evaluation_future...')

    # Define parameters
    nmodels = len(model_list)
    nscenes = len(scene_list) - 1
    nlats = pred_nlats
    nlons = pred_nlons
    years = [i for i in range(ssp_years[0], ssp_years[-1]+1)]
    nYears = len(years)
    nseasons = len(season_dict.keys())

    # Go through all target variables
    for var0 in ('t', 'p', ):

        # Define pathTmp
        pathTmp = '../results/' + experiment + '/GCMs_evaluation_future/' + var0.upper() + '/'
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        pathOut = pathFigures
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Define preds
        preds = preds_dict[var0]
        npreds = len(preds)

        # Go through all predictors
        for ipred in range(npreds):
            predName = list(preds.keys())[ipred]

            # Go through all scenes
            for iscene in range(1, nscenes + 1):
                sceneName = scene_list[iscene]

                if sceneName != 'historical':

                    matrix = np.zeros((nseasons, nscenes, nmodels, nYears, nlats, nlons))
                    standardized_matrix = np.zeros((nseasons, nscenes, nmodels, nYears, nlats, nlons))
                    
                    # Go through all models and storage mean values
                    for imodel in range(nmodels):
                        model = model_list[imodel]
                        print(var0, predName, sceneName, model, 'evaluation future')

                        # Read model scene   
                        
                        aux = read.lres_data(var0, 'pred', model=model, scene=sceneName, predName=predName)
                        times = aux['times']
                        sceneData = aux['data'][:, 0, :, :]
                        
                        # Loading mean and std for standardizating predictors
                        path_standard = '../aux/STANDARDIZATION/PRED/' + var0.upper() + '/' 
                        mean_refperiod = np.mean(np.load(path_standard + model + '_mean.npy')[ipred,:,:])
                        std_refperiod = np.mean(np.load(path_standard + model + '_std.npy')[ipred,:,:])
                        
                        # Calculate annual cycle
                        datevec = times
                        if predName in ['tmax','tmin','pcp']:
                            varcy = sceneData
                        else:
                            varcy = (sceneData - mean_refperiod)/ std_refperiod
                        lst_months = []
                        for mon in range(1,13):
                            kkmon = [ii for ii,val in enumerate(datevec) if val.month == mon]
                            if predName != 'pcp':
                                monmean = varcy[kkmon,:,:].mean(axis=0)
                                lst_months.append(monmean.mean())
                            else:
                                nyears = datevec[len(datevec)-1].year - datevec[0].year
                                nyears_copy = nyears
                                monsum = varcy[kkmon,:,:].sum(axis=0)/nyears
                                lst_months.append(monsum.mean())
                        cycle = lst_months
                        np.save(pathTmp + '_'.join((var0, predName, model, sceneName, 'ANNUAL', 'cycle')), cycle)

                        # Go through all seasons
                        iseason = 0
                        for season in season_dict.values():
                            # Select season data
                            aux = postpro_lib.get_season(sceneData, times, season)
                            data_season, times_season = aux['data'], aux['times']

                            for iyear in range(nYears):
                                year = years[iyear]
                                idates = [i for i in range(len(times_season)) if times_season[i].year == year]
                                matrix[iseason, iscene-1, imodel, iyear] = np.mean(data_season[idates], axis=0)
                                
                                standardized_matrix[iseason, iscene-1, imodel, iyear] = np.mean((data_season[idates]-mean_refperiod)/std_refperiod, axis=0)
                                # mean and std (all models with a given variable, season and scenenario)
                                meanmodels = np.mean(matrix, axis = 2)
                                stdmodels = np.std(matrix, axis = 2) 
                                per25models = np.percentile(matrix,25, axis = 2)
                                per50models = np.percentile(matrix,50, axis = 2)
                                per75models = np.percentile(matrix,75, axis = 2)
                                
                                standardized_meanmodels = np.mean(standardized_matrix, axis = 2)
                                standardized_stdmodels = np.std(standardized_matrix, axis = 2) 
                                standardized_per25models = np.percentile(standardized_matrix,25, axis = 2)
                                standardized_per50models = np.percentile(standardized_matrix,50, axis = 2)
                                standardized_per75models = np.percentile(standardized_matrix,75, axis = 2)
                                
                            iseason += 1
                    
                    # Save results: raw data and standadized data
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'matrix')), matrix)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'meanmodels')), meanmodels)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'stdmodels')), stdmodels)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'per25models')), per25models)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'per50models')), per50models)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'per75models')), per75models)
                    
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_matrix')), standardized_matrix)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_meanmodels')), standardized_meanmodels)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_stdmodels')), standardized_stdmodels)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_per25models')), standardized_per25models)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_per50models')), standardized_per50models)
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_per75models')), standardized_per75models)
                                       
                    # Evolution tube plot - predictands
                    if predName in ['tmax','tmin']: 
                        color_dict = {'ssp119': 'darkblue', 'ssp126': 'lightblue', 'ssp245': 'orange', 'ssp370': 'salmon', 'ssp585': 'darkred'}
                        matrix_meanmodels = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'meanmodels.npy')))
                        matrix_stdmodels = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'stdmodels.npy')))
                        matrix_per25models = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'per25models.npy')))
                        matrix_per50models = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'per50models.npy')))
                        matrix_per75models = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'per75models.npy')))
                        iseason = 0
                        for season in season_dict.values():
                            #datamean = matrix_meanmodels[iseason, iscene-1]
                            #datamean = np.mean(datamean.reshape(nYears, -1), axis=1)
                            #datastd = matrix_stdmodels[iseason, iscene-1]
                            #datastd = np.mean(datastd.reshape(nYears, -1), axis=1)
                            dataper25 = matrix_per25models[iseason, iscene-1]
                            dataper25 = np.mean(dataper25.reshape(nYears, -1), axis=1)
                            dataper50 = matrix_per50models[iseason, iscene-1]
                            dataper50 = np.mean(dataper50.reshape(nYears, -1), axis=1)
                            dataper75 = matrix_per75models[iseason, iscene-1]
                            dataper75 = np.mean(dataper75.reshape(nYears, -1), axis=1)
                            hmm = np.load('../results/' + experiment + '/GCMs_evaluation_historical/' + var0.upper() + '/' + '_'.join((var0, predName, 'historical', season, 'multimodel_mean.npy')))
                            hmmm = np.mean(hmm)
                            fig, ax = plt.subplots()
                            plt.fill_between(years, dataper25-hmmm,dataper75-hmmm, color=color_dict[sceneName], alpha = 0.3)  
                            plt.plot(years, dataper50-hmmm, color=color_dict[sceneName], label = 'multi-model mean')
                            plt.title(' '.join((predName, sceneName, season)))
                            ax.set_ylabel(predName + ' ' + 'anomaly (°C)')
                            plt.legend()
                            # plt.show()
                            # exit()
                            filename = '_'.join((experiment, 'evolTube', var0, predName, sceneName, season))
                            plt.savefig(pathOut + filename)
                            plt.close()

                            iseason += 1
                    
                    elif predName in ['pcp']:
                        color_dict = {'ssp119': 'darkblue', 'ssp126': 'lightblue', 'ssp245': 'orange', 'ssp370': 'salmon', 'ssp585': 'darkred'}
                        matrix_meanmodels = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'meanmodels.npy')))
                        matrix_stdmodels = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'stdmodels.npy')))
                        matrix_per25models = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'per25models.npy')))
                        matrix_per50models = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'per50models.npy')))
                        matrix_per75models = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'per75models.npy')))
                        iseason = 0
                        for season in season_dict.values():
                            #datamean = matrix_meanmodels[iseason, iscene-1]
                            #datamean = np.mean(datamean.reshape(nYears, -1), axis=1)
                            #datastd = matrix_stdmodels[iseason, iscene-1]
                            #datastd = np.mean(datastd.reshape(nYears, -1), axis=1)
                            dataper25 = matrix_per25models[iseason, iscene-1]
                            dataper25 = np.mean(dataper25.reshape(nYears, -1), axis=1)
                            dataper50 = matrix_per50models[iseason, iscene-1]
                            dataper50 = np.mean(dataper50.reshape(nYears, -1), axis=1)
                            dataper75 = matrix_per75models[iseason, iscene-1]
                            dataper75 = np.mean(dataper75.reshape(nYears, -1), axis=1)
                            hmm = np.load('../results/' + experiment + '/GCMs_evaluation_historical/' + var0.upper() + '/' + '_'.join((var0, predName, 'historical', season, 'multimodel_mean.npy')))
                            hmmm = np.mean(hmm)
                            fig, ax = plt.subplots()
                            plt.fill_between(years, 100*(dataper25-hmmm)/hmmm,100*(dataper75-hmmm)/hmmm, color=color_dict[sceneName], alpha = 0.3)#color="k"                         
                            plt.plot(years, 100*(dataper50-hmmm)/hmmm, color=color_dict[sceneName], label = 'multi-model mean')
                            plt.title(' '.join((predName, sceneName, season)))
                            ax.set_ylabel(predName + ' ' + 'anomaly (%)')
                            plt.legend()
                            # plt.show()
                            # exit()
                            filename = '_'.join((experiment, 'evolTube', 'all', var0, predName, sceneName, season))
                            plt.savefig(pathOut + filename)
                            plt.close()

                            iseason += 1
                    
                    # Evolution tube plot - predictors
                    else:
                        color_dict = {'ssp119': 'darkblue', 'ssp126': 'lightblue', 'ssp245': 'orange', 'ssp370': 'salmon', 'ssp585': 'darkred'}
                        matrix_meanmodels = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_meanmodels.npy')))
                        matrix_stdmodels = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_stdmodels.npy')))
                        matrix_per25models = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_per25models.npy')))
                        matrix_per50models = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_per50models.npy')))
                        matrix_per75models = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_per75models.npy')))
                        iseason = 0
                        for season in season_dict.values():
                            #datamean = matrix_meanmodels[iseason, iscene-1]
                            #datamean = np.mean(datamean.reshape(nYears, -1), axis=1)
                            #datastd = matrix_stdmodels[iseason, iscene-1]
                            #datastd = np.mean(datastd.reshape(nYears, -1), axis=1)
                            dataper25 = matrix_per25models[iseason, iscene-1]
                            dataper25 = np.mean(dataper25.reshape(nYears, -1), axis=1)
                            dataper50 = matrix_per50models[iseason, iscene-1]
                            dataper50 = np.mean(dataper50.reshape(nYears, -1), axis=1)
                            dataper75 = matrix_per75models[iseason, iscene-1]
                            dataper75 = np.mean(dataper75.reshape(nYears, -1), axis=1)
                            fig, ax = plt.subplots()
                            plt.fill_between(years, dataper25,dataper75, color=color_dict[sceneName], alpha = 0.3)#color="k"
                            plt.plot(years, dataper50, color=color_dict[sceneName], label = 'multi-model mean')
                            plt.title(' '.join((predName, sceneName, season)))
                            ax.set_ylabel('standardized ' + predName)
                            plt.legend()
                            # plt.show()
                            # exit()
                            filename = '_'.join((experiment, 'evolTube', 'all', var0, predName, sceneName, season))
                            plt.savefig(pathOut + filename)
                            plt.close()

                            iseason += 1

                    # Annual cycle plot
                    fig, ax = plt.subplots()
                    for model in model_list:
                        cycle_load = np.load(pathTmp + '_'.join((var0, predName, model, sceneName, 'ANNUAL', 'cycle.npy')))
                        x_months = [i for i in ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
                        y_cycle = [i for i in cycle_load]
                        ax.plot(x_months, y_cycle, label= model )
                    ax.set_xlabel('month')
                    if predName in ['tmax','tmin', 'pcp']:
                        ax.set_ylabel(unitplot(predName))
                    else: 
                        ax.set_ylabel('standardized ' + predName)
                    plt.title(' '.join(('annual cycle', predName)))
                    ax.legend()
                    filename = '_'.join((experiment, 'annualCycle', 'None', var0, predName, sceneName, 'None'))
                    # plt.show()
                    # exit()
                    plt.savefig(pathOut + filename)
                    plt.close()                     
                    
                    # Evolution Spaghetti plot -predictands
                    if predName in ['tmax','tmin']:
                        matrix = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'matrix.npy')))
                        iseason = 0
                        for season in season_dict.values():
                            fig, ax = plt.subplots()
                            for imodel in range(nmodels):
                                model = model = model_list[imodel]
                                data = matrix[iseason, iscene-1, imodel]
                                data = np.mean(data.reshape(nYears, -1), axis=1)
                                data = gaussian_filter1d(data, 5)
                                hmm = np.load('../results/' + experiment + '/GCMs_evaluation_historical/' + var0.upper() + '/' + '_'.join((var0, predName, 'historical', season, 'multimodel_mean.npy')))
                                hmmm = np.mean(hmm)
                                plt.plot(years, data-hmmm, label=model)
                            plt.title(' '.join((predName, sceneName, season)))
                            ax.set_ylabel(predName + ' ' + 'anomaly (°C)')
                            plt.legend()
                            # plt.show()
                            # exit()
                            filename = '_'.join((experiment, 'evolSpaghetti', 'all', var0, predName, sceneName, season))
                            plt.savefig(pathOut + filename)
                            plt.close()

                            iseason += 1
                    
                    elif predName in ['pcp']:
                        matrix = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'matrix.npy')))
                        iseason = 0
                        for season in season_dict.values():
                            fig, ax = plt.subplots()
                            for imodel in range(nmodels):
                                model = model_list[imodel]
                                data = matrix[iseason, iscene-1, imodel]
                                data = np.mean(data.reshape(nYears, -1), axis=1)
                                data = gaussian_filter1d(data, 5)
                                hmm = np.load('../results/' + experiment + '/GCMs_evaluation_historical/' + var0.upper() + '/' + '_'.join((var0, predName, 'historical', season, 'multimodel_mean.npy')))
                                hmmm = np.mean(hmm)
                                plt.plot(years, 100*(data-hmmm)/hmmm, label=model)                       
                            plt.title(' '.join((predName, sceneName, season)))
                            ax.set_ylabel(predName + ' ' + 'anomaly (%)')
                            plt.legend()
                            # plt.show()
                            # exit()
                            filename = '_'.join((experiment, 'evolSpaghetti', 'all', var0, predName, sceneName, season))
                            plt.savefig(pathOut + filename)
                            plt.close()
                            
                            iseason += 1
                    
                    # Evolution Spaghetti plot -predictors
                    else:
                        matrix = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'standardized_matrix.npy')))
                        iseason = 0
                        for season in season_dict.values():
                            fig, ax = plt.subplots()
                            for imodel in range(nmodels):
                                model = model_list[imodel]
                                data = matrix[iseason, iscene-1, imodel]
                                data = np.mean(data.reshape(nYears, -1), axis=1)
                                data = gaussian_filter1d(data, 5)
                                plt.plot(years, data, label=model)
                            plt.title(' '.join((predName, sceneName, season)))
                            ax.set_ylabel('standardized ' + predName)
                            plt.legend()
                            # plt.show()
                            # exit()
                            filename = '_'.join((experiment, 'evolSpaghetti', 'all', var0, predName, sceneName, season))
                            plt.savefig(pathOut + filename)
                            plt.close()

                            iseason += 1
########################################################################################################################
def GCMs_evaluation():
    """
    Evaluate GCMs in a historical period and in the future
    """

    print('GCMs_evaluation...')

    GCMs_evaluation_historical()
    #GCMs_evaluation_future()