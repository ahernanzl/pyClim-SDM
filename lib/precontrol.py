import sys
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import MOS_lib
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

########################################################################################################################
def missing_data_check():
    """
    Check for missing data in predictors by GCMs. It can be used to discard some predictors/levels.
    """

    print('missing_data_check...')

    # Go through all target variables
    for targetVar in targetVars:

        # Define pathTmp
        pathTmp = '../results/' + experiment + '/missing_data_check/' + targetVar.upper() + '/'
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        pathOut = pathFigures
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Define preds
        preds = preds_dict[targetVar]
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
                    data = read.lres_data(targetVar, 'pred', model=model, scene=sceneName, predName=predName)['data']

                    # Calculate percentaje of nans
                    perc_nan = 100 * np.count_nonzero(np.isnan(data), axis=0)[0] / data.shape[0]
                    PERC_NAN[ipred, iscene, imodel] = perc_nan
                    print(targetVar, predName, sceneName, model, 'perc_nan', np.max(perc_nan))

                    if np.max(perc_nan) != 0:
                        # Plot map
                        filename = '_'.join((experiment, 'nansMap', targetVar, predName, model+'-'+sceneName, 'None'))
                        title = ' '.join((predName, model, sceneName, 'pertentage of NANs'))
                        plot.map(targetVar, perc_nan, 'perc_nan', grid='pred', path=pathOut, filename=filename, title=title)

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
            filename = '_'.join((experiment, 'nansMatrix', targetVar, 'None', sceneName, 'None.png'))
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
    for targetVar in targetVars:

        # Define pathTmp
        pathTmp = '../results/' + experiment + '/predictors_correlation/' + targetVar.upper() + '/'
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        pathOut = pathFigures
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # For interpolation
        interp_mode = 'bilinear'
        i_4nn = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper() + '_' + interp_mode + '/i_4nn.npy')
        j_4nn = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper() + '_' + interp_mode + '/j_4nn.npy')
        w_4nn = np.load(pathAux + 'ASSOCIATION/' + targetVar.upper() + '_' + interp_mode + '/w_4nn.npy')

        # Read data predictand
        obs = read.hres_data(targetVar, period='calibration')['data']

        # Define preds
        preds = preds_dict[targetVar]
        npreds = len(preds)

        # Go through all seasons
        for season in season_dict:

            R = np.zeros((npreds, hres_npoints[targetVar]))

            # Calculate correlations for each predictor
            for ipred in range(npreds):
                predName = list(preds.keys())[ipred]

                # Read data predictor
                data = read.lres_data(targetVar, 'pred', predName=predName)['data']

                # Select season
                data_season = postpro_lib.get_season(data, calibration_dates, season)['data']
                obs_season = postpro_lib.get_season(obs, calibration_dates, season)['data']

                # Go through all points
                for ipoint in range(hres_npoints[targetVar]):

                    # Interpolate to one point
                    X = grids.interpolate_predictors(data_season, i_4nn[ipoint], j_4nn[ipoint], w_4nn[ipoint], interp_mode)[:, 0]
                    y = obs_season[:, ipoint]

                    # Remove missing data
                    ivalid = np.where(abs(y - predictands_codification[targetVar]['special_value']) > 0.01)
                    X = X[ivalid]
                    y = y[ivalid]

                    # Calculate correlation
                    if targetVar == 'pr' or (targetVar == myTargetVar and myTargetVarIsGaussian == False):
                        R[ipred, ipoint] = spearmanr(X, y)[0]
                    else:
                        R[ipred, ipoint] = pearsonr(X, y)[0]

                # Save results
                print(targetVar, predName, 'correlation', season, np.mean(abs(R[ipred])))
                np.save(pathTmp + '_'.join((predName, 'correlation', season)), R[ipred])

            # Plot correlation maps and boxplots
            for ipred in range(npreds):
                predName = list(preds.keys())[ipred]

                # Load correlation
                R[ipred] = np.load(pathTmp + '_'.join((predName, 'correlation', season+'.npy')))
                print(targetVar, predName, 'correlation', season, np.mean(abs(R[ipred])))

                # Plot map
                title = ' '.join((targetVar.upper(), predName, 'correlation', season))
                filename = '_'.join((experiment, 'correlationMap', targetVar, predName, 'None', season))
                plot.map(targetVar, abs(R[ipred]), 'correlation', path=pathOut, filename=filename, title=title)

            # Boxplot
            fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
            ax.boxplot(abs(R.T), showfliers=False)
            ax.set_xticklabels(list(preds.keys()), rotation=90)
            plt.ylim((0, 1))
            plt.title(' '.join((targetVar.upper(), 'correlation', season)))
            # plt.show()
            # exit()
            filename = '_'.join((experiment, 'correlationBoxplot', targetVar, 'None', 'None', season))
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
    for targetVar in targetVars:

        # Define pathTmp
        pathTmp = '../results/' + experiment + '/GCMs_evaluation_historical/' + targetVar.upper() + '/'
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        pathOut = pathFigures
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Define preds
        preds = preds_dict[targetVar]
        # Adding tmax, tmin and pcp to preds
        # if var0 == 't':
        #     preds['tmax'] = {reaNames['tmax'], modNames['tmax']}
        #     preds['tmin'] = {reaNames['tmin'], modNames['tmin']}
        # else:
        preds[targetVar] = {reaNames[targetVar], modNames[targetVar]}
        npreds = len(preds)
        nmodels = len(model_list)
        nlats = pred_nlats
        nlons = pred_nlons

        # Go through all seasons
        for season in season_dict:
            n = 0
            # Go through all predictors
            for ipred in range(npreds):
                predName = list(preds.keys())[ipred]
                if predName == targetVar:
                    # print(targetVar, season, predName)

                    # Read reanalysis
                    rea = read.lres_data(targetVar, 'pred', predName=predName)['data']
                    time_first, time_last = calibration_dates.index(reference_first_date), calibration_dates.index(
                        reference_last_date) + 1
                    rea = rea[time_first:time_last]
                    rea_dates = read.lres_data(targetVar, 'pred', predName=predName)['times']
                    rea_dates = rea_dates[time_first:time_last]

                    # Go through all models
                    lst_sceneData_mean_season = []

                    for imodel in range(nmodels):
                        model = model_list[imodel]

                        ncVar = modNames[targetVar]

                        modelName, modelRun = model.split('_')[0], model.split('_')[1]
                        calendar_and_units = read.netCDF('../input_data/models/',
                                               ncVar + '_' + modelName + '_' + sceneName + '_' + modelRun + '_' + historicalPeriodFilename + '.nc',
                                               ncVar)
                        calendar = calendar_and_units['calendar']
                        unitspred = calendar_and_units['units']

                        if predName in ('tasmax', 'tasmin', 'tas'):
                            if unitspred == 'K':
                                unitspred = degree_sign
                            elif (unitspred == 'Unknown') and (np.nanmean(calendar_and_units['data']) > 100):
                                unitspred = degree_sign
                        elif predName == 'pr':
                            if unitspred == 'kg m-2 s-1':
                                unitspred = 'mm'
                            elif (unitspred == 'Unknown') and (model == 'reanalysis'):
                                unitspred = 'mm'
                        elif predName == 'clt':
                            if unitspred != '%' and (int(np.nanmax(calendar_and_units['data'])) <= 1):
                                unitspred = '%'
                        elif predName == 'zg':
                            if unitspred != 'm':
                                unitspred = 'm'

                        aux = read.lres_data(targetVar, 'pred', model=model, scene=sceneName, predName=predName)
                        scene_dates = aux['times']
                        if calendar in ['360_day', '360']:
                            time_first, time_last = scene_dates.index(reference_first_date), reference_dates[-2]
                        else:
                            time_first, time_last = scene_dates.index(reference_first_date), scene_dates.index(
                                reference_last_date) + 1
                        sceneData = aux['data']
                        sceneData = sceneData[time_first:time_last]
                        scene_dates = scene_dates[time_first:time_last]

                        # if predName == targetVar:
                        #     n = n + 1
                        # else:
                        #     # Calculate mean and std for standardizating predictors (models & reanalysis)
                        #     mean_refperiod = np.nanmean(sceneData, axis=0)
                        #     std_refperiod = np.nanstd(sceneData, axis=0)
                        #     rea_mean_refperiod = np.nanmean(rea, axis=0)
                        #     rea_std_refperiod = np.nanstd(rea, axis=0)

                        # Calculate annual cycle
                        datevec = scene_dates
                        if predName == targetVar:
                            varcy = sceneData
                        # else:
                        #     varcy = (sceneData - mean_refperiod)/ std_refperiod
                            lst_months = []
                            for mon in range(1,13):
                                kkmon = [ii for ii,val in enumerate(datevec) if val.month == mon]
                                if predName != 'pr':
                                    monmean = np.nanmean(varcy[kkmon,:,:], axis=0)
                                    lst_months.append(np.nanmean(monmean))
                                else:
                                    nyears = datevec[len(datevec) - 1].year - datevec[0].year
                                    nyears_copy = nyears
                                    monsum = np.nansum(varcy[kkmon,:,:], axis=0)/nyears
                                    lst_months.append(np.nanmean(monsum))
                            cycle = lst_months

                        # # Calculate annual values (used in spaghetti plots)
                        # datevec = postpro_lib.get_season(sceneData, scene_dates, season)['times']
                        # if predName == targetVar:
                        #     varcy = postpro_lib.get_season(sceneData, scene_dates, season)['data']
                        # else:
                        #     varcy = (postpro_lib.get_season(sceneData, scene_dates, season)['data'] - mean_refperiod)/ std_refperiod
                        # lst_years = []
                        # years = [i for i in range(datevec[0].year, datevec[len(datevec) - 1].year)]
                        # for iyear in years:
                        #     kkyear = [ii for ii,val in enumerate(datevec) if val.year == iyear]
                        #     if predName != 'pr':
                        #         yearmean = np.nanmean(varcy[kkyear,:,:], axis=0)
                        #         lst_years.append(np.nanmean(yearmean))
                        #     else:
                        #         yearsum = np.nansum(varcy[kkyear,:,:], axis=0)
                        #         lst_years.append(np.nanmean(yearsum))
                        # spaghetti = lst_years

                        # Add rea data to annual cycle
                        datevec = rea_dates
                        if predName == targetVar:
                            varcy = rea
                        # else:
                        #     varcy = (rea - rea_mean_refperiod)/ rea_std_refperiod
                            lst_months = []
                            for mon in range(1,13):
                                kkmon = [ii for ii,val in enumerate(datevec) if val.month == mon]
                                if predName != 'pr':
                                    monmean = np.nanmean(varcy[kkmon,:,:], axis=0)
                                    lst_months.append(np.nanmean(monmean))
                                else:
                                    nyears = datevec[len(datevec)- 1].year - datevec[0].year
                                    monsum = np.nansum(varcy[kkmon,:,:], axis=0)/nyears
                                    lst_months.append(np.nanmean(monsum))
                            cycle_rea = lst_months

                        # # Calculate rea annual values (used in spaghetti plots)
                        # datevec = postpro_lib.get_season(rea, reference_dates, season)['times']
                        # if predName == targetVar:
                        #     varcy = postpro_lib.get_season(rea, reference_dates, season)['data']
                        # else:
                        #     varcy = (postpro_lib.get_season(rea, reference_dates, season)['data'] - rea_mean_refperiod)/ rea_std_refperiod
                        # lst_years = []
                        # for iyear in years:
                        #     kkyear = [ii for ii,val in enumerate(datevec) if val.year == iyear]
                        #     if predName != 'pr':
                        #         yearmean = np.nanmean(varcy[kkyear,:,:], axis=0)
                        #         lst_years.append(np.nanmean(yearmean))
                        #     else:
                        #         yearsum = np.nansum(varcy[kkyear,:,:], axis=0)
                        #         lst_years.append(np.nanmean(yearsum))
                        # spaghetti_rea = lst_years


                        # Select season data
                        rea_season = postpro_lib.get_season(rea, reference_dates, season)['data']
                        sceneData_season = postpro_lib.get_season(sceneData, scene_dates, season)['data']

                        # Calculate mean values and absolute bias
                        rea_mean_season = np.nanmean(rea_season[:, 0, :, :], axis=0)
                        sceneData_mean_season = np.nanmean(sceneData_season[:, 0, :, :], axis=0)
                        bias = sceneData_mean_season - rea_mean_season

                        # Calculate relative bias
                        if predName == 'pr':
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
                        if predName == targetVar:
                            lst_sceneData_mean_season.append(sceneData_mean_season)

                            # Save results
                            np.save(pathTmp + '_'.join((targetVar, predName, model, sceneName, season, 'bias')), bias)
                            np.save(pathTmp + '_'.join((targetVar, predName, model, sceneName, annualName, 'cycle')), cycle)
                            np.save(pathTmp + '_'.join((targetVar, predName, 'Reanalysis', annualName, 'cycle_rea')), cycle_rea)
                            # np.save(pathTmp + '_'.join((targetVar, predName, model, sceneName, season, 'spaghetti')), spaghetti)
                            # np.save(pathTmp + '_'.join((targetVar, predName, 'Reanalysis', sceneName, season, 'spaghetti_rea')), spaghetti_rea)
                            print(targetVar, predName, model, sceneName, season)

                            # Q-Q plot
                            ''' 
                            fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                            a = rea_season #rea_mean_season
                            b = sceneData_season #sceneData_mean_season
                            percs = np.linspace(0,100, 100)
                            qn_a = np.nanpercentile(a, percs)
                            qn_b = np.nanpercentile(b, percs)
                            ax.plot(qn_a,qn_b, ls="", marker="o")#plt
                            x = np.linspace(np.nanmin((np.nanmin(qn_a),np.nanmin(qn_b))), np.nanmax((np.nanmax(qn_a),np.nanmax(qn_b))))
                            ax.plot(x,x, color="k", ls="--")#plt
                            ax.set_xlabel('Reanalysis' + ' ' + unitspred)
                            ax.set_ylabel(model + ' ' + unitspred)
                            plt.title(' '.join(('qqPlot', predName, model, sceneName, season)))
                            filename = '_'.join((experiment, 'qqPlot', targetVar, predName, model.replace('_', '-'), season))
                            #plt.show()
                            # exit()
                            plt.savefig(pathOut + filename)
                            plt.close()
                            '''
                            # Set ylabel, perc_list and c_list
                            if targetVar == 'pr':
                                perc_list = (99, 90, 75, 50)
                                c_list = ('g', 'm', 'b', 'k')
                            else:
                                perc_list = (5, 25, 50, 75, 95)
                                c_list = ('k', 'c', 'r', 'm', 'g')

                            fig = plt.figure()
                            ax = fig.add_subplot(111)
                            ax.set_aspect('equal', adjustable='box')

                            m, M = 100, -100
                            for i in reversed(range(len(c_list))):
                                p, c = perc_list[i], c_list[i]
                                # print('perc', p)
                                px, py = np.nanpercentile(rea_season, p, axis=0)[0], np.nanpercentile(sceneData_season, p, axis=0)[0]
                                plt.plot(px, py, '+', c=c, label='p' + str(p))
                                # plt.plot(px, py, '+', c=c, markersize=2, label='p'+str(p))
                                M = int(max(np.max(px), np.max(py), M))
                                m = int(min(np.min(px), np.min(py), m))
                            plt.xlim(m, M)
                            plt.ylim(m, M)
                            plt.xlabel('Reanalysis' + ' ' + unitspred)
                            plt.ylabel(model + ' ' + unitspred)
                            h = []
                            for i in range(len(c_list)):
                                h.append(Line2D([0], [0], marker='o', markersize=np.sqrt(20), color=c_list[i],
                                                linestyle='None'))
                            plt.legend(h, ['p' + str(x) for x in perc_list], markerscale=2, scatterpoints=1,
                                       fontsize=10)
                            m -= 5
                            M += 5
                            plt.plot(range(m, M), range(m, M))

                            plt.title(' '.join(('qqPlot', predName, model, sceneName, season)))
                            filename = '_'.join(
                                (experiment, 'qqPlot', targetVar, predName, model.replace('_', '-'), season))
                            # plt.show()
                            # exit()
                            plt.savefig(pathOut + filename)
                            plt.close()
                    # Saving multi-model statistics
                    if predName == targetVar:
                        historical_multimodel_mean = np.nanmean(lst_sceneData_mean_season, axis=0)
                        # historical_multimodel_std = np.nanstd(lst_sceneData_mean_season, axis=0)
                        # historical_multimodel_per25 = np.nanpercentile(lst_sceneData_mean_season, 25, axis=0)
                        # historical_multimodel_per50 = np.nanpercentile(lst_sceneData_mean_season, 50, axis=0)
                        # historical_multimodel_per75 = np.nanpercentile(lst_sceneData_mean_season, 75, axis=0)

                        np.save(pathTmp + '_'.join((targetVar, predName, sceneName, season, 'multimodel_mean')), historical_multimodel_mean)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, season, 'multimodel_std')), historical_multimodel_std)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, season, 'multimodel_per25')), historical_multimodel_per25)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, season, 'multimodel_per50')), historical_multimodel_per50)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, season, 'multimodel_per75')), historical_multimodel_per75)

                        # Go through all models
                        matrix = np.zeros((nmodels, nlats, nlons))
                        for imodel in range(nmodels):
                            model = model_list[imodel]
                            matrix[imodel] = np.load(pathTmp + '_'.join((targetVar, predName, model, sceneName, season, 'bias.npy')))
                        matrix = matrix.reshape(nmodels, -1)

                        matrixT = matrix.T
                        mask = ~np.isnan(matrixT)
                        matrixT = [d[m] for d, m in zip(matrixT.T, mask.T)]

                        # Boxplot
                        fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                        ax.boxplot(matrixT, showfliers=False)
                        ax.set_xticklabels(model_list, rotation=45, fontsize=5)
                        plt.axhline(y=0, ls='--', c='grey')
                        plt.title(' '.join(('bias', predName, season)))
                        if predName != 'pr':
                            ax.set_ylabel(unitspred)
                        else:
                            ax.set_ylabel('pr relative bias (%)')
                        # plt.show()
                        # exit()
                        filename = '_'.join((experiment, 'biasBoxplot', targetVar, predName, sceneName, season))
                        plt.savefig(pathOut + filename)
                        plt.close()


                        # Annual cycle plot
                        fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                        for model in model_list:
                            cycle_load = np.load(pathTmp + '_'.join((targetVar, predName, model, sceneName, annualName, 'cycle.npy')))
                            x_months = [i for i in ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
                            y_cycle = [i for i in cycle_load]
                            ax.plot(x_months, y_cycle, label= model )
                        cycle_rea_load = np.load(pathTmp + '_'.join((targetVar, predName, 'Reanalysis', annualName, 'cycle_rea.npy')))
                        x_months = [i for i in ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
                        y_cycle_rea = [i for i in cycle_rea_load]
                        ax.plot(x_months, cycle_rea, label= 'Reanalysis', linestyle = '--', linewidth = 4, c = 'k' )
                        ax.set_xlabel('month')
                        if predName == targetVar:
                            ax.set_ylabel(unitspred)
                        else:
                            ax.set_ylabel('standardized ' + predName)
                        plt.title(' '.join(('annual cycle', predName, sceneName)))
                        ax.legend()
                        filename = '_'.join((experiment, 'annualCycle', targetVar, predName, sceneName, 'None'))
                        # plt.show()
                        # exit()
                        plt.savefig(pathOut + filename)
                        plt.close()

                        # # Evolution Spaghetti plot in reference period
                        # fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                        # for model in model_list:
                        #     spaghetti_load = np.load(pathTmp + '_'.join((targetVar, predName, model, sceneName, season, 'spaghetti.npy')))
                        #     y_spaghetti = [i for i in spaghetti_load]
                        #     y_spaghetti = gaussian_filter1d(y_spaghetti, 5)
                        #     ax.plot(years, y_spaghetti, label= model)
                        # spaghetti_rea_load = np.load(pathTmp + '_'.join((targetVar, predName, 'Reanalysis', sceneName, season, 'spaghetti_rea.npy')))
                        # spaghetti_rea_load = gaussian_filter1d(spaghetti_rea_load, 5)
                        # ax.plot(years, spaghetti_rea_load, label= 'Reanalysis', linestyle = '--', linewidth = 4, c = 'k'  )
                        # plt.title(' '.join((predName, sceneName, season)))
                        # if predName == targetVar:
                        #     ax.set_ylabel(unitspred)
                        # else:
                        #     ax.set_ylabel('standardized ' + predName)
                        # plt.legend()
                        # # plt.show()
                        # # exit()
                        # filename = '_'.join((experiment, 'evolSpaghetti', targetVar, predName, sceneName, season))
                        # plt.savefig(pathOut + filename)
                        # plt.close()


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
    for targetVar in targetVars:

        # Define pathTmp
        pathTmp = '../results/' + experiment + '/GCMs_evaluation_future/' + targetVar.upper() + '/'
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        pathOut = pathFigures
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Define preds
        preds = preds_dict[targetVar]
        preds[targetVar] = {reaNames[targetVar], modNames[targetVar]}
        npreds = len(preds)
        n = 0

        # Go through all predictors
        for ipred in range(npreds):
            predName = list(preds.keys())[ipred]
            if predName == targetVar:
                # Go through all scenes
                for iscene in range(1, nscenes + 1):
                    sceneName = scene_list[iscene]

                    if sceneName != 'historical':

                        matrix = np.zeros((nseasons, nscenes, nmodels, nYears, nlats, nlons))
                        # standardized_matrix = np.zeros((nseasons, nscenes, nmodels, nYears, nlats, nlons))

                        # Go through all models and storage mean values
                        for imodel in range(nmodels):
                            model = model_list[imodel]
                            print(targetVar, predName, sceneName, model, 'evaluation future')
                            ncVar = modNames[targetVar]
                            modelName, modelRun = model.split('_')[0], model.split('_')[1]
                            readunits = read.netCDF('../input_data/models/',
                                               ncVar + '_' + modelName + '_' + sceneName + '_' + modelRun + '_' + sspPeriodFilename + '.nc',
                                               ncVar)
                            calendar = readunits['calendar']
                            unitspred = readunits['units']
                            if predName in ('tasmax', 'tasmin', 'tas'):
                                if unitspred == 'K':
                                    unitspred = degree_sign
                                elif (unitspred == 'Unknown') and (np.nanmean(readunits['data']) > 100):
                                    unitspred = degree_sign
                            elif predName == 'pr':
                                if unitspred == 'kg m-2 s-1':
                                    unitspred = 'mm'
                                elif (unitspred == 'Unknown') and (model == 'reanalysis'):
                                    unitspred = 'mm'
                            elif predName == 'clt':
                                if unitspred != '%' and (int(np.nanmax(readunits['data'])) <= 1):
                                    unitspred = '%'
                            elif predName == 'zg':
                                if unitspred != 'm':
                                    unitspred = 'm'
                            aux2 = read.lres_data(targetVar, 'pred', model=model, scene='historical', predName=predName)
                            scene_dates = aux2['times']
                            if calendar in ['360_day', '360']:
                                time_first, time_last = scene_dates.index(reference_first_date), reference_dates[-2]
                            else:
                                time_first, time_last = scene_dates.index(reference_first_date), scene_dates.index(
                                    reference_last_date) + 1

                            # Read model scene8
                            aux = read.lres_data(targetVar, 'pred', model=model, scene=sceneName, predName=predName)
                            times = aux['times']
                            sceneData = aux['data'][:, 0, :, :]

                            # if predName == targetVar:
                            #     n = n + 1
                            # else:
                            #     # Loading mean and std for standardizating predictors
                            #     mean_refperiod = np.nanmean(aux2['data'][time_first:time_last], axis=0)
                            #     std_refperiod = np.nanstd(aux2['data'][time_first:time_last], axis=0)
                            # Calculate annual cycle
                            datevec = times
                            # datevec = scene_dates
                            if predName == targetVar:
                                varcy = sceneData
                            # else:
                            #     varcy = (sceneData - mean_refperiod)/ std_refperiod
                                lst_months = []
                                for mon in range(1,13):
                                    kkmon = [ii for ii,val in enumerate(datevec) if val.month == mon]
                                    if predName != 'pr':
                                        monmean = np.nanmean(varcy[kkmon,:,:], axis=0)
                                        lst_months.append(np.nanmean(monmean))
                                    else:
                                        nyears = datevec[len(datevec)-1].year - datevec[0].year
                                        nyears_copy = nyears
                                        monsum = np.nansum(varcy[kkmon,:,:], axis=0)/nyears
                                        lst_months.append(np.nanmean(monsum))
                                cycle = lst_months
                                np.save(pathTmp + '_'.join((targetVar, predName, model, sceneName, annualName, 'cycle')), cycle)
                                print('cycle')

                            # Go through all seasons
                            iseason = 0
                            for season in season_dict:

                                # Select season data
                                aux = postpro_lib.get_season(sceneData, times, season)
                                data_season, times_season = aux['data'], aux['times']

                                for iyear in range(nYears):
                                    year = years[iyear]
                                    # print(sceneName, model, season, year)
                                    idates = [i for i in range(len(times_season)) if times_season[i].year == year]
                                    matrix[iseason, iscene-1, imodel, iyear] = np.nanmean(data_season[idates], axis=0)
                                    # standardized_matrix[iseason, iscene-1, imodel, iyear] = np.nanmean((data_season[idates]-mean_refperiod)/std_refperiod, axis=0)

                                    # mean and std (all models with a given variable, season and scenenario)
                                    # meanmodels = np.nanmean(matrix, axis = 2)
                                    # stdmodels = np.nanstd(matrix, axis = 2)
                                    # per25models = np.nanpercentile(matrix,25, axis = 2)
                                    # per50models = np.nanpercentile(matrix,50, axis = 2)
                                    # per75models = np.nanpercentile(matrix,75, axis = 2)

                                    # standardized_meanmodels = np.nanmean(standardized_matrix, axis = 2)
                                    # standardized_stdmodels = np.nanstd(standardized_matrix, axis = 2)
                                    # standardized_per25models = np.nanpercentile(standardized_matrix,25, axis = 2)
                                    # standardized_per50models = np.nanpercentile(standardized_matrix,50, axis = 2)
                                    # standardized_per75models = np.nanpercentile(standardized_matrix,75, axis = 2)

                                iseason += 1


                        # Save results: raw data and standadized data
                        np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'matrix')), matrix)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'meanmodels')), meanmodels)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'stdmodels')), stdmodels)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'per25models')), per25models)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'per50models')), per50models)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'per75models')), per75models)
                        #
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_matrix')), standardized_matrix)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_meanmodels')), standardized_meanmodels)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_stdmodels')), standardized_stdmodels)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_per25models')), standardized_per25models)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_per50models')), standardized_per50models)
                        # np.save(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_per75models')), standardized_per75models)
                        # print('matrix')

                        # # Evolution tube plot - predictands
                        # if (predName == targetVar) and (predName != 'pr'):
                        #     color_dict = {'ssp119': 'darkblue', 'ssp126': 'lightblue', 'ssp245': 'orange', 'ssp370': 'salmon', 'ssp585': 'darkred'}
                        #     # matrix_meanmodels = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'meanmodels.npy')))
                        #     # matrix_stdmodels = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'stdmodels.npy')))
                        #     matrix_per25models = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'per25models.npy')))
                        #     matrix_per50models = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'per50models.npy')))
                        #     matrix_per75models = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'per75models.npy')))
                        #     iseason = 0
                        #     for season in season_dict:
                        #         #datamean = matrix_meanmodels[iseason, iscene-1]
                        #         #datamean = np.nanmean(datamean.reshape(nYears, -1), axis=1)
                        #         #datastd = matrix_stdmodels[iseason, iscene-1]
                        #         #datastd = np.nanmean(datastd.reshape(nYears, -1), axis=1)
                        #         dataper25 = matrix_per25models[iseason, iscene-1]
                        #         dataper25 = np.nanmean(dataper25.reshape(nYears, -1), axis=1)
                        #         dataper50 = matrix_per50models[iseason, iscene-1]
                        #         dataper50 = np.nanmean(dataper50.reshape(nYears, -1), axis=1)
                        #         dataper75 = matrix_per75models[iseason, iscene-1]
                        #         dataper75 = np.nanmean(dataper75.reshape(nYears, -1), axis=1)
                        #         hmm = np.load('../results/' + experiment + '/GCMs_evaluation_historical/' + targetVar.upper() + '/' + '_'.join((targetVar, predName, 'historical', season, 'multimodel_mean.npy')))
                        #         hmmm = np.nanmean(hmm)
                        #         fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                        #         plt.fill_between(years, dataper25-hmmm,dataper75-hmmm, color=color_dict[sceneName], alpha = 0.3)
                        #         plt.plot(years, dataper50-hmmm, color=color_dict[sceneName], label = sceneName + ' ' + '(' + str(nmodels) + ')')
                        #         plt.title(' '.join((predName, sceneName, season)))
                        #         if targetVar in ('tas', 'tasmax', 'tasmin'):
                        #             ax.set_ylabel(predName + ' ' + 'anomaly (°C)')
                        #         elif targetVar in ('clt', 'hurs'):
                        #             ax.set_ylabel(predName + ' ' + 'anomaly (%)')
                        #         elif targetVar in ('uas', 'vas', 'sfcWind'):
                        #             ax.set_ylabel(predName + ' ' + 'anomaly (m/s)')
                        #         plt.legend()
                        #         # plt.show()
                        #         # exit()
                        #         filename = '_'.join((experiment, 'evolTube', targetVar, predName, sceneName, season))
                        #         plt.savefig(pathOut + filename)
                        #         plt.close()
                        #
                        #         iseason += 1
                        #
                        # elif predName == 'pr':
                        #     color_dict = {'ssp119': 'darkblue', 'ssp126': 'lightblue', 'ssp245': 'orange', 'ssp370': 'salmon', 'ssp585': 'darkred'}
                        #     # matrix_meanmodels = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'meanmodels.npy')))
                        #     # matrix_stdmodels = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'stdmodels.npy')))
                        #     matrix_per25models = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'per25models.npy')))
                        #     matrix_per50models = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'per50models.npy')))
                        #     matrix_per75models = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'per75models.npy')))
                        #     iseason = 0
                        #     for season in season_dict:
                        #         #datamean = matrix_meanmodels[iseason, iscene-1]
                        #         #datamean = np.nanmean(datamean.reshape(nYears, -1), axis=1)
                        #         #datastd = matrix_stdmodels[iseason, iscene-1]
                        #         #datastd = np.nanmean(datastd.reshape(nYears, -1), axis=1)
                        #         dataper25 = matrix_per25models[iseason, iscene-1]
                        #         dataper25 = np.nanmean(dataper25.reshape(nYears, -1), axis=1)
                        #         dataper50 = matrix_per50models[iseason, iscene-1]
                        #         dataper50 = np.nanmean(dataper50.reshape(nYears, -1), axis=1)
                        #         dataper75 = matrix_per75models[iseason, iscene-1]
                        #         dataper75 = np.nanmean(dataper75.reshape(nYears, -1), axis=1)
                        #         hmm = np.load('../results/' + experiment + '/GCMs_evaluation_historical/' + targetVar.upper() + '/' + '_'.join((targetVar, predName, 'historical', season, 'multimodel_mean.npy')))
                        #         hmmm = np.nanmean(hmm)
                        #         fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                        #         plt.fill_between(years, 100*(dataper25-hmmm)/hmmm,100*(dataper75-hmmm)/hmmm, color=color_dict[sceneName], alpha = 0.3)#color="k"
                        #         plt.plot(years, 100*(dataper50-hmmm)/hmmm, color=color_dict[sceneName], label = sceneName + ' ' + '(' + str(nmodels) + ')')
                        #         plt.title(' '.join((predName, sceneName, season)))
                        #         ax.set_ylabel(predName + ' ' + 'anomaly (%)')
                        #         plt.legend()
                        #         # plt.show()
                        #         # exit()
                        #         filename = '_'.join((experiment, 'evolTube', targetVar, predName, sceneName, season))
                        #         plt.savefig(pathOut + filename)
                        #         plt.close()
                        #
                        #         iseason += 1
                        #
                        # # Evolution tube plot - predictors
                        # else:
                        #     color_dict = {'ssp119': 'darkblue', 'ssp126': 'lightblue', 'ssp245': 'orange', 'ssp370': 'salmon', 'ssp585': 'darkred'}
                        #     matrix_meanmodels = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_meanmodels.npy')))
                        #     matrix_stdmodels = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_stdmodels.npy')))
                        #     matrix_per25models = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_per25models.npy')))
                        #     matrix_per50models = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_per50models.npy')))
                        #     matrix_per75models = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_per75models.npy')))
                        #     iseason = 0
                        #     for season in season_dict:
                        #         #datamean = matrix_meanmodels[iseason, iscene-1]
                        #         #datamean = np.nanmean(datamean.reshape(nYears, -1), axis=1)
                        #         #datastd = matrix_stdmodels[iseason, iscene-1]
                        #         #datastd = np.nanmean(datastd.reshape(nYears, -1), axis=1)
                        #         dataper25 = matrix_per25models[iseason, iscene-1]
                        #         dataper25 = np.nanmean(dataper25.reshape(nYears, -1), axis=1)
                        #         dataper50 = matrix_per50models[iseason, iscene-1]
                        #         dataper50 = np.nanmean(dataper50.reshape(nYears, -1), axis=1)
                        #         dataper75 = matrix_per75models[iseason, iscene-1]
                        #         dataper75 = np.nanmean(dataper75.reshape(nYears, -1), axis=1)
                        #         fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                        #         plt.fill_between(years, dataper25,dataper75, color=color_dict[sceneName], alpha = 0.3)#color="k"
                        #         plt.plot(years, dataper50, color=color_dict[sceneName], label = sceneName + ' ' + '(' + str(nmodels) + ')')
                        #         plt.title(' '.join((predName, sceneName, season)))
                        #         ax.set_ylabel('standardized ' + predName)
                        #         plt.legend()
                        #         # plt.show()
                        #         # exit()
                        #         filename = '_'.join((experiment, 'evolTube', targetVar, predName, sceneName, season))
                        #         plt.savefig(pathOut + filename)
                        #         plt.close()
                        #
                        #         iseason += 1

                        # Annual cycle plot
                        if predName == targetVar:
                            fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                            for model in model_list:
                                cycle_load = np.load(pathTmp + '_'.join((targetVar, predName, model, sceneName, annualName, 'cycle.npy')))
                                x_months = [i for i in ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
                                y_cycle = [i for i in cycle_load]
                                ax.plot(x_months, y_cycle, label= model )
                            ax.set_xlabel('month')
                            # if predName == targetVar:
                            ax.set_ylabel(unitspred)
                            # else:
                            #     ax.set_ylabel('standardized ' + predName)
                            plt.title(' '.join(('annual cycle', predName)))
                            ax.legend()
                            filename = '_'.join((experiment, 'annualCycle', targetVar, predName, sceneName, 'None'))
                            # plt.show()
                            # exit()
                            plt.savefig(pathOut + filename)
                            plt.close()

                        # Evolution Spaghetti plot -predictands
                        if (predName == targetVar) and (predName != 'pr'):
                            matrix = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'matrix.npy')))
                            iseason = 0
                            for season in season_dict:
                                fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                                for imodel in range(nmodels):
                                    model = model_list[imodel]
                                    data = matrix[iseason, iscene-1, imodel]
                                    data = np.nanmean(data.reshape(nYears, -1), axis=1)
                                    data = gaussian_filter1d(data, 5)
                                    hmm = np.load('../results/' + experiment + '/GCMs_evaluation_historical/' + targetVar.upper() + '/' + '_'.join((targetVar, predName, 'historical', season, 'multimodel_mean.npy')))
                                    hmmm = np.nanmean(hmm)
                                    plt.plot(years, data-hmmm, label=model)
                                plt.title(' '.join((predName, sceneName, season)))
                                if targetVar in ('tas','tasmax','tasmin'):
                                    ax.set_ylabel(predName + ' ' + 'anomaly (°C)')
                                elif targetVar in ('clt', 'hurs'):
                                    ax.set_ylabel(predName + ' ' + 'anomaly (%)')
                                elif targetVar in ('uas', 'vas', 'sfcWind'):
                                    ax.set_ylabel(predName + ' ' + 'anomaly (m/s)')
                                plt.legend()
                                # plt.show()
                                # exit()
                                filename = '_'.join((experiment, 'evolSpaghetti', targetVar, predName, sceneName, season))
                                plt.savefig(pathOut + filename)
                                plt.close()

                                iseason += 1

                        elif predName == 'pr':
                            matrix = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'matrix.npy')))
                            iseason = 0
                            for season in season_dict:
                                fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                                for imodel in range(nmodels):
                                    model = model_list[imodel]
                                    data = matrix[iseason, iscene-1, imodel]
                                    data = np.nanmean(data.reshape(nYears, -1), axis=1)
                                    data = gaussian_filter1d(data, 5)
                                    hmm = np.load('../results/' + experiment + '/GCMs_evaluation_historical/' + targetVar.upper() + '/' + '_'.join((targetVar, predName, 'historical', season, 'multimodel_mean.npy')))
                                    hmmm = np.nanmean(hmm)
                                    plt.plot(years, 100*(data-hmmm)/hmmm, label=model)
                                plt.title(' '.join((predName, sceneName, season)))
                                ax.set_ylabel(predName + ' ' + 'anomaly (%)')
                                plt.legend()
                                # plt.show()
                                # exit()
                                filename = '_'.join((experiment, 'evolSpaghetti', targetVar, predName, sceneName, season))
                                plt.savefig(pathOut + filename)
                                plt.close()

                                iseason += 1




                        # # Evolution Spaghetti plot -predictors
                        # else:
                        #     matrix = np.load(pathTmp + '_'.join((targetVar, predName, sceneName, 'standardized_matrix.npy')))
                        #     iseason = 0
                        #     for season in season_dict:
                        #         fig, ax = plt.subplots(figsize=(8,6), dpi = 300)
                        #         for imodel in range(nmodels):
                        #             model = model_list[imodel]
                        #             data = matrix[iseason, iscene-1, imodel]
                        #             data = np.nanmean(data.reshape(nYears, -1), axis=1)
                        #             data = gaussian_filter1d(data, 5)
                        #             plt.plot(years, data, label=model)
                        #         plt.title(' '.join((predName, sceneName, season)))
                        #         ax.set_ylabel('standardized ' + predName)
                        #         plt.legend()
                        #         # plt.show()
                        #         # exit()
                        #         filename = '_'.join((experiment, 'evolSpaghetti', targetVar, predName, sceneName, season))
                        #         plt.savefig(pathOut + filename)
                        #         plt.close()
                        #
                        #         iseason += 1

########################################################################################################################
def GCMs_evaluation():
    """
    Evaluate GCMs in a historical period and in the future
    """

    print('GCMs_evaluation...')

    GCMs_evaluation_historical()
    GCMs_evaluation_future()