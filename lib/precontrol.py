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
def predictors_strength():
    """
    Test the strength of the predictors/predictand relationships. It can be used to select the most relevant predictors
    for the downscaling.
    """

    print('predictors_strength...')

    # For interpolation
    interp_mode = 'bilinear'
    i_4nn = np.load(pathAux + 'ASSOCIATION/' + interp_mode + '/i_4nn.npy')
    j_4nn = np.load(pathAux + 'ASSOCIATION/' + interp_mode + '/j_4nn.npy')
    w_4nn = np.load(pathAux + 'ASSOCIATION/' + interp_mode + '/w_4nn.npy')

    # Go through all target variables
    for var in ('tmax', 'tmin', 'pcp', ):

        # Define pathTmp
        pathTmp = '../results/' + experiment + '/predictors_strength/' + var.upper() + '/'
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        pathOut = pathFigures
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Read data predictand
        obs = read.hres_data(var, period='calibration')['data']

        # Define preds
        if var == 'pcp':
            preds = preds_p
        else:
            preds = preds_t
        npreds = len(preds)

        # Go through all seasons
        for season in season_dict.values():

            R = np.zeros((npreds, hres_npoints))

            # Calculate correlations for each predictor
            for ipred in range(npreds):
                predName = list(preds.keys())[ipred]

                # Read data predictor
                data = read.lres_data(var, 'pred', predName=predName)['data']

                # Select season
                data_season = postpro_lib.get_season(data, calibration_dates, season)['data']
                obs_season = postpro_lib.get_season(obs, calibration_dates, season)['data']

                # Go through all points
                for ipoint in range(hres_npoints):

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
                filename = '_'.join((experiment, 'correlationMap', 'daily', var, predName, 'None', season))
                plot.map(abs(R[ipred]), 'correlation', path=pathOut, filename=filename, title=title)

            # Boxplot
            fig, ax = plt.subplots()
            ax.boxplot(abs(R.T), showfliers=False)
            ax.set_xticklabels(list(preds.keys()), rotation=90)
            plt.ylim((0, 1))
            plt.title(' '.join((var.upper(), 'correlation', season)))
            # plt.show()
            # exit()
            filename = '_'.join((experiment, 'correlationBoxplot', 'daily', var, 'None', 'None', season))
            plt.savefig(pathOut + filename)





########################################################################################################################
def GCMs_availability():
    """
    Check for missing data in predictors by GCMs. It can be used to discard some predictors/levels.
    """

    print('GCMs_availability...')

    # Go through all target variables
    for var0 in ('t', 'p', ):

        # Define pathTmp
        pathTmp = '../results/' + experiment + '/GCMs_availability/' + var0.upper() + '/'
        if not os.path.exists(pathTmp):
            os.makedirs(pathTmp)
        pathOut = pathFigures
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

        # Define preds
        if var0 == 'p':
            preds = preds_p
        else:
            preds = preds_t
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
                        filename = '_'.join((experiment, 'nansMap', 'daily', var0, predName, model+'-'+sceneName, 'None'))
                        title = ' '.join((predName, model, sceneName, 'pertentage of NANs'))
                        plot.map(perc_nan, 'perc_nan', grid='pred', path=pathOut, filename=filename, title=title)

                    imodel += 1
                iscene += 1

        # Save results
        np.save(pathTmp+'PERC_NAN', PERC_NAN)
        PERC_NAN = np.load(pathTmp+'PERC_NAN.npy')

        # Plot heatmaps
        nscenes = len(scene_names_list)
        predNames = [list(preds.keys())[i] for i in range(npreds)]
        modelNames = model_names_list

        # Go through all scenes
        for iscene in range(nscenes):
            sceneName = scene_names_list[iscene]
            matrix = np.mean(PERC_NAN[:, iscene, :].reshape(npreds, nmodels, -1), axis=2).T

            xticklabels = predNames
            g = sns.heatmap(matrix, annot=True, vmin=0, vmax=100, fmt='.1f',
                            cbar_kws={'label': '%'},
                            xticklabels=xticklabels, yticklabels=True, square=True, cmap='RdYlGn_r')
            g.tick_params(left=False, bottom=False)
            g.yaxis.set_label_position("right")
            g.set_yticklabels(modelNames, rotation=0)
            g.set_xticklabels(predNames, rotation=90)
            plt.title(sceneName + ' pertentage of NANs')
            # plt.show()
            # exit()
            filename = '_'.join((experiment, 'nansMatrix', 'daily', var0, 'None', sceneName, 'None.png'))
            plt.savefig(pathOut + filename)
            plt.close()



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
        if var0 == 'p':
            preds = preds_p
        else:
            preds = preds_t
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

                # Go through all models
                for imodel in range(nmodels):
                    model = model_list[imodel]

                    # Read model
                    calendar = read.netCDF('../input_data/models/',
                                           'psl_' + model + '_' + sceneName + '_' + modelRealizationFilename + '_' +
                                           historicalPeriodFilename + '.nc', 'psl')['calendar']
                    aux = read.lres_data(var0, 'pred', model=model, scene=sceneName, predName=predName)
                    scene_dates = aux['times']
                    if calendar == '360':
                        time_first, time_last = scene_dates.index(reference_first_date), -1
                    else:
                        time_first, time_last = scene_dates.index(reference_first_date), scene_dates.index(
                            reference_last_date) + 1
                    sceneData = aux['data']
                    sceneData = sceneData[time_first:time_last]
                    scene_dates = scene_dates[time_first:time_last]

                    # Select season data
                    rea_season = postpro_lib.get_season(rea, reference_dates, season)['data']
                    sceneData_season = postpro_lib.get_season(sceneData, scene_dates, season)['data']

                    # Calculate mean values and absolute bias
                    rea_mean_season = np.nanmean(rea_season[:, 0, :, :], axis=0)
                    sceneData_mean_season = np.nanmean(sceneData_season[:, 0, :, :], axis=0)
                    bias = sceneData_mean_season - rea_mean_season

                    # Save results
                    np.save(pathTmp + '_'.join((var0, predName, model, sceneName, season, 'bias')), bias)
                    print(var0, predName, model, sceneName, season)

                    # # Plot maps
                    # filename = '_'.join((experiment, 'reaMap', 'all', var0, predName, 'None', season))
                    # title = ' '.join((predName, 'reanalysis'))
                    # plot.map(rea_mean_season, grid='pred', path=pathOut, filename=filename, title=title)
                    # filename = '_'.join((experiment, 'modMap', 'all', var0, predName, model, season))
                    # title = ' '.join((predName, model))
                    # plot.map(sceneData_mean_season, grid='pred', path=pathOut, filename=filename, title=title)
                    # bias = np.load(pathTmp + '_'.join((var0, predName, model, sceneName, season, 'bias.npy')))
                    # filename = '_'.join((experiment, 'biasMap', 'all', var0, predName, model, season))
                    # title = ' '.join((predName, model, 'bias'))
                    # plot.map(bias, grid='pred', path=pathOut, filename=filename, title=title)

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
                # plt.show()
                # exit()
                filename = '_'.join((experiment, 'biasBoxplot', 'all', var0, predName, 'None', season))
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
        if var0 == 'p':
            preds = preds_p
        else:
            preds = preds_t
        npreds = len(preds)

        # Go through all predictors
        for ipred in range(npreds):
            predName = list(preds.keys())[ipred]

            # Go through all scenes
            for iscene in range(1, nscenes + 1):
                sceneName = scene_list[iscene]

                if sceneName != 'historical':

                    matrix = np.zeros((nseasons, nscenes, nmodels, nYears, nlats, nlons))

                    # Go through all models and storage mean values
                    for imodel in range(nmodels):
                        model = model_list[imodel]
                        print(var0, predName, sceneName, model, 'evaluation future')

                        # Read model scene
                        aux = read.lres_data(var0, 'pred', model=model, scene=sceneName, predName=predName)
                        times = aux['times']
                        sceneData = aux['data'][:, 0, :, :]

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
                            iseason += 1

                    # Save results
                    np.save(pathTmp + '_'.join((var0, predName, sceneName, 'matrix')), matrix)

                    matrix = np.load(pathTmp + '_'.join((var0, predName, sceneName, 'matrix.npy')))
                    iseason = 0
                    for season in season_dict.values():
                        for imodel in range(nmodels):
                            model = model = model_list[imodel]
                            data = matrix[iseason, iscene-1, imodel]
                            data = np.mean(data.reshape(nYears, -1), axis=1)
                            data = gaussian_filter1d(data, 8)
                            plt.plot(years, data, label=model)
                        plt.title(' '.join((predName, sceneName, season)))
                        plt.legend()
                        # plt.show()
                        # exit()
                        filename = '_'.join((experiment, 'evolSpaghetti', 'all', var0, predName, sceneName, season))
                        plt.savefig(pathOut + filename)
                        plt.close()

                        iseason += 1
