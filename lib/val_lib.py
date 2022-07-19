import sys
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import derived_predictors
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
import MOS_lib
import plot
import postpro_lib
import postprocess
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
def daily_boxplots(metric, by_season):
    """
    Boxplots of metric, by seasons (optional) and by subregions (optional).
    :param metric: correlation or variance
    :param methods:
    :param by_season: boolean
    """

    vars = []
    for method in methods:
        var = method['var']
        if var not in vars:
            vars.append(var)

    # Go through all variables
    for targetVar in targetVars:
        nmethods = len([x for x in methods if x['var'] == targetVar])

        # Go through all methods
        imethod = 0
        for method_dict in methods:
            var = method_dict['var']
            if var == targetVar:
                methodName = method_dict['methodName']
                print(var, methodName, metric)

                # Read data
                d = postpro_lib.get_data_eval(var, methodName)
                ref, times_ref, obs, est, times_scene = d['ref'], d['times_ref'], d['obs'], d['est'], d['times_scene']
                del d

                # Select season
                for season in season_dict:
                    if season == annualName or by_season == True:
                        if season == season_dict[annualName]:
                            obs_season = obs
                            est_season = est
                            times = times_scene
                        else:
                            obs_season = postpro_lib.get_season(obs, times_scene, season)['data']
                            aux = postpro_lib.get_season(est, times_scene, season)
                            est_season = aux['data']
                            times = aux['times']
                        ivalid = [i for i in range(len(times)) if
                                  (np.count_nonzero(np.isnan(obs_season[i])) == 0 and
                                                    np.count_nonzero(np.isnan(est_season)[i]) == 0)]
                        obs_season = obs_season[ivalid]
                        est_season = est_season[ivalid]
                        matrix = np.zeros((hres_npoints[targetVar], ))
                        if metric == 'correlation':
                            for ipoint in range(hres_npoints[targetVar]):
                                X = obs_season[:, ipoint]
                                Y = est_season[:, ipoint]
                                if var == 'pr' or (targetVar == myTargetVar and myTargetVarIsGaussian == False):
                                    r = round(spearmanr(X, Y)[0], 3)
                                else:
                                    r = round(pearsonr(X, Y)[0], 3)
                                if np.isnan(r) == True:
                                    r = 0
                                matrix[ipoint] = r
                        elif metric == 'variance':
                            obs_var = np.var(obs_season, axis=0)
                            est_var = np.var(est_season, axis=0)
                            th = 0.001
                            est_var[est_var < th] = 0
                            obs_var[obs_var < th] = np.nan
                            bias = 100 * (est_var - obs_var) / obs_var
                            bias[(np.isnan(obs_var)) * (est_var == 0)] = 0
                            bias[np.isinf(bias)] = np.nan
                            matrix[:] = bias
                        elif metric == 'rmse':
                            matrix = np.round(np.sqrt(np.nanmean((est_season - obs_season) ** 2, axis=0)), 2)
                        np.save('../tmp/'+targetVar+'_'+methodName+'_'+season+'_' +metric, matrix)
                imethod += 1


    # Select season
    for season in season_dict:
        if season == annualName or by_season == True:
            for targetVar in vars:
            # for targetVar in ('pcp', ):
                nmethods = len([x for x in methods if x['var'] == targetVar])
                # Read regions csv
                df_reg = pd.read_csv(pathAux + 'ASSOCIATION/' + targetVar.upper() +'/regions.csv')
                matrix = np.zeros((hres_npoints[targetVar], nmethods))
                imethod = 0
                names = []
                for method_dict in methods:
                    var = method_dict['var']
                    if var == targetVar:
                        methodName = method_dict['methodName']
                        names.append(methodName)
                        print(metric, season, var, methodName)
                        matrix[:, imethod] = np.load('../tmp/' + targetVar + '_' + methodName + '_' + season + '_'+metric+'.npy')
                        imethod += 1


                # Go through all regions
                for index, row in df_reg.iterrows():
                    if plotAllRegions == True or ((plotAllRegions == False) and (index == 0)):
                        regType, regName, subDir = row['regType'], row['regName'], row['subDir']
                        iaux = [int(x) for x in row['ipoints'][1:-1].split(', ')]
                        npoints = len(iaux)
                        print(regType, regName, npoints, 'points', str(index) + '/' + str(df_reg.shape[0]))
                        matrix_region = matrix[iaux]

                        # Create pathOut
                        if plotAllRegions == False:
                            pathOut = pathFigures
                        else:
                            path = pathFigures + 'daily_'+metric+'/' + targetVar.upper() + '/'
                            pathOut = path + subDir
                        if not os.path.exists(pathOut):
                            os.makedirs(pathOut)

                        if metric == 'correlation':
                            units = ''
                        elif metric == 'variance':
                            units = '%'
                        elif metric == 'rmse':
                            units = predictands_units[targetVar]

                        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                        medianprops = dict(color="black")
                        g = ax.boxplot(matrix_region, showfliers=False, patch_artist=True, medianprops=medianprops)
                        # plt.ylim((-.2, 1))
                        # fill with colors
                        i = 0
                        color = [methods_colors[x['methodName']] for x in methods if x['var'] == targetVar]
                        for patch in g['boxes']:
                            patch.set_facecolor(color[i])
                            i += 1
                        if metric == 'correlation':
                            title = ' '.join((targetVar, metric, season))
                        elif metric == 'variance':
                            title = ' '.join((targetVar, 'bias', metric, season))
                        elif metric == 'rmse':
                            title = ' '.join((targetVar, metric, season))
                            # title = targetVar
                        # plt.title(title, fontsize=20)
                        plt.title(title)
                        plt.ylabel(units, rotation=90)
                        ax.set_xticklabels(names, rotation=90)
                        if metric == 'variance':
                            plt.hlines(y=0, xmin=-1, xmax=nmethods+1, linestyles='--', color='grey')
                        else:
                            plt.hlines(y=0.5, xmin=-1, xmax=nmethods+1, linewidth=0)
                        # plt.show()
                        # exit()
                        plt.savefig(pathOut + '_'.join(('EVALUATION'+bc_sufix, metric+'Boxplot', targetVar, 'None', 'all',
                                                        season))+ '.png', bbox_inches='tight')
                        plt.close()


########################################################################################################################
def climdex_boxplots(by_season):
    """
    Bias boxplots for all climdex, by seasons (optional) and by subregions (optional).
    :param methods:
    :param by_season: boolean
    """



    vars = []
    for method in methods:
        var = method['var']
        if var not in vars:
            vars.append(var)

    # Go through all variables
    for targetVar in vars:
        nmethods = len([x for x in methods if x['var'] == targetVar])

        # Read regions csv
        df_reg = pd.read_csv(pathAux + 'ASSOCIATION/'+targetVar.upper()+'/regions.csv')

        # Go through all climdex
        for climdex_name in climdex_names[targetVar]:

            # Select season
            for season in season_dict:
                if season == annualName or by_season == True:

                    # Go through all methods
                    imethod = 0
                    names = []
                    matrix = np.zeros((hres_npoints[targetVar], nmethods))
                    for method_dict in methods:
                        var = method_dict['var']
                        if var == targetVar:
                            methodName = method_dict['methodName']
                            names.append(methodName)
                            print(var, climdex_name, season, methodName)

                            pathIn = '../results/EVALUATION'+bc_sufix+'/'+targetVar.upper()+'/'+methodName+'/climdex/'
                            obs = np.nanmean(np.load(pathIn + '_'.join((climdex_name, 'obs', season))+'.npy'), axis=0)
                            est = np.nanmean(np.load(pathIn + '_'.join((climdex_name, 'est', season))+'.npy'), axis=0)

                            biasMode = units_and_biasMode_climdex[targetVar + '_' + climdex_name]['biasMode']
                            if biasMode == 'abs':
                                units = units_and_biasMode_climdex[targetVar + '_' + climdex_name]['units']
                                bias = est - obs
                            elif biasMode == 'rel':
                                units = '%'
                                th = 0.001
                                est[est < th] = 0
                                obs[obs < th] = 0
                                bias = 100 * (est - obs) / obs
                                bias[(obs == 0) * (est == 0)] = 0
                                bias[np.isinf(bias)] = np.nan
                            matrix[:, imethod] = bias
                            imethod += 1


                    # Go through all regions
                    for index, row in df_reg.iterrows():
                        if plotAllRegions == True or ((plotAllRegions == False) and (index == 0)):
                            regType, regName, subDir = row['regType'], row['regName'], row['subDir']
                            iaux = [int(x) for x in row['ipoints'][1:-1].split(', ')]
                            npoints = len(iaux)
                            print(regType, regName, npoints, 'points', str(index) + '/' + str(df_reg.shape[0]))
                            matrix_region = matrix[iaux]

                            # Create pathOut
                            if plotAllRegions == False:
                                pathOut = pathFigures
                            else:
                                path = pathFigures + 'biasBoxplot/' + targetVar.upper() + '/'
                                pathOut = path + subDir
                            if not os.path.exists(pathOut):
                                os.makedirs(pathOut)

                            fig, ax = plt.subplots(dpi=300)
                            medianprops = dict(color="black")
                            g = ax.boxplot(matrix_region, showfliers=False, patch_artist=True, medianprops=medianprops)
                            # plt.ylim((-.2, 1))
                            # fill with colors
                            i = 0
                            color = [methods_colors[x['methodName']] for x in methods if x['var'] == targetVar]
                            for patch in g['boxes']:
                                patch.set_facecolor(color[i])
                                i += 1
                            # if biasMode == 'rel':
                            #     plt.ylim((-100, 100))
                            if climdex_name == 'FWI90p':
                                plt.ylim((-60, 60))
                            # title = ' '.join((targetVar.upper(), climdex_name, 'bias', season))
                            # plt.title(title)
                            title = ' '.join((targetVar, climdex_name))
                            plt.title(title, fontsize=20)
                            ax.set_xticklabels(names, rotation=90)
                            plt.hlines(y=0, xmin=-1, xmax=nmethods + 1, linestyles='--', color='grey')
                            plt.ylabel(units, rotation=90)
                            # plt.show()
                            # exit()

                            filename = '_'.join(('EVALUATION'+bc_sufix, 'biasClimdexBoxplot', targetVar, climdex_name, 'all',
                                                 season))
                            plt.savefig(pathOut + filename + '.png', bbox_inches='tight')
                            plt.close()




########################################################################################################################
def monthly_maps(metric, targetVar, methodName):
    """
    Correlation or R2_score maps for monthly accumulated precipitation.
    metric: correlation or R2
    """

    print('monthly', metric, targetVar, methodName)

    # Read data
    d = postpro_lib.get_data_eval(targetVar, methodName)
    ref, times_ref, obs, est, times_scene = d['ref'], d['times_ref'], d['obs'], d['est'], d['times_scene']
    del d
    npoints = obs.shape[1]
    firstYear = times_scene[0].year
    lastYear = times_scene[-1].year

    dates = []
    for year in range(firstYear, lastYear + 1):
        for month in range(1, 13):
            dates.append((month, year))
    ndates = len(dates)
    est_acc = np.zeros((ndates, npoints))
    obs_acc = np.zeros((ndates, npoints))

    for idate in range(ndates):
        month, year = dates[idate][0], dates[idate][1]
        idates = [i for i in range(len(times_scene)) if ((times_scene[i].year == year) and (times_scene[i].month == month))]
        obs_acc[idate] = np.nansum(obs[idates], axis=0)
        est_acc[idate] = np.nansum(est[idates], axis=0)


    filename = '_'.join(('EVALUATION'+bc_sufix, metric+'MapMonthly', targetVar, 'None', methodName, 'None'))
    # Correlation
    if metric == 'correlation':
        title = ' '.join(('monthly', metric, targetVar.upper(), methodName))
        r = np.zeros((npoints,))
        for ipoint in range(npoints):
            r[ipoint] = pearsonr(obs_acc[:, ipoint], est_acc[:, ipoint])[0]
        plot.map(targetVar, r, 'corrMonth', path=pathFigures, filename=filename, title=title, regType=None, regName=None)

    # R2_score
    if metric == 'R2':
        title = ' '.join(('monthly', metric.upper() +'_score', targetVar.upper(), methodName))
        R2 = 1 - np.sum((est_acc-obs_acc)**2, axis=0) / np.sum(obs_acc**2, axis=0)
        plot.map(targetVar, R2, 'r2', path=pathFigures, filename=filename, title=title, regType=None, regName=None)


########################################################################################################################
def QQplot(targetVar, methodName, obs, est, pathOut, season):
    '''
    Save scatter plot of several percentiles of daily data distribution
    '''

    # Create pathOut
    if plotAllRegions == False:
        pathOut = pathFigures
    else:
        pathOut += 'QQplot/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)

    units = predictands_units[targetVar]

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
        print('perc', p)
        px, py = np.nanpercentile(obs, p, axis=0), np.nanpercentile(est, p, axis=0)
        plt.plot(px, py, '+', c=c, label='p'+str(p))
        # plt.plot(px, py, '+', c=c, markersize=2, label='p'+str(p))
        M = int(max(np.max(px), np.max(py), M))
        m = int(min(np.min(px), np.min(py), m))
    plt.xlim(m, M)
    plt.ylim(m, M)
    plt.xlabel('observed (' + units + ')')
    plt.ylabel('downscaled (' + units + ')')
    h = []
    for i in range(len(c_list)):
        h.append(Line2D([0], [0], marker='o', markersize=np.sqrt(20), color=c_list[i], linestyle='None'))
    plt.legend(h, ['p'+str(x) for x in perc_list], markerscale=2, scatterpoints=1, fontsize=10)
    m -= 5
    M += 5
    plt.plot(range(m, M), range(m, M))
    title = ' '.join((targetVar.upper(), methodName, season))
    plt.title(title)
    # plt.show()
    # exit()
    filename = '_'.join(('EVALUATION'+bc_sufix, 'qqPlot', targetVar, 'None', methodName, season)) + '.png'
    plt.savefig(pathOut + filename)
    plt.close()


########################################################################################################################
def continuous(targetVar, methodName, obs, est, pathOut, season):
    '''
    Plots the following figures for the whole testing period:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R2 score
    - Hist2d obs vs est (performed on a selection of 5000 random points (otherwise memory limit might be exceeded)
    - Hist2d sorted distributions (performed on a selection of 5000 random points (otherwise memory limit might be exceeded)
    '''

    print('validate continuous scores', targetVar, methodName)

    if plotAllRegions == False:
        pathOut = pathFigures
        subtitle = ''
        filename = '_'.join((targetVar, methodName, season))
    else:
        pathOut += 'scores_continuous/'
        subtitle = targetVar + ' ' + methodName + '\n' + season
        filename = '_'.join((targetVar, methodName, season))

    # # MAE
    # filename = '_'.join(('EVALUATION'+bc_sufix, 'maeMap'', targetVar, 'None', methodName, season))
    # MAE = np.round(np.nanmean(abs(est - obs), axis=0), 2)
    # plot.map(targetVar], MAE,  targetVar]+'_mae', path=pathOut, filename='MAE_' + filename, title='')

    if targetVar == 't':
        # RMSE
        filename = '_'.join(('EVALUATION'+bc_sufix, 'rmseMap', targetVar, 'None', methodName, season))
        title = ' '.join(('daily RMSE', targetVar.upper(), methodName, season))
        RMSE = np.round(np.sqrt(np.nanmean((est - obs) ** 2, axis=0)), 2)
        plot.map(targetVar, RMSE,  targetVar+'_rmse', path=pathOut, filename=filename, title=title)
    else:
        # # R2_score
        filename = '_'.join(('EVALUATION'+bc_sufix, 'r2Map', targetVar, 'None', methodName, season))
        title = ' '.join(('daily R2_score', targetVar.upper(), methodName, season))
        R2 = 1 - np.nansum((obs-est)**2, axis=0) / np.nansum((obs-np.nanmean(obs, axis=0))**2, axis=0)
        plot.map(targetVar, R2,  'r2', path=pathOut, filename=filename, title=title)



########################################################################################################################
def dichotomous(targetVar, methodName, obs, est, pathOut, season):
    '''
    Plots the following figures for the whole testing period:
    Accuracy (proportion of correct classified)
    '''
    print('validate dichotomous scores', targetVar, methodName)

    if plotAllRegions == False:
        pathOut = pathFigures
        filename = '_'.join((targetVar, methodName, season))
    else:
        pathOut += 'scores_continuous/'
        filename = '_'.join((targetVar, methodName, season))

    # Calculate hits, misses, false_alarms and correct_negatives.
    # Conditions are written this way so they handle np.nan properly
    hits = 1.*np.sum(((obs>=wetDry_th) * (est>=wetDry_th)), axis=0)
    misses = 1.*np.sum(((obs>=wetDry_th) * (est<wetDry_th)), axis=0)
    false_alarms = 1.*np.sum(((obs<wetDry_th) * (est>=wetDry_th)), axis=0)
    correct_negatives = 1.*np.sum(((obs<wetDry_th) * (est<wetDry_th)), axis=0)

    # Prevent zero division errors
    hits[hits == 0] = 0.001
    misses[misses == 0] = 0.001
    false_alarms[false_alarms == 0] = 0.001
    correct_negatives[correct_negatives == 0] = 0.001

    # Accuracy score
    filename = '_'.join(('EVALUATION'+bc_sufix, 'accuracyMap', targetVar, 'None', methodName,
                                season))
    title = ' '.join(('Daily accuracy_score', targetVar.upper(), methodName, season))
    accuracy = (hits+correct_negatives) / (hits+correct_negatives+misses+false_alarms)
    plot.map(targetVar, accuracy,  'acc', path=pathOut, filename=filename, title=title)

