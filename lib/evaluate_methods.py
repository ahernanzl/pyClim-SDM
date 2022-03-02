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
import evaluate_GCMs
import evaluate_methods
import grids
import launch_jobs
import plot
import postpro_lib
import postprocess
import derived_predictors
import preprocess
import process
import read
import standardization
import TF_lib
import val_lib
import WG_lib
import write


########################################################################################################################
def annual_cycle():
    """
    Plots annual cycle by subregions (optional) for all methods together.
    """


    for VAR in ('pcp', 'tmax', 'tmin', ):

        nmethods = len([x for x in methods if x['var'] == VAR])

        # Create empty array to accumulate results
        obs_matrix = np.zeros((nmethods, 12, hres_npoints))
        est_matrix = np.zeros((nmethods, 12, hres_npoints))

        # Define plot style
        if VAR[0] == 't':
            units = degree_sign
            colors = t_methods_colors
            linestyles = t_methods_linestyles
        else:
            units = 'mm'
            colors = p_methods_colors
            linestyles = p_methods_linestyles

        # Go through all methods
        imethod = 0
        for method_dict in methods:
            var = method_dict['var']
            if var == VAR:
                methodName = method_dict['methodName']
                print(var, methodName)

                # Read data
                d = postpro_lib.get_data_eval(var, methodName)
                ref, times_ref, obs, est, times_scene = d['ref'], d['times_ref'], d['obs'], d['est'], d['times_scene']
                del d

                nYears = 1 + times_scene[-1].year - times_scene[0].year

                # Calculate monthly data
                for imonth in range(12):
                    idates = [i for i in range(len(times_scene)) if times_scene[i].month == imonth+1]
                    if var[0] == 't':
                        obs_matrix[imethod, imonth] = np.mean(obs[idates], axis=0)
                        est_matrix[imethod, imonth] = np.mean(est[idates], axis=0)
                    else:
                        obs_matrix[imethod, imonth] = np.sum(obs[idates], axis=0) / nYears
                        est_matrix[imethod, imonth] = np.sum(est[idates], axis=0) / nYears

                imethod += 1

        np.save('../tmp/obs_matrix_'+VAR, obs_matrix)
        np.save('../tmp/est_matrix_'+VAR, est_matrix)
        obs_matrix = np.load('../tmp/obs_matrix_'+VAR+'.npy')
        est_matrix = np.load('../tmp/est_matrix_'+VAR+'.npy')

        # Read regions csv
        df_reg = pd.read_csv(pathAux + 'ASSOCIATION/regions.csv')

        # Go through all regions
        for index, row in df_reg.iterrows():
            if plotAllRegions == True or ((plotAllRegions == False) and (index == 0)):
                regType, regName, subDir = row['regType'], row['regName'], row['subDir']
                iaux = [int(x) for x in row['ipoints'][1:-1].split(', ')]
                npoints = len(iaux)
                print(regType, regName, npoints, str(index) + '/' + str(df_reg.shape[0]))

                # Create pathOut
                if plotAllRegions == False:
                    pathOut = pathFigures
                else:
                    path = pathFigures + 'annual_cycle/' + VAR.upper() + '/'
                    pathOut = path + subDir
                if not os.path.exists(pathOut):
                    os.makedirs(pathOut)

                # Select region
                if regType == typeCompleteRegion:
                    obs_reg = np.mean(obs_matrix, axis=2)
                    est_reg = np.mean(est_matrix, axis=2)
                else:
                    obs_reg = np.mean(obs_matrix[:, :, iaux], axis=2)
                    est_reg = np.mean(est_matrix[:, :, iaux], axis=2)

                # Plot annual_cycle
                imethod = 0
                for method_dict in methods:
                    var = method_dict['var']
                    if var == VAR:
                        methodName = method_dict['methodName']
                        if imethod == 0:
                            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                            plt.plot(obs_reg[imethod], c='k', label='OBS')
                        plt.plot(est_reg[imethod], c=colors[methodName], label=methodName, linestyle=linestyles[methodName])
                        imethod += 1

                        if imethod == nmethods:
                            if var[0] == 't':
                                if nmethods > 10:
                                    plt.legend(ncol=1, bbox_to_anchor = (1.05, .99), fontsize=18)
                                else:
                                    plt.legend()
                                title_size = 25
                            else:
                                if nmethods > 10:
                                    plt.legend(ncol=3, bbox_to_anchor = (0.06, 1.1))
                                else:
                                    plt.legend()
                                title_size = 15

                            plt.title(var.upper(), fontsize=title_size)
                            plt.xticks(ticks=range(12), labels=range(1, 13))
                            plt.xlabel('month')
                            plt.ylabel(units)
                            plt.tight_layout()
                            # plt.show()
                            # exit()
                            plt.savefig(pathOut + '_'.join(('EVALUATION', 'annualCycle', 'None', var, 'None', 'all', 'None')))
                            plt.close()


########################################################################################################################
def daily_data(by_season=True):
    """
    Plots boxplots (correlation and variance) of all methods, and daily maps, QQplots, continuous and dichotomous for
    each method.
    """

    # Correlation and biasVariance boxplots of all methods together
    val_lib.daily_boxplots('correlation', by_season)
    val_lib.daily_boxplots('variance', by_season)

    # Go through all methods
    for method_dict in methods:
        var, methodName = method_dict['var'], method_dict['methodName']
        path = '../results/EVALUATION/' + var.upper() + '/' + methodName + '/daily_data/'

        # Read data
        d = postpro_lib.get_data_eval(var, methodName)
        ref, times_ref, obs, est, times_scene = d['ref'], d['times_ref'], d['obs'], d['est'], d['times_scene']
        del d

        # Read regions csv
        df_reg = pd.read_csv(pathAux+'ASSOCIATION/regions.csv')

        # Go through all regions
        for index, row in df_reg.iterrows():
            if plotAllRegions == True or ((plotAllRegions == False) and (index == 0)):
                regType, regName, subDir = row['regType'], row['regName'], row['subDir']
                iaux = [int(x) for x in row['ipoints'][1:-1].split(', ')]
                npoints = len(iaux)
                print(regType, regName, npoints, str(index) + '/' + str(df_reg.shape[0]))

                # Create pathOut
                if plotAllRegions == False:
                    pathOut = pathFigures
                else:
                    pathOut = pathFigures + 'daily_data/' + var.upper() + '/' + subDir
                    if not os.path.exists(pathOut):
                        os.makedirs(pathOut)

                # Select region
                if regType == typeCompleteRegion:
                    obs_reg = obs
                    est_reg = est
                else:
                    obs_reg = obs[:, iaux]
                    est_reg = est[:, iaux]

                # Select season
                for season in season_dict.values():
                    if season == 'ANNUAL' or by_season == True:
                        print(season)
                        if season == season_dict['ANNUAL']:
                            obs_reg_season = obs_reg
                            est_reg_season = est_reg
                            times = times_scene
                        else:
                            obs_reg_season = postpro_lib.get_season(obs_reg, times_scene, season)['data']
                            aux = postpro_lib.get_season(est_reg, times_scene, season)
                            est_reg_season = aux['data']
                            times = aux['times']

                        # Validation of daily data
                        val_lib.QQplot(var, methodName, obs_reg_season, est_reg_season, pathOut, season)
                        if regType == typeCompleteRegion:
                            val_lib.continuous(var, methodName, obs_reg_season, est_reg_season, pathOut, season)
                            if var == 'pcp':
                                val_lib.dichotomous(var, methodName, obs_reg_season, est_reg_season, pathOut, season)


########################################################################################################################
def monthly_data():
    """
    Plots monthly accumulated correlation maps of pcp.
    """

    # Go through all methods
    for method_dict in methods:
        var, methodName = method_dict['var'], method_dict['methodName']

        # Monthly correlation and R2 for precipitation
        if var == 'pcp':
            val_lib.monthly_maps('correlation', var, methodName)
            val_lib.monthly_maps('R2', var, methodName)


########################################################################################################################
def climdex(by_season=True):
    """
    Plots bias boxplots of all methods, and bias maps of mean climdex and scatter plot mean climdex for  each method.
    """

    # Bias boxplots of all methods together
    val_lib.climdex_boxplots(by_season)

    # Go through all methods
    for method_dict in methods:
        var, methodName = method_dict['var'], method_dict['methodName']

        # bias_type = 'abs'
        bias_type = 'rel'

        # Read regions csv
        df_reg = pd.read_csv(pathAux+'ASSOCIATION/regions.csv')

        # Go through all regions
        for index, row in df_reg.iterrows():
            if plotAllRegions == True or ((plotAllRegions == False) and (index == 0)):
                regType, regName, subDir = row['regType'], row['regName'], row['subDir']
                iaux = [int(x) for x in row['ipoints'][1:-1].split(', ')]
                npoints = len(iaux)
                print(regType, regName, npoints, str(index) + '/' + str(df_reg.shape[0]))

                # Select climdex
                for climdex_name in climdex_names[var]:
                    print(climdex_name)

                    # Define palette and units
                    if climdex_name in ('TXm', 'TNm', 'TXx', 'TNx', 'TXn', 'TNn', ):
                        units = degree_sign
                    elif climdex_name in ('TX90p', 'TX10p', 'TN90p', 'TN10p'):
                        units = '%'
                    elif climdex_name in ('FD', 'WSDI', 'CSDI', 'R01', 'CDD', 'CWD'):
                        units = 'days'
                    elif climdex_name in ('Pm', 'R95p', 'R95pFRAC', 'PRCPTOT', 'SDII'):
                        units = 'mm'
                    elif climdex_name in ('p1', 'p5', 'p10', 'p90', 'p95', 'p99'):
                        if var == 'pcp':
                            units = 'mm'
                        else:
                            units = degree_sign
                    else:
                        units = ''
                        print('define climdex units')
                        # exit()

                    # Select season
                    for season in season_dict.values():
                        if season == 'ANNUAL' or by_season == True:
                            # Read data and select region
                            pathIn = '../results/EVALUATION/' + var.upper() + '/' + methodName + '/climdex/'

                            # Create pathOut
                            if plotAllRegions == False:
                                pathOut = pathFigures
                            else:
                                path = pathFigures + 'biasScatterPlot/' + var.upper() + '/'
                                pathOut = path + subDir
                                if not os.path.exists(pathOut):
                                    os.makedirs(pathOut)

                            obs_climdex = np.load(pathIn + '_'.join((climdex_name, 'obs', season)) + '.npy')[:, iaux]
                            est_climdex = np.load(pathIn + '_'.join((climdex_name, 'est', season)) + '.npy')[:, iaux]

                            # Calculate mean for obs  and est (remember that climdex whith no annaul value, such as p10, p95, etc)
                            # have the same value for all years
                            mean_obs = np.mean(obs_climdex, axis=0)
                            mean_est = np.mean(est_climdex, axis=0)
                            if ((bias_type == 'abs') or (var[0] == 't')):
                                bias = mean_est - mean_obs
                            else:
                                mean_obs[mean_obs==0] = 0.001
                                bias = 100 * (mean_est - mean_obs) / mean_obs

                            #-------------------- Bias maps mean values    -----------------------------------------------------
                            if plotAllRegions == False or index == 0:
                                if climdex_name in ('TXm', 'TNm', 'PRCPTOT', 'R01', 'R95p'):

                                    palette = var + '_' + climdex_name
                                    if ((bias_type == 'abs') or (var[0] == 't')):
                                        bias_palette = palette + '_bias'
                                    else:
                                        bias_palette = palette + '_rel_bias'

                                    # Plot obs, est and bias (est-obs) maps
                                    filename = '_'.join(('EVALUATION', 'obsMap', 'climdex', var, climdex_name, methodName,
                                                         season))
                                    plot.map(mean_obs, palette, path=pathFigures, filename=filename, title='')
                                    filename = '_'.join(('EVALUATION', 'estMap', 'climdex', var, climdex_name, methodName,
                                                         season))
                                    plot.map(mean_est, palette, path=pathFigures, filename=filename, title='')
                                    filename = '_'.join(('EVALUATION', 'biasMap', 'climdex', var, climdex_name, methodName,
                                                         season))
                                    plot.map(bias, bias_palette, path=pathFigures, filename=filename, title='')

                            #-------------------- Scatter plot mean values -----------------------------------------------------
                            m = int(min(np.min(mean_obs), np.min(mean_est)))
                            M = int(max(np.max(mean_obs), np.max(mean_est)))
                            # Plot figure
                            fig = plt.figure()
                            ax = fig.add_subplot(111)
                            ax.set_aspect('equal', adjustable='box')
                            # plt.plot(mean_obs, mean_est, '+', c='g', markersize=2)
                            plt.plot(mean_obs, mean_est, '+', c='g')
                            plt.xlim(m, M)
                            plt.ylim(m, M)
                            plt.xlabel('obs (' + units + ')')
                            plt.ylabel('down (' + units + ')')
                            m -= 5
                            M += 5
                            plt.plot(range(m, M), range(m, M))
                            # if plotAllRegions == False and season == season_dict['ANNUAL']:

                            filename = '_'.join(('EVALUATION', 'scatterPlot', 'climdex', var, climdex_name, methodName,
                                                 season))
                            if plotAllRegions == False:
                                # plt.show()
                                # exit()
                                plt.savefig(pathFigures + filename)
                            else:
                                plt.title('_'.join((var, methodName, season)))
                                # plt.show()
                                # exit()
                                plt.savefig(pathOut + filename + '.png')
                            plt.close()

