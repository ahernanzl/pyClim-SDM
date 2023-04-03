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
class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd,
                 fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd  # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi / 2
        tlocs = np.arccos(rlocs)  # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)  # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")  # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)  # Unused

        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l


    def add_arrow(self, stddev_0, corrcoef_0, stddev_1, corrcoef_1, *args, **kwargs):
        """
        Add arrow from point_0 to point_1 (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """
        theta0, radius0 = np.arccos(corrcoef_0), stddev_0
        theta1, radius1 = np.arccos(corrcoef_1), stddev_1
        d_theta, d_radius = theta1-theta0, radius1-radius0

        try:
            theta0, radius0, d_theta, d_radius = theta0[0], radius0[0], d_theta[0], d_radius[0]
        except:
            pass
        self.ax.arrow(theta0, radius0, d_theta, d_radius, *args, **kwargs)  # (theta, radius)

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd ** 2 + rs ** 2 - 2 * self.refstd * rs * np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


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
                        # not_valid_obs = np.where(np.isnan(obs_season))[0]
                        # not_valid_est = np.where(np.isnan(est_season))[0]
                        # ivalid = [i for i in range(len(times)) if (i not in not_valid_obs) and (i not in not_valid_est)]
                        # obs_season = obs_season[ivalid]
                        # est_season = est_season[ivalid]
                        matrix = np.zeros((hres_npoints[targetVar], ))
                        if metric == 'correlation':
                            for ipoint in range(hres_npoints[targetVar]):
                                X = obs_season[:, ipoint]
                                Y = est_season[:, ipoint]
                                try:
                                    if var == 'pr' or (targetVar == myTargetVar and myTargetVarIsGaussian == False):
                                        r = round(spearmanr(X, Y)[0], 3)
                                    else:
                                        r = round(pearsonr(X, Y)[0], 3)
                                except:
                                    r = np.nan
                                if np.isnan(r) == True:
                                    r = 0
                                matrix[ipoint] = r
                        elif metric == 'variance':
                            obs_var = np.nanvar(obs_season, axis=0)
                            est_var = np.nanvar(est_season, axis=0)
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

                        # Deal with nans
                        mask = ~np.isnan(matrix_region)
                        matrix_region = [d[m] for d, m in zip(matrix_region.T, mask.T)]

                        # Create pathOut
                        if plotAllRegions == False:
                            pathOut = pathFigures
                        else:
                            path = pathFigures + 'daily_'+metric+'/' + targetVar.upper() + '/'
                            pathOut = path + subDir
                        if not os.path.exists(pathOut):
                            os.makedirs(pathOut)


                        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                        medianprops = dict(color="black")
                        g = ax.boxplot(matrix_region, showfliers=False, patch_artist=True, medianprops=medianprops)
                        # fill with colors
                        i = 0
                        color = [methods_colors[x['methodName']] for x in methods if x['var'] == targetVar]
                        for patch in g['boxes']:
                            patch.set_facecolor(color[i])
                            i += 1

                        if metric == 'correlation':
                            units = ''
                            # title = ' '.join((targetVar, metric, season))
                            title = targetVar
                            plt.ylim((0, 1))
                        elif metric == 'variance':
                            units = '%'
                            # title = ' '.join((targetVar, 'bias', metric, season))
                            title = targetVar
                        elif metric == 'rmse':
                            units = predictands_units[targetVar]
                            # title = ' '.join((targetVar, metric, season))
                            title = targetVar
                        plt.title(title, fontsize=20)
                        # plt.title(title)
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
def daily_spatial_correlation_boxplots():
    """
    Boxplots of correlation, by seasons (optional) and by subregions (optional).
    :param methods:
    """

    methods.reverse()

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
                print(var, methodName)

                # Read data
                d = postpro_lib.get_data_eval(var, methodName)
                ref, times_ref, obs, est, times_scene = d['ref'], d['times_ref'], d['obs'], d['est'], d['times_scene']
                del d

                # Select season
                for season in season_dict:
                    if season == annualName:
                        if season == season_dict[annualName]:
                            obs_season = obs
                            est_season = est
                            times = times_scene
                        else:
                            obs_season = postpro_lib.get_season(obs, times_scene, season)['data']
                            aux = postpro_lib.get_season(est, times_scene, season)
                            est_season = aux['data']
                            times = aux['times']

                        not_valid_obs = np.where(np.isnan(obs_season))[1]
                        not_valid_est = np.where(np.isnan(est_season))[1]
                        ivalid = [i for i in range(len(times)) if (i not in not_valid_obs) and (i not in not_valid_est)]
                        obs_season = obs_season[ivalid, :]
                        est_season = est_season[ivalid, :]

                        ndays_valid = obs_season.shape[0]
                        matrix = np.zeros((ndays_valid, ))

                        for iday in range(ndays_valid):
                            # if iday % 1000 == 0:
                            #     print(iday, ndays_valid)
                            X = obs_season[iday, :]
                            Y = est_season[iday, :]
                            # not_valid_obs = np.where(np.isnan(X))[0]
                            # not_valid_est = np.where(np.isnan(Y))[0]
                            # # ivalid = [i for i in range(hres_npoints[targetVar])]
                            # # del ivalid[not_valid_obs]
                            # # del ivalid[not_valid_est]
                            # ivalid = [i for i in range(hres_npoints[targetVar]) if
                            #           (i not in not_valid_obs) and (i not in not_valid_est)]
                            # X = X[ivalid]
                            # Y = Y[ivalid]
                            try:
                                if var == 'pr' or (targetVar == myTargetVar and myTargetVarIsGaussian == False):
                                    r = round(spearmanr(X, Y)[0], 3)
                                else:
                                    r = round(pearsonr(X, Y)[0], 3)
                            except:
                                r = np.nan
                            if np.isnan(r) == True:
                                r = 0
                            matrix[iday] = r
                        np.save('../tmp/'+targetVar+'_'+methodName+'_'+season+'_spatial_corr', matrix)
                imethod += 1

    # Select season
    for season in season_dict:
        if season == annualName:
            for targetVar in vars:
                nmethods = len([x for x in methods if x['var'] == targetVar])
                ndays = testing_ndates
                matrix = np.zeros((ndays, nmethods))
                imethod = 0
                names = []
                for method_dict in methods:
                    var = method_dict['var']
                    if var == targetVar:
                        methodName = method_dict['methodName']
                        names.append(methodName)
                        print('spatial correlation', season, var, methodName)
                        aux = np.load('../tmp/' + targetVar + '_' + methodName + '_' + season + '_spatial_corr.npy')
                        matrix[:, imethod] = np.nan                        
                        matrix[:aux.size, imethod] = aux
                        imethod += 1

                # Create pathOut
                if plotAllRegions == False:
                    pathOut = pathFigures
                else:
                    path = pathFigures + 'daily_spatialCorrBoxplot/' + targetVar.upper() + '/'
                    pathOut = path + subDir
                # pathOut = pathFigures
                if not os.path.exists(pathOut):
                    os.makedirs(pathOut)


                fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                medianprops = dict(color="black")
                g = ax.boxplot(matrix, vert=False, showfliers=False, patch_artist=True, medianprops=medianprops)
                # fill with colors
                i = 0
                color = [methods_colors[x['methodName']] for x in methods if x['var'] == targetVar]
                for patch in g['boxes']:
                    patch.set_facecolor(color[i])
                    i += 1

                units = ''
                # title = ' '.join((targetVar, metric, season))
                title = targetVar
                plt.xlim((0, 1))
                plt.title(title, fontsize=20)
                # plt.title(title)
                ax.set_yticklabels(names, rotation=0)
                # plt.ylines(y=0.5, xmin=-1, xmax=nmethods+1, linewidth=0)
                # plt.show()
                # exit()
                plt.savefig(pathOut + '_'.join(('EVALUATION'+bc_sufix, 'spatialCorrBoxplot', targetVar, 'all',
                                                season))+ '.png', bbox_inches='tight')
                plt.close()

    methods.reverse()

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
                            # obs = np.nanmean(np.load(pathIn + '_'.join((climdex_name, 'obs', season))+'.npy'), axis=0)
                            # est = np.nanmean(np.load(pathIn + '_'.join((climdex_name, 'est', season))+'.npy'), axis=0)
                            obs = np.nanmean(read.netCDF(pathIn, '_'.join((climdex_name, 'obs', season))+'.nc', climdex_name)['data'], axis=0)
                            est = np.nanmean(read.netCDF(pathIn, '_'.join((climdex_name, 'est', season))+'.nc', climdex_name)['data'], axis=0)

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

                            # Deal with nans
                            mask = ~np.isnan(matrix_region)
                            matrix_region = [d[m] for d, m in zip(matrix_region.T, mask.T)]

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
                                plt.ylim((-20, 100))
                            title = ' '.join((targetVar.upper(), climdex_name, 'bias', season))
                            # plt.title(title)
                            # if apply_bc == False:
                            #     title = climdex_name
                            # else:
                            #     title = climdex_name + '    bias corrected'
                            plt.title(title, fontsize=16)
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
def climdex_Taylor_diagrams(by_season):
    """
    Taylor diagram for the spatial distribution of all climdex, by seasons (optional) and by subregions (optional).
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
        df_reg = pd.read_csv(pathAux + 'ASSOCIATION/' + targetVar.upper() + '/regions.csv')

        # Go through all climdex
        for climdex_name in climdex_names[targetVar]:

            # Select season
            for season in season_dict:
                if season == annualName or by_season == True:

                    # Go through all regions
                    for index, row in df_reg.iterrows():
                        if plotAllRegions == True or (
                                (plotAllRegions == False) and (index == 0)):
                            regType, regName, subDir = row['regType'], row['regName'], row[
                                'subDir']
                            iaux = [int(x) for x in row['ipoints'][1:-1].split(', ')]
                            npoints = len(iaux)
                            print(regType, regName, npoints, 'points',
                                  str(index) + '/' + str(df_reg.shape[0]))

                            # Go through all methods
                            imethod = 0
                            names = []
                            colors = [methods_colors[x['methodName']] for x in methods if x['var'] == targetVar]
                            r_matrix = np.zeros((nmethods, 1))
                            std_matrix = np.zeros((nmethods, 1))
                            for method_dict in methods:
                                var = method_dict['var']
                                if var == targetVar:
                                    methodName = method_dict['methodName']
                                    names.append(methodName)
                                    print(var, climdex_name, season, methodName)

                                    pathIn = '../results/EVALUATION' + bc_sufix + '/' + targetVar.upper() + '/' + methodName + '/climdex/'
                                    # obs = np.nanmean(np.load(pathIn + '_'.join((climdex_name, 'obs', season))+'.npy'), axis=0)
                                    # est = np.nanmean(np.load(pathIn + '_'.join((climdex_name, 'est', season))+'.npy'), axis=0)
                                    obs = np.nanmean(read.netCDF(pathIn, '_'.join(
                                        (climdex_name, 'obs', season)) + '.nc', climdex_name)[
                                                         'data'], axis=0)
                                    est = np.nanmean(read.netCDF(pathIn, '_'.join(
                                        (climdex_name, 'est', season)) + '.nc', climdex_name)[
                                                         'data'], axis=0)

                                    obs_region = obs[iaux]
                                    est_region = est[iaux]
                                    r_matrix[imethod] = pearsonr(obs_region, est_region)[0]
                                    std_matrix[imethod] = np.std(est_region) / np.std(obs_region)
                                    imethod += 1

                            # Create pathOut
                            if plotAllRegions == False:
                                pathOut = pathFigures
                            else:
                                path = pathFigures + 'biasBoxplot/' + targetVar.upper() + '/'
                                pathOut = path + subDir
                            if not os.path.exists(pathOut):
                                os.makedirs(pathOut)

                            # Create Taylor diagram
                            fig = plt.figure(dpi=300)
                            dia = TaylorDiagram(1, fig=fig, rect=111, label="Reference", srange=(0, 1.7))
                            for imethod in range(nmethods):
                                dia.add_sample(std_matrix[imethod], r_matrix[imethod], color=colors[imethod],
                                               marker='*', markersize=10, label=names[imethod])
                            dia.add_grid()
                            contours = dia.add_contours(colors='0.5')
                            plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

                            # Add a figure legend
                            fig.legend(dia.samplePoints,
                                       [p.get_label() for p in dia.samplePoints],
                                       fontsize=10, bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
                            title = ' '.join(
                                (targetVar.upper(), climdex_name, season))
                            # plt.title(title)
                            # if apply_bc == False:
                            #     title = climdex_name
                            # else:
                            #     title = climdex_name + '    bias corrected'
                            # plt.title(title, fontsize=16)
                            filename = '_'.join(('EVALUATION' + bc_sufix,
                                                 'TaylorDiagram', targetVar,
                                                 climdex_name, regName,
                                                 season))
                            # plt.show()
                            # exit()
                            plt.savefig(pathOut + filename + '.png', bbox_inches='tight')
                            plt.close()


########################################################################################################################
def monthly_boxplots(metric):
    """
    Boxplots of metric,  by subregions (optional).
    :param metric: correlation or variance
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

                # Accumulate months

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
                    idates = [i for i in range(len(times_scene)) if
                              ((times_scene[i].year == year) and (times_scene[i].month == month))]
                    obs_acc[idate] = np.nansum(obs[idates], axis=0)
                    est_acc[idate] = np.nansum(est[idates], axis=0)

                not_valid_obs = np.where(np.isnan(obs_acc))[0]
                not_valid_est = np.where(np.isnan(est_acc))[0]
                ivalid = [i for i in range(len(dates)) if (i not in not_valid_obs) and (i not in not_valid_est)]
                obs_acc = obs_acc[ivalid]
                est_acc = est_acc[ivalid]
                matrix = np.zeros((hres_npoints[targetVar], ))
                if metric == 'correlation':
                    for ipoint in range(hres_npoints[targetVar]):
                        X = obs_acc[:, ipoint]
                        Y = est_acc[:, ipoint]
                        try:
                            r = round(pearsonr(X, Y)[0], 3)
                        except:
                            r = np.nan
                        if np.isnan(r) == True:
                            r = 0
                        matrix[ipoint] = r
                elif metric == 'R2':
                    matrix = 1 - np.nansum((est_acc - obs_acc) ** 2, axis=0) / np.nansum(obs_acc ** 2, axis=0)
                np.save('../tmp/'+targetVar+'_'+methodName+'_monthly_' +metric, matrix)
            imethod += 1


    # Correlations have been calculated and storaged. Now they will be plotted
    for targetVar in vars:
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
                print(metric, var, methodName)
                matrix[:, imethod] = np.load('../tmp/' + targetVar + '_' + methodName + '_monthly_'+metric+'.npy')
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

                fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                medianprops = dict(color="black")
                g = ax.boxplot(matrix_region, showfliers=False, patch_artist=True, medianprops=medianprops)
                # fill with colors
                i = 0
                color = [methods_colors[x['methodName']] for x in methods if x['var'] == targetVar]
                for patch in g['boxes']:
                    patch.set_facecolor(color[i])
                    i += 1

                units = ''
                title = ' '.join((targetVar, 'monthly', metric))
                # title = targetVar
                plt.ylim((0, 1))
                plt.title(title, fontsize=20)
                # plt.title(title)
                plt.ylabel(units, rotation=90)
                ax.set_xticklabels(names, rotation=90)
                plt.hlines(y=0.5, xmin=-1, xmax=nmethods+1, linewidth=0)
                # plt.show()
                # exit()
                plt.savefig(pathOut + '_'.join(('EVALUATION'+bc_sufix, metric+'BoxplotMonthly', targetVar, 'None', 'all'))+
                            '.png', bbox_inches='tight')
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

