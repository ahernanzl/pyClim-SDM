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
def hyperparameters(estimator, estimator_name, ipoint):
    '''
    Plot hyperparameters from gridSearchCV for some points.
    It is only prepared for regressor, not for classifiers.
    '''

    print('hyperparameters', estimator_name, ipoint)

    # Define hyperparameters depending on regressor type
    if estimator_name == 'LS-SVM':
        hyp1, hyp2 = 'alpha', 'gamma'
    elif estimator_name == 'SVR':
        hyp1, hyp2 = 'C', 'gamma'
    elif estimator_name == 'MLPR':
        exit('Plot hyperparameters not implemented yed ' + estimator_name)

    # Get hiyperparameters and scores from gridSearchCV
    params = estimator.cv_results_['params']
    x1 = np.unique(np.asarray([x[hyp1] for x in params]))
    x2 = np.unique(np.asarray([x[hyp2] for x in params]))
    scores = estimator.cv_results_['mean_test_score'].reshape(x1.size, x2.size)

    # Plot hyperparameters matrix
    plt.pcolor(scores.T, vmin=0)
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.xlabel(hyp1)
    plt.xticks(list(range(x1.size)), [str("{:.0e}".format(x)) for x in x1], rotation=70)
    plt.ylabel(hyp2)
    plt.yticks(list(range(x2.size)), [str("{:.0e}".format(x)) for x in x2])
    plt.colorbar(extend='min')
    plt.title(estimator_name)
    # plt.show()
    # exit()

    # Creat pathOut
    pathOut = pathAux + 'TRAINED_MODELS/PCP_regressors/Hyperparameters_' + estimator_name + '/'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    plt.savefig(pathOut + str(ipoint) + '.png')
    plt.close('all')


########################################################################################################################
def spatial_domains():
    '''
    Plot regions for different weighting for synoptic analogy
    '''

    lat_up, lat_down = pred_lat_up + grid_res * 5, pred_lat_down - grid_res * 6
    lon_left, lon_right = pred_lon_left - grid_res * 10, pred_lon_right + grid_res * 10
    grid_res = 1.5
    nlats = int(((lat_up - lat_down) / 1.5) + 1)
    nlons = int(((lon_right - lon_left) / 1.5 ) + 1)
    lats = np.linspace(lat_up, lat_down, nlats)
    lons = np.linspace(lon_left, lon_right, nlons)

    # Define mapç
    fig, ax = plt.subplots()
    map = Basemap(llcrnrlon=lons[0] - grid_res, llcrnrlat=lats[-1] - grid_res,
                  urcrnrlon=lons[-1] + grid_res, urcrnrlat=lats[0] + grid_res,
                  projection='merc', resolution='i')
    map.drawcoastlines()

    # Create grid
    mlons, mlats = np.meshgrid(lons, lats)
    X, Y = list(map(mlons, mlats))

    # Plot map
    map.scatter(X, Y, c='k', s=1, marker='.')

    # labels = [left,right,top,bottom]
    map.drawparallels(lats[1::2], linewidth=0.2, labels=[1, 0, 0, 0], fontsize=6)
    meridians = map.drawmeridians(lons[1::2], linewidth=0.2, labels=[0, 0, 0, 1], fontsize=6)

    # Rotate lon labels
    for m in meridians:
        try:
            meridians[m][1][0].set_rotation(90)
        except:
            pass

    # saf grid (for synoptic analogy)
    domains = ['00', 'NW', 'NE', 'SW', 'SE']
    colors = ['g', 'orange', 'purple', 'b', 'r']
    shifts = [.75, .25, .5, 1, 1.25]
    t = .5
    fw = 10
    fh = 3
    text = [(0, 0, t, -fh*t), (0, 0, t, -fh*t), (-1, 0, -fw*t, -fh*t), (0, -1, t, t), (-1, -1, -fw*t, t)]
    for idomain in range(len(domains)):
        domain = domains[idomain]
        color = colors[idomain]
        shift = shifts[idomain]
        if domain == '00':
            saf_lat_up, saf_lat_down = pred_lat_up + grid_res * 3, pred_lat_down - grid_res * 3
            saf_lon_left, saf_lon_right = pred_lon_left - grid_res * 5, pred_lon_right + grid_res * 5
        elif domain == 'NE':
            saf_lat_up, saf_lat_down = pred_lat_up + grid_res * 5, pred_lat_down - grid_res * 1
            saf_lon_left, saf_lon_right = pred_lon_left - grid_res * 9, pred_lon_right + grid_res * 1
        elif domain == 'NW':
            saf_lat_up, saf_lat_down = pred_lat_up + grid_res * 5, pred_lat_down - grid_res * 1
            saf_lon_left, saf_lon_right = pred_lon_left - grid_res * 1, pred_lon_right + grid_res * 9
        elif domain == 'SE':
            saf_lat_up, saf_lat_down = pred_lat_up + grid_res * 1, pred_lat_down - grid_res * 5
            saf_lon_left, saf_lon_right = pred_lon_left - grid_res * 9, pred_lon_right + grid_res * 1
        elif domain == 'SW':
            saf_lat_up, saf_lat_down = pred_lat_up + grid_res * 1, pred_lat_down - grid_res * 5
            saf_lon_left, saf_lon_right = pred_lon_left - grid_res * 1, pred_lon_right + grid_res * 9
        saf_grid_res = 1.5
        saf_nlats = int(((saf_lat_up - saf_lat_down) / 1.5) + 1)
        saf_nlons = int(((saf_lon_right - saf_lon_left) / 1.5 ) + 1)
        saf_lats = np.linspace(saf_lat_up, saf_lat_down, saf_nlats)
        saf_lons = np.linspace(saf_lon_left, saf_lon_right, saf_nlons)

        map.plot(np.array([saf_lons[0]-shift, saf_lons[-1]+shift, saf_lons[-1]+shift, saf_lons[0]-shift, saf_lons[0]-shift]),
                 np.array([saf_lats[-1]-shift, saf_lats[-1]-shift, saf_lats[0]+shift, saf_lats[0]+shift, saf_lats[-1]-shift]),
                 latlon=True, color=color)

        x, y = map(saf_lons[text[idomain][0]]+text[idomain][2], saf_lats[text[idomain][1]]+text[idomain][3])
        plt.text(x, y, domain, fontsize=12, color=color, fontweight='bold')



    lats = read.hres_metadata()['lats'].values
    lons = read.hres_metadata()['lons'].values
    X, Y = list(map(lons, lats))
    map.scatter(X, Y, c=0*lats, s=3, cmap='Accent_r')
    # plt.show()
    # exit()

    # Save image
    pathOut = '../results/Figures/'
    filename = 'spatial_domains.png'
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    plt.savefig(pathOut + filename)


########################################################################################################################
def weights_regions():
    '''
    Plot regions for different weighting for synoptic analogy
    '''

    w = W_saf[:saf_nlats * saf_nlons].reshape(saf_nlats, saf_nlons)

    # Define map
    map = Basemap(llcrnrlon=saf_lons[0] - saf_grid_res, llcrnrlat=saf_lats[-1] - saf_grid_res,
                  urcrnrlon=saf_lons[-1] + saf_grid_res, urcrnrlat=saf_lats[0] + saf_grid_res,
                  projection='merc', resolution='i')
    map.drawcoastlines()

    # Create grid
    mlons, mlats = np.meshgrid(saf_lons, saf_lats)
    X, Y = list(map(mlons, mlats))

    # Plot map
    s = 10

    # Use this if weights are different for each grid point
    map.scatter(X[w == np.unique(w)[0]], Y[w == np.unique(w)[0]], c='b', s=s, marker='x')
    map.scatter(X[w == np.unique(w)[1]], Y[w == np.unique(w)[1]], c='r', s=s, marker='^')
    map.scatter(X[w == np.unique(w)[2]], Y[w == np.unique(w)[2]], c='g', s=s, marker='.')

    # Use this if all grid points weight the same
    # map.scatter(X[w == 1], Y[w == 1], c='r', s=s, marker='.')

    # labels = [left,right,top,bottom]
    map.drawparallels(saf_lats[1::2], linewidth=0.2, labels=[1, 0, 0, 0], fontsize=6)
    meridians = map.drawmeridians(saf_lons[1::2], linewidth=0.2, labels=[0, 0, 0, 1], fontsize=6)

    # Rotate lon labels
    for m in meridians:
        try:
            meridians[m][1][0].set_rotation(90)
        except:
            pass
    # plt.show()
    # exit()

    # Save image
    pathOut = pathAux + 'WEATHER_TYPES/'
    filename = 'weights_regions.png'
    print('See ' + pathOut + filename + ' to decide about the weighting regions.')
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    plt.savefig(pathOut + filename)

########################################################################################################################
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


########################################################################################################################
def map(data, palette=None, lats=[None, None], lons=[None, None], path=None, filename=None, title=None, plot_library='Basemap',
        regType=None, regName=None, colorbar_out=False, pointSize=None, grid=None, plot_lat_lon=True):
    """
    :param data:
    :param palette: differente palettes are defined in the function
    :param lats: if none provided, hres_lats will be used
    :param lons: if none provided, hres_lons will be used
    :param path: pathOut
    :param filename: fileOut. If none provided, only show map, but does not save it.
    :param title: figure title
    :param plot_library: only Basemap tested. Cartopy also implemented but not tested.
    :param regType: to plot only a subregion
    :param regName: subregion name
    :param colorbar_out: if true, saves figure without colorbar and a horizontal colorbar as an independent image
    :param pointSize: different sizes needed depending on the region size and spatial resolution for optimal visualization
    :param grid: pred, saf or ext
    :param plot_lat_lon: plot x_ticks and y_ticks
    :return: void
    """

    if (filename != None) and (not os.path.exists(path)):
        os.makedirs(path)

    # Create dictionary of all possible palettes
    dict = {}
    dict.update({None: {'units': '', 'bounds': None, 'cmap': None, 'vmin': data.min(), 'vmax': data.max(), 'n_bin': 10,
                        'colors': ['g', 'y', 'r'], 'ext': 'neither'}})
    dict.update({'target_region': {'units': '', 'bounds': None, 'cmap': None, 'vmin': -1, 'vmax': 3, 'n_bin': 5,
                        'colors': ['g', 'y', 'r'], 'ext': 'neither'}})
    dict.update({'tmax_TXm': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': -20, 'vmax': 45, 'n_bin': 65,
                              'colors': ['m', 'c', 'b', 'g', 'y', 'r'], 'ext': 'both'}})
    dict.update({'tmax_p10': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': -20, 'vmax': 45, 'n_bin': 65,
                              'colors': ['m', 'c', 'b', 'g', 'y', 'r'], 'ext': 'both'}})
    dict.update({'tmax_p90': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': -20, 'vmax': 45, 'n_bin': 65,
                              'colors': ['m', 'c', 'b', 'g', 'y', 'r'], 'ext': 'both'}})
    dict.update({'tmin_TNm': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': -20, 'vmax': 45, 'n_bin': 65,
                              'colors': ['m', 'c', 'b', 'g', 'y', 'r'], 'ext': 'both'}})
    dict.update({'tmin_p10': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': -20, 'vmax': 45, 'n_bin': 65,
                              'colors': ['m', 'c', 'b', 'g', 'y', 'r'], 'ext': 'both'}})
    dict.update({'tmin_p90': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': -20, 'vmax': 45, 'n_bin': 65,
                              'colors': ['m', 'c', 'b', 'g', 'y', 'r'], 'ext': 'both'}})

    dict.update({'tmax_TXm_bias': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                   'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmax_p10_bias': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                   'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmax_p90_bias': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                   'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmin_TNm_bias': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                   'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmin_p10_bias': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                   'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmin_p90_bias': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                   'n_bin': None, 'colors': None, 'ext': 'both'}})

    dict.update({'tmax_TXm_change': {'units': degree_sign, 'bounds': np.array(range(-10, 12, 2)), 'cmap': 'bwr',
                                     'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmax_p10_change': {'units': degree_sign, 'bounds': np.array(range(-10, 12, 2)), 'cmap': 'bwr',
                                     'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmax_p90_change': {'units': degree_sign, 'bounds': np.array(range(-10, 12, 2)), 'cmap': 'bwr',
                                     'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmin_TNm_change': {'units': degree_sign, 'bounds': np.array(range(-10, 12, 2)), 'cmap': 'bwr',
                                     'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmin_p10_change': {'units': degree_sign, 'bounds': np.array(range(-10, 12, 2)), 'cmap': 'bwr',
                                     'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmin_p90_change': {'units': degree_sign, 'bounds': np.array(range(-10, 12, 2)), 'cmap': 'bwr',
                                     'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'both'}})

    dict.update({'tmax_TXm_biasMedian': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                         'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmax_p10_biasMedian': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                         'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmax_p90_biasMedian': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                         'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmin_TNm_biasMedian': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                         'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmin_p10_biasMedian': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                         'n_bin': None, 'colors': None, 'ext': 'both'}})
    dict.update({'tmin_p90_biasMedian': {'units': degree_sign, 'bounds': np.array(
        [-2.5, -1.5, -1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1, 1.5, 2.5]), 'cmap': 'bwr', 'vmin': None, 'vmax': None,
                                         'n_bin': None, 'colors': None, 'ext': 'both'}})

    dict.update({'tmax_TXm_biasSpread': {'units': degree_sign, 'bounds': np.array([0, .2, .4, .6, .8, 1, 1.5, 2.5]),
                                         'cmap': 'Purples', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None,
                                         'ext': 'max'}})
    dict.update({'tmax_p10_biasSpread': {'units': degree_sign, 'bounds': np.array([0, .2, .4, .6, .8, 1, 1.5, 2.5]),
                                         'cmap': 'Purples', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None,
                                         'ext': 'max'}})
    dict.update({'tmax_p90_biasSpread': {'units': degree_sign, 'bounds': np.array([0, .2, .4, .6, .8, 1, 1.5, 2.5]),
                                         'cmap': 'Purples', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None,
                                         'ext': 'max'}})
    dict.update({'tmin_TNm_biasSpread': {'units': degree_sign, 'bounds': np.array([0, .2, .4, .6, .8, 1, 1.5, 2.5]),
                                         'cmap': 'Purples', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None,
                                         'ext': 'max'}})
    dict.update({'tmin_p10_biasSpread': {'units': degree_sign, 'bounds': np.array([0, .2, .4, .6, .8, 1, 1.5, 2.5]),
                                         'cmap': 'Purples', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None,
                                         'ext': 'max'}})
    dict.update({'tmin_p90_biasSpread': {'units': degree_sign, 'bounds': np.array([0, .2, .4, .6, .8, 1, 1.5, 2.5]),
                                         'cmap': 'Purples', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None,
                                         'ext': 'max'}})

    dict.update({'pcp_PRCPTOT': {'units': 'mm', 'bounds': np.linspace(0, 600, 13), 'cmap': 'terrain_r', 'vmin': None,
                                 'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_PRCPTOT_rel_bias': {'units': '%',
                                          'bounds': np.array([-100, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100]),
                                          'cmap': 'BrBG', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None,
                                          'ext': 'max'}})
    dict.update({'pcp_PRCPTOT_rel_change': {'units': '%', 'bounds': np.array(
        [-100, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100]), 'cmap': 'BrBG', 'vmin': None, 'vmax': None,
                                            'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_PRCPTOT_rel_biasMedian': {'units': '%', 'bounds': np.array(
        [-100, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100]), 'cmap': 'BrBG', 'vmin': None, 'vmax': None,
                                                'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_PRCPTOT_rel_biasSpread': {'units': '%', 'bounds': np.array([0, 10, 20, 30, 40, 50, 100]),
                                                'cmap': 'Purples', 'vmin': None, 'vmax': None, 'n_bin': None,
                                                'colors': None, 'ext': 'max'}})

    dict.update({'pcp_R95p': {'units': 'mm', 'bounds': np.linspace(0, 200, 11), 'cmap': 'terrain_r', 'vmin': None,
                              'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_R95p_rel_bias': {'units': '%', 'bounds': np.array(
        [-100, -80, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100, 150]), 'cmap': 'BrBG', 'vmin': None, 'vmax': None,
                                       'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_R95p_rel_change': {'units': '%', 'bounds': np.array(
        [-100, -80, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100, 150]), 'cmap': 'BrBG', 'vmin': None, 'vmax': None,
                                         'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_R95p_rel_biasMedian': {'units': '%', 'bounds': np.array(
        [-100, -80, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100, 150]), 'cmap': 'BrBG', 'vmin': None, 'vmax': None,
                                             'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_R95p_rel_biasSpread': {'units': '%', 'bounds': np.array([0, 10, 20, 30, 40, 50, 100]),
                                             'cmap': 'Purples', 'vmin': None, 'vmax': None, 'n_bin': None,
                                             'colors': None, 'ext': 'max'}})

    dict.update({'pcp_R01': {'units': 'days', 'bounds': np.linspace(0, 55, 12), 'cmap': 'terrain_r', 'vmin': None,
                             'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_R01_rel_bias': {'units': '%',
                                      'bounds': np.array([-100, -80, -50, -30, -20, -10, 10, 20, 30, 50, 100, 200]),
                                      'cmap': 'BrBG', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None,
                                      'ext': 'max'}})
    dict.update({'pcp_R01_rel_change': {'units': '%',
                                        'bounds': np.array([-100, -80, -50, -30, -20, -10, 10, 20, 30, 50, 100, 200]),
                                        'cmap': 'BrBG', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None,
                                        'ext': 'max'}})
    dict.update({'pcp_R01_rel_biasMedian': {'units': '%', 'bounds': np.array(
        [-100, -80, -50, -30, -20, -10, 10, 20, 30, 50, 100, 200]), 'cmap': 'BrBG', 'vmin': None, 'vmax': None,
                                            'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_R01_rel_biasSpread': {'units': '%', 'bounds': np.array([0, 10, 20, 30, 50, 100]),
                                            'cmap': 'Purples', 'vmin': None, 'vmax': None, 'n_bin': None,
                                            'colors': None, 'ext': 'max'}})


    dict.update({'pcp_SDII': {'units': 'mm', 'bounds': np.linspace(0, 20, 9), 'cmap': 'terrain_r', 'vmin': None,
                                 'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_SDII_rel_bias': {'units': '%',
                                          'bounds': np.array([-100, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100]),
                                          'cmap': 'BrBG', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None,
                                          'ext': 'max'}})
    dict.update({'pcp_SDII_rel_change': {'units': '%', 'bounds': np.array(
        [-100, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100]), 'cmap': 'BrBG', 'vmin': None, 'vmax': None,
                                            'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_SDII_rel_biasMedian': {'units': '%', 'bounds': np.array(
        [-100, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100]), 'cmap': 'BrBG', 'vmin': None, 'vmax': None,
                                                'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'pcp_SDII_rel_biasSpread': {'units': '%', 'bounds': np.array([0, 10, 20, 30, 40, 50, 100]),
                                                'cmap': 'Purples', 'vmin': None, 'vmax': None, 'n_bin': None,
                                                'colors': None, 'ext': 'max'}})

    ############ Theses palettes are to be tuned
    dict.update({'p': {'units': 'mm', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 50, 'n_bin': 50,
                       'colors': ['g', 'y', 'r', 'm', 'b', 'c'], 'ext': 'max'}})
    # dict.update({'%': {'units': '%', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 100, 'n_bin': 50, 'colors': ['g', 'y', 'r', 'm', 'b', 'c'], 'ext': 'neither'}})
    # dict.update({'%_bias': {'units': '%', 'bounds': None, 'cmap': None, 'vmin': -20, 'vmax': 20, 'n_bin': 50, 'colors': ['b', 'w', 'r'], 'ext': 'both'}})
    # dict.update({'t_bias': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': -1, 'vmax': 1, 'n_bin': 21, 'colors': ['b', 'w', 'r'], 'ext': 'both'}})
    # dict.update({'p_bias': {'units': 'mm', 'bounds': None, 'cmap': None, 'vmin': -10, 'vmax': 10, 'n_bin': 20, 'colors': ['b', 'w', 'r'], 'ext': 'both'}})
    dict.update({'t_mae': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 3, 'n_bin': 6,
                           'colors': ['g', 'y', 'r'], 'ext': 'max'}})
    dict.update({'p_mae': {'units': 'mm', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 6, 'n_bin': 6,
                           'colors': ['g', 'y', 'r'], 'ext': 'max'}})
    dict.update({'t_rmse': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 3, 'n_bin': 6,
                            'colors': ['g', 'y', 'r'], 'ext': 'max'}})
    dict.update({'p_rmse': {'units': 'mm', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 10, 'n_bin': 10,
                            'colors': ['g', 'y', 'r'], 'ext': 'max'}})
    dict.update({'prob': {'units': '', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 1, 'n_bin': 5,
                          'colors': ['r', 'y', 'g'], 'ext': 'neither'}})
    dict.update({'corrMonth': {'units': '', 'bounds': None, 'cmap': None, 'vmin': 0.4, 'vmax': 1, 'n_bin': 6,
                               'colors': ['r', 'y', 'g'], 'ext': 'min'}})
    dict.update({'correlation': {'units': '', 'bounds': None, 'cmap': None, 'vmin': 0.5, 'vmax': 1, 'n_bin': 10,
                               'colors': ['r', 'y', 'g'], 'ext': 'min'}})
    dict.update({'pearson': {'units': '', 'bounds': None, 'cmap': None, 'vmin': 0.5, 'vmax': 1, 'n_bin': 10,
                               'colors': ['r', 'y', 'g'], 'ext': 'min'}})
    dict.update({'spearman': {'units': '', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 1, 'n_bin': 10,
                               'colors': ['r', 'y', 'g'], 'ext': 'neither'}})
    dict.update({'probInv': {'units': '', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 1, 'n_bin': 5, 'colors': ['g', 'y', 'r'], 'ext': 'neither'}})
    dict.update({'r2': {'units': '', 'bounds': None, 'cmap': None, 'vmin': .5, 'vmax': 1, 'n_bin': 5,
                        'colors': ['r', 'y', 'g'], 'ext': 'min'}})
    dict.update({'t_r2': {'units': '', 'bounds': None, 'cmap': None, 'vmin': 0.8, 'vmax': 1, 'n_bin': 4,
                        'colors': ['r', 'y', 'g'], 'ext': 'min'}})
    dict.update({'p_r2': {'units': '', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 1, 'n_bin': 5,
                        'colors': ['r', 'y', 'g'], 'ext': 'min'}})
    dict.update({'acc': {'units': '', 'bounds': None, 'cmap': None, 'vmin': 0.5, 'vmax': 1, 'n_bin': 10, 'colors': ['r', 'y', 'g'], 'ext': 'min'}})
    # dict.update({'std_interp_preds': {'units': '', '': None, 'cmap': None, 'vmin': -2, 'vmax': 2, 'n_bin': 41, 'colors': ['m', 'c', 'b', 'g', 'y', 'r'], 'ext': 'both'}})
    # dict.update({'abs_bias': {'units': '', 'bounds': None, 'cmap': None, 'vmin': -2, 'vmax': 2, 'n_bin': 25, 'colors': ['b', 'w', 'r'], 'ext': 'both'}})
    # dict.update({'rel_bias': {'units': '', 'bounds': None, 'cmap': None, 'vmin': -5, 'vmax': 5, 'n_bin': 25, 'colors': ['b', 'w', 'r'], 'ext': 'both'}})
    # dict.update({'pcp_p95': {'units': 'mm', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 40, 'n_bin': 20, 'colors': ['g', 'y', 'r', 'm', 'b', 'c'], 'ext': 'max'}})
    # dict.update({'pcp_p95': {'units': 'mm', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 30, 'n_bin': 7, 'colors': ['r', 'orange', 'y', 'g', 'c', 'b', 'purple', 'pink'], 'ext': 'max'}})
    # dict.update({'pcp_p95': {'units': 'mm', 'bounds': np.linspace(0, 30, 7), 'cmap': 'terrain_r', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'max'}})
    # dict.update({'pcp_p95_bias': {'units': 'mm', 'bounds': None, 'cmap': None, 'vmin': -10, 'vmax': 10, 'n_bin': 20, 'colors': ['brown', 'w', 'g'], 'ext': 'both'}})
    # dict.update({'pcp_p95_rel_bias': {'units': '%', 'bounds': None, 'cmap': None, 'vmin': -50, 'vmax': 50, 'n_bin': 20, 'colors': ['brown', 'w', 'g'], 'ext': 'both'}})
    # dict.update({'pcp_p95_rel_bias': {'units': '%', 'bounds': np.array([-150, -100, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100, 150]), 'cmap': 'BrBG', 'vmin': None, 'vmax': None, 'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'change_TXm_mean': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 12,
                                      'n_bin': 12, 'colors': ['g', 'y', 'r'], 'ext': 'both'}})
    dict.update({'change_TNm_mean': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 12,
                                      'n_bin': 12, 'colors': ['g', 'y', 'r'], 'ext': 'both'}})
    # dict.update({'change_Pm_mean': {'units': '%', 'bounds': None, 'cmap': None, 'vmin': -100, 'vmax': 100, 'n_bin': 11,
    #                                  'colors': ['r', 'y', 'g'], 'ext': 'both'}})
    dict.update({'change_PRCPTOT_mean': {'units': '%', 'bounds': np.array(
        [-100, -50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 100]), 'cmap': 'BrBG', 'vmin': None, 'vmax': None,
                                            'n_bin': None, 'colors': None, 'ext': 'max'}})
    dict.update({'change_Pm_mean': {'units': '%', 'bounds': np.array([-150, -100, -50, -40, -30, -20, -10, 0, 10, 20,
                                      30, 40, 50, 100, 150]), 'cmap': None, 'vmin': None, 'vmax': None, 'n_bin': None,
                                     'colors': ['r', 'y', 'g'], 'ext': 'both'}})
    dict.update({'change_TXm_std': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 4,
                                     'n_bin': 8, 'colors': ['g', 'y', 'r'], 'ext': 'max'}})
    dict.update({'change_TNm_std': {'units': degree_sign, 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 4,
                                     'n_bin': 8, 'colors': ['g', 'y', 'r'], 'ext': 'max'}})
    dict.update({'change_Pm_std': {'units': '%', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 100, 'n_bin': 10,
                                    'colors': ['g', 'y', 'r'], 'ext': 'max'}})
    dict.update({'change_PRCPTOT_std': {'units': '%', 'bounds': None, 'cmap': None, 'vmin': 0, 'vmax': 100, 'n_bin': 10,
                                    'colors': ['g', 'y', 'r'], 'ext': 'max'}})
    
    # dict.update({'none': {'units': degree_sign, 'bounds': np.array([0, 1]), 'cmap': 'bwr', 'ext': 'both'}})
    #
    # Select palette
    units = dict[palette]['units']
    bounds = dict[palette]['bounds']
    cmap = dict[palette]['cmap']
    ext = dict[palette]['ext']
    c = dict[palette]['colors']
    vmin = dict[palette]['vmin']
    vmax = dict[palette]['vmax']
    n_bin = dict[palette]['n_bin']

    if bounds is None:
        irregular_bins = False
    else:
        irregular_bins = True

    if grid == None:
        # Read lats lons
        if lats[0] == None:
            lats = read.hres_metadata()['lats'].values
            lons = read.hres_metadata()['lons'].values

        if regType != None:
            regNames = pd.read_csv(pathAux + 'ASSOCIATION/association.csv')[regType].values
            iaux = [i for i in range(len(regNames)) if regNames[i] == regName]
            lats, lons = lats[iaux], lons[iaux]

        # Set map limits
        latmin = np.min(lats) - 2
        latMax = np.max(lats) + 2
        lonmin = np.min(lons) - 1
        lonMax = np.max(lons) + 1

        if palette == 'target_region':
            # Set map limits
            latmin = np.min(lats) - 5
            latMax = np.max(lats) + 15
            lonmin = np.min(lons) - 10
            lonMax = np.max(lons) + 20

    elif grid != None:
        # Read lats lons
        if grid == 'ext':
            lats, lons, grid_res = ext_lats, ext_lons, ext_grid_res
        elif grid == 'saf':
            lats, lons, grid_res = saf_lats, saf_lons, saf_grid_res
        elif grid == 'pred':
            lats, lons, grid_res = pred_lats, pred_lons, grid_res

        # Give lats/lons proper format for colormesh function
        lats = np.sort(lats)
        lons = np.sort(lons)
        lats = np.append(lats, lats[-1] + grid_res) - grid_res / 2
        lons = np.append(lons, lons[-1] + grid_res) - grid_res / 2

        # Set map limits
        latmin = np.min(lats)
        latMax = np.max(lats)
        lonmin = np.min(lons)
        lonMax = np.max(lons)


    # Create colormap
    if irregular_bins == True:
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        cmap = plt.get_cmap(cmap)
    else:
        cmap = LinearSegmentedColormap.from_list('my_list', c, N=n_bin)
    if palette in ('pcp_PRCPTOT', 'pcp_SDII', 'pcp_p95', 'pcp_R95p', 'pcp_R01'):
        cmap = truncate_colormap(cmap, .3)

    # Define map
    # fig = plt.figure(dpi=300)
    if plot_library == 'Basemap':

        map = Basemap(llcrnrlon=lonmin, llcrnrlat=latmin, urcrnrlon=lonMax, urcrnrlat=latMax, projection='merc',
                      resolution='l')
        map.drawcoastlines(linewidth=0.5)
        if plot_lat_lon == True:
            # labels = [left,right,top,bottom]
            map.drawparallels(list(range(-90, 90, 4)), linewidth=0, labels=[1, 0, 0, 0], fontsize=6)
            meridians = map.drawmeridians(list(range(-180, 180, 4)), linewidth=0, labels=[0, 0, 0, 1], fontsize=6)
            for m in meridians:
                try:
                    meridians[m][1][0].set_rotation(45)
                except:
                    pass

        if grid == None:
            if pointSize != None:
                s = pointSize
            elif pseudoreality == False:
                # s = 1
                s = 10
            elif pseudoreality == True:
                s = 10
            if palette == 'target_region':
                s = .05
            X, Y = list(map(lons, lats))
            if irregular_bins == True:
                map.scatter(X, Y, c=data, s=s, norm=norm, cmap=cmap)
            else:
                map.scatter(X, Y, c=data, s=s, cmap=cmap, vmin=vmin, vmax=vmax)
        elif grid != None:
            x = np.linspace(0, map.urcrnrx, lons.shape[0])
            y = np.linspace(0, map.urcrnry, lats.shape[0])
            X, Y = np.meshgrid(x, y)
            if irregular_bins == True:
                map.pcolormesh(X, Y, data, norm=norm, cmap=cmap)
            else:
                map.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax)


    # elif plot_library == 'Cartopy':
    #     fig = plt.figure(figsize=(8, 6), dpi=300)
    #     projection = ccrs.PlateCarree()
    #     resolution = '50m'
    #     ax = plt.axes(projection=projection)
    #     ax.coastlines(resolution=resolution)
    #
    #     # ax.add_feature(cfeature.BORDERS, resolution=resolution, edgecolor="k")
    #     # country_borders = cfeature.NaturalEarthFeature(
    #     #     category='cultural',
    #     #     name='‘admin_0_boundary_lines_land',
    #     #     scale='110m',
    #     #     facecolor='none')
    #     # ax.add_feature(country_borders, edgecolor='gray')
    #     # ax.add_feature(Cartopy.feature.OCEAN, zorder=100, color='w')
    #
    #     if pointSize != None:
    #         s = pointSize
    #     elif pseudoreality == False:
    #         s = 1
    #     elif pseudoreality == True:
    #         s = 10
    #     if palette == 'target_region':
    #         s = .05
    #
    #     if irregular_bins == True:
    #         plt.scatter(lons, lats, c=data, s=s, norm=norm, cmap=cmap, transform=projection)
    #     else:
    #         plt.scatter(lons, lats, c=data, s=s, cmap=cmap, vmin=vmin, vmax=vmax, transform=projection)

    else:
        print('plot_library', plot_library, 'not implemented')


    # Plot map
    if colorbar_out == False:
        # fraction, pad = 0.027, 0.04
        if irregular_bins == True:
            # cbar = plt.colorbar(extend=ext, ticks=bounds, fraction=fraction, pad=pad)
            cbar = plt.colorbar(extend=ext, ticks=bounds)
        else:
            # cbar = plt.colorbar(extend=ext, fraction=fraction, pad=pad)
            cbar = plt.colorbar(extend=ext)
        cbar.set_label(label=units, rotation=0)

    # Load region polygon
    if regType == 'PROV':
        map.readshapefile(
            '../input_data/shapefiles/PROV_&_CCAA/recintos_provinciales_inspire_peninbal_etrs89/recintos_provinciales_inspire_peninbal_etrs89',
            'comarques', drawbounds=False)
        regName_col = 'NAMEUNIT'
    elif regType == 'CCAA':
        map.readshapefile(
            '../input_data/shapefiles/PROV_&_CCAA/recintos_autonomicas_inspire_peninbal_etrs89/recintos_autonomicas_inspire_peninbal_etrs89',
            'comarques', drawbounds=False)
        regName_col = 'NAMEUNIT'
    elif regType == 'CCHH':
        map.readshapefile('../input_data/shapefiles/CUENCAS/DemarcacionesHidrograficasPHC2015_2021', 'comarques',
                          drawbounds=False)
        regName_col = 'nameText'
    if regType not in (None, typeCompleteRegion):
        for info, shape in zip(map.comarques_info, map.comarques):
            if info[regName_col] == regName:
                x, y = zip(*shape)
                map.plot(x, y, marker=None, color='m')

    # plt.show()
    # exit()

    if title != None:
        plt.title(title)
    if filename == None:
        plt.show()
        # exit()
    else:
        filename = filename.replace('/', '_')
        plt.savefig(path + '/' + filename + '.png')
        plt.close('all')

    if colorbar_out == True:
        filecbar = path + '/' + filename + '_colorbar.png'
        fig = plt.figure()
        ax = fig.add_axes([0.05, 0.80, 0.9, .02])

        if irregular_bins == True:
            cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap, norm=norm, ticks=bounds, extend=ext)
        else:
            cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap,
                                           norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), extend=ext)

        cb.set_label(label=units, rotation=0)
        plt.savefig(filecbar)
        plt.close('all')
        w, h, leftSpace, bottonSpace = str(1000), str(70), str(5), str(75)
        dim = w + 'x' + h + '+' + leftSpace + '+' + bottonSpace
        os.system('convert ' + filecbar + ' -crop ' + dim + ' ' + filecbar)

