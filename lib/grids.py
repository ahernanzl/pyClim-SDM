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
def interpolate_predictors(pred, i_4nn, j_4nn, w_4nn, interp):
    '''
    :param pred: (ndays, npreds, nlats, nlons)
    :param interpolation_method:
        - nearest: using coords UTM
        - bilinear: using coords lat/lon. Vertices are sorted clockwise from the top-left corner.
        - inverse_distances: using coords UTM. The 4 nearest neighbours do not always form a square.
    :return: predOut(ndays, npreds)
    '''

    warnings.filterwarnings("ignore")

    # Define npreds, ndays, predOut
    npreds, ndays = pred.shape[1], pred.shape[0]
    predOut = np.zeros((ndays, npreds))

    # nearest uses coords UTM.
    if interp == 'nearest':
        predOut = pred[:, :, i_4nn[0], j_4nn[0]]

    # inverse_distances uses coords UTM and bilinear uses lat/lon.
    elif interp in ('inverse_distances', 'bilinear'):
        predOut = np.sum(w_4nn[np.newaxis, np.newaxis, :] * pred[:, :, i_4nn, j_4nn], axis=2)

    return predOut



########################################################################################################################
def association(interp, targetVar):
    """
    Associates each high resolution grid point to the low resolution grid in three different ways:
    - nearest: using coords UTM
    - inverse_distances: using coords UTM. The 4 nearest neighbours do not always form a square.
    - bilinear: using coords lat/lon. Vertices are sorted clockwise from the top-left corner.

    Inputs: lat/lon of both grids
    Outpus: csv file containing list of hight resolution points with coordinates of nearest low resolution point and
    distance to it
    """

    print('grids.association() starts')

    # Read lat lon of high resolution grid
    df_association = read.hres_metadata(targetVar)
    ids = df_association.index.values

    for n in range(4):
        df_association['lat' + str(n+1)] = np.nan
        df_association['lon' + str(n+1)] = np.nan
        df_association['dist' + str(n+1)] = np.nan
        df_association['i' + str(n+1)] = np.nan
        df_association['j' + str(n+1)] = np.nan
        df_association['w' + str(n+1)] = np.nan

    # Goes through each high resolution point and looks for its 4 nearest low resolution neighbours
    for ipoint in range(hres_npoints[targetVar]):
        if ipoint % 100 == 0:
            print(targetVar, 'grids.association', interp, 100*ipoint/hres_npoints[targetVar], '%')
        hres_lat = df_association.iloc[[ipoint]]['lats'].values[0]
        hres_lon = df_association.iloc[[ipoint]]['lons'].values[0]

        # Nearest_neighbour or inverse distances
        if interp in ('nearest', 'inverse_distances'):

            # Get distances
            dist_array = np.zeros([pred_nlats, pred_nlons])
            for i in range(0, pred_nlats):
                for j in range(0, pred_nlons):
                    dist_array[i][j] = dist((hres_lat,hres_lon), (pred_lats[i], pred_lons[j])).km

            # Sort by distance
            index = np.argsort(dist_array.reshape(pred_nlats*pred_nlons))

            # Fill the dataframe
            d = np.zeros((4,))
            for n in range(4):
                i = int(index[n] / pred_nlons)
                j = index[n] % pred_nlons
                df_association['lat' + str(n+1)].iloc[ipoint] = pred_lats[i]
                df_association['lon' + str(n+1)].iloc[ipoint] = pred_lons[j]
                df_association['dist' + str(n+1)].iloc[ipoint] = dist_array[i,j]
                df_association['i' + str(n+1)].iloc[ipoint] = i
                df_association['j' + str(n+1)].iloc[ipoint] = j
                d[n] = dist_array[i, j]


            # Get weights
            warnings.filterwarnings("ignore")
            w = 1/d
            w /= w.sum()
            w[np.isnan(w)] = 1
            for n in range(4):
                df_association['w' + str(n+1)].iloc[ipoint] = w[n]


        # Bilinear
        if interp == 'bilinear':

            # Get lat lon vertices
            latDown = max(pred_lats[pred_lats <= hres_lat])
            latUp = latDown + grid_res
            lonLeft = max(pred_lons[pred_lons <= hres_lon])
            lonRight = lonLeft + grid_res

            # Get i j vertices (lats are ordered top-bottom, and lons left-right)
            ilatUp = np.where(pred_lats == latUp)[0]
            ilatDown = ilatUp + 1
            jlonLeft = np.where(pred_lons == lonLeft)[0]
            jlonRight = jlonLeft + 1

            # Get weights for bilinear interpolation (vertices ordered clockwise from top-left)
            w1 = (hres_lat - latDown) * (lonRight - hres_lon) / grid_res**2
            w2 = (hres_lat - latDown) * (hres_lon - lonLeft) / grid_res**2
            w3 = (latUp - hres_lat) * (hres_lon - lonLeft) / grid_res**2
            w4 = (latUp - hres_lat) * (lonRight - hres_lon) / grid_res**2

            # Fill dataframe
            df_association['w1'].iloc[ipoint] = w1
            df_association['w2'].iloc[ipoint] = w2
            df_association['w3'].iloc[ipoint] = w3
            df_association['w4'].iloc[ipoint] = w4
            df_association['lat1'].iloc[ipoint] = latUp
            df_association['lat2'].iloc[ipoint] = latUp
            df_association['lat3'].iloc[ipoint] = latDown
            df_association['lat4'].iloc[ipoint] = latDown
            df_association['lon1'].iloc[ipoint] = lonLeft
            df_association['lon2'].iloc[ipoint] = lonRight
            df_association['lon3'].iloc[ipoint] = lonRight
            df_association['lon4'].iloc[ipoint] = lonLeft
            df_association['i1'].iloc[ipoint] = ilatUp
            df_association['i2'].iloc[ipoint] = ilatUp
            df_association['i3'].iloc[ipoint] = ilatDown
            df_association['i4'].iloc[ipoint] = ilatDown
            df_association['j1'].iloc[ipoint] = jlonLeft
            df_association['j2'].iloc[ipoint] = jlonRight
            df_association['j3'].iloc[ipoint] = jlonRight
            df_association['j4'].iloc[ipoint] = jlonLeft

    # Save results
    pathOut=pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp+'/'

    try:
        os.makedirs(pathOut)
    except:
        pass
    df_association.to_csv(pathOut+'association.csv')

    # Create arrays for speed
    df_association = pd.read_csv(pathOut+'/association.csv')
    i_4nn = np.asarray(df_association[["i1", "i2", "i3", "i4"]], dtype='int')
    j_4nn = np.asarray(df_association[["j1", "j2", "j3", "j4"]], dtype='int')
    w_4nn = np.asarray(df_association[["w1", "w2", "w3", "w4"]])
    lats_4nn = np.asarray(df_association[["lat1", "lat2", "lat3", "lat4"]])
    lons_4nn = np.asarray(df_association[["lon1", "lon2", "lon3", "lon4"]])
    hres_lats = df_association[["lats"]].values
    hres_lons = df_association[["lons"]].values

    # Save data
    np.save(pathOut+'i_4nn', i_4nn)
    np.save(pathOut+'j_4nn', j_4nn)
    np.save(pathOut+'w_4nn', w_4nn)
    np.save(pathOut+'lats_4nn', lats_4nn)
    np.save(pathOut+'lons_4nn', lons_4nn)
    np.save(pathOut+'hres_lats', hres_lats)
    np.save(pathOut+'hres_lons', hres_lons)


########################################################################################################################
def subregions(targetVar):
    """
    For each region_type adds column to association.csv and each cell is filled with the region_name of the point.
    For each region_name adds row to regions.csv with a cell listing all points contained in the region.

    If no division by regions is desired (divideByRegions=False at settings), only 1 col/row will be added.

    If a divission by regions is desired, several things must be adapted inside this function. At the moment only
        EspañaPB is implemented (Balearic Islands are treated manually).

    For a new project shapefiles and regTypes must be defined.
    """

    print('calculating subregions', targetVar)

    pathOutReg = pathAux+'ASSOCIATION/'+targetVar.upper()+'/'
    if not os.path.exists(pathOutReg):
        os.makedirs(pathOutReg)

    # Read hres metadata
    df_association = pd.read_csv(pathAux+'ASSOCIATION/'+targetVar.upper()+'_bilinear/association.csv')

    # Define name of complete region for csv
    if nameCompleteRegion == 'myRegionName':
        df_association[typeCompleteRegion] = 'myRegionName'
    elif nameCompleteRegion == 'EspañaPB':
        df_association[typeCompleteRegion] = 'ESPAÑA PENINSULAR'
    elif nameCompleteRegion == 'Canarias':
        df_association[typeCompleteRegion] = 'CANARIAS'
    else:
        print('nameCompleteRegion ' + nameCompleteRegion + ' not implemented.')
        print('grids.subregions() needs to be tuned for this new region/project.')
        print('shapefiles with polygons or any other way to define regions for each hres point are needed.')
        exit()
    # for interp in ('nearest', 'bilinear'):
    #     df_association.to_csv(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp+'/association.csv')

    # If a division by regions will be used
    if divideByRegions == True:
        if nameCompleteRegion in ['EspañaPB', 'Canarias']:

            # Read shapefiles
            if nameCompleteRegion == 'EspañaPB':
                prov_shp = gpd.read_file('../input_data/shapefiles/PROV_&_CCAA/recintos_provinciales_inspire_peninbal_etrs89/recintos_provinciales_inspire_peninbal_etrs89.shp')
                ccaa_shp = gpd.read_file('../input_data/shapefiles/PROV_&_CCAA/recintos_autonomicas_inspire_peninbal_etrs89/recintos_autonomicas_inspire_peninbal_etrs89.shp')
            elif nameCompleteRegion == 'Canarias':
                prov_shp = gpd.read_file('../input_data/shapefiles/PROV_&_CCAA/recintos_provinciales_inspire_canarias_wgs84/recintos_provinciales_inspire_canarias_wgs84.shp')
                ccaa_shp = gpd.read_file('../input_data/shapefiles/PROV_&_CCAA/recintos_autonomicas_inspire_canarias_wgs84/recintos_autonomicas_inspire_canarias_wgs84.shp')
            cuencas_shp = gpd.read_file('../input_data/shapefiles/CUENCAS/DemarcacionesHidrograficasPHC2015_2021.shp')

            # Define field names for each divission
            prov_dict = {'col_name': 'PROV', 'shp': prov_shp, 'geometry': 'geometry', 'regName': 'NAMEUNIT'}
            ccaa_dict = {'col_name': 'CCAA', 'shp': ccaa_shp, 'geometry': 'geometry', 'regName': 'NAMEUNIT'}
            cuencas_dict = {'col_name': 'CCHH', 'shp': cuencas_shp, 'geometry': 'geometry', 'regName': 'nameText'}

            # Initialized dataFrame
            df_association['PROV'] = np.nan
            df_association['CCAA'] = np.nan
            df_association['CCHH'] = np.nan

            # Go through all divissions and points to classify each point
            for divission in (prov_dict, ccaa_dict, cuencas_dict):
                for ipoint in range(hres_npoints[targetVar]):
                    if ipoint%100==0:
                        print(divission['col_name'], int(100*ipoint/hres_npoints[targetVar]),'%')
                    lat, lon = df_association.iloc[ipoint]['lats'], df_association.iloc[ipoint]['lons']
                    for iregion in range(divission['shp'].shape[0]):
                        regName = divission['shp'].loc[iregion, divission['regName']]
                        reg_poly = divission['shp'].loc[iregion, divission['geometry']]
                        point = Point(lon, lat)
                        if point.within(reg_poly) == True:
                            if (divission['col_name'], regName) == ('PROV', 'Illes Balears'):
                                if lon < 2:
                                    regName = 'Illes Balears (Ibiza-Formentera)'
                                elif lon > 3.6:
                                    regName = 'Illes Balears (Menorca)'
                                else:
                                    regName = 'Illes Balears (Mallorca)'
                            df_association[divission['col_name']][ipoint] = regName

            # # Save results
            # for interp in ('nearest', 'bilinear'):
            #     df_association.to_csv(pathAux+'ASSOCIATION/'+targetVar.upper()+'_'+interp+'/association.csv')

            # Create regions file
            df_reg = pd.DataFrame(columns=['regType', 'regName', 'subDir', 'ipoints'])
            for regType in (typeCompleteRegion, 'PROV', 'CCAA', 'CCHH'):
                regName_list = list(df_association[regType].unique())
                regName_list = [x for x in regName_list if str(x) != 'nan']
                for regName in regName_list:
                    # subDir = (regType.upper() + '/' + regName.upper().replace(' ', '_') + '/').replace(',', '')
                    subDir = regType.upper() + '/' + regName.upper().replace(' ', '_').replace(',', '').replace('/', '_').replace('(', '').replace(')', '') + '/'
                    ipoints = [i for i in range(int(df_association.shape[0])) if str(df_association[regType][i]) == regName]
                    df_reg = df_reg.append({'regType': regType, 'regName': regName, 'subDir': subDir, 'ipoints': ipoints}, ignore_index=True)

            df_reg.to_csv(pathOutReg + 'regions.csv')

            # # Plot all regions and their points to detect errors that must be corrected manually
            # df_reg = pd.read_csv(pathAux+'ASSOCIATION/'+targetVar.upper()+'/regions.csv')
            # for index, row in df_reg.iterrows():
            #     regType, regName = row['regType'], row['regName']
            #     iaux = [int(x) for x in row['ipoints'][1:-1].split(', ')]
            #     print(regType, regName, str(index) + '/' + str(df_reg.shape[0]))
            #     plot.map(map(targetVar, np.zeros((len(iaux))), 'prob', path=pathAux+'ASSOCIATION/Maps_regions/',
            #              filename=regType+'_'+regName, title=regType+' '+regName, plot_mode='scatter', regType=regType,
            #              regName=regName)

        else:
            print('nameCompleteRegion ' + nameCompleteRegion + ' not implemented.')
            print('grids.subregions() needs to be tuned for this new region/project.')
            print('shapefiles with polygons or any other way to define regions for each hres point are needed.')
            exit()

    # If no divission by regions is made
    else:
        # Create regions file
        df_reg = pd.DataFrame(columns=['regType', 'regName', 'subDir', 'ipoints'])
        regType = typeCompleteRegion
        regName_list = list(df_association[regType].unique())
        regName_list = [x for x in regName_list if str(x) != 'nan']
        for regName in regName_list:
            # subDir = (regType.upper() + '/' + regName.upper().replace(' ', '_') + '/').replace(',', '')
            subDir = regType.upper() + '/' + regName.upper().replace(' ', '_').replace(',', '').replace('/', '_').replace('(', '').replace(')', '') + '/'
            ipoints = [i for i in range(int(df_association.shape[0])) if str(df_association[regType][i]) == regName]
            df_reg = df_reg.append({'regType': regType, 'regName': regName, 'subDir': subDir, 'ipoints': ipoints}, ignore_index=True)
        df_reg.to_csv(pathOutReg + 'regions.csv')
