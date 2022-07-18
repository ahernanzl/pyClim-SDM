# This script prepare sfcWind (wind speed) from uas and vas.
# It can be used to prepare other customized target variables
# If so, activate the variables used as inputs (uas and vas in this case) and config/manual_settings.py as predictors
# for at least one variable


import sys
import shutil
shutil.copyfile('../config/manual_settings.py', '../config/settings.py')
sys.path.append(('../config/'))
from imports import *
from settings import *
from advanced_settings import *

from cdo import *
cdo = Cdo()


########################################################################################################################
def prepare_hres():

    pathIn = '../input_data/hres/'
    os.system('cp '+pathInHres + 'uas_hres_metadata.txt ' + pathInHres + 'sfcWind_hres_metadata.txt')

    u = np.load(pathIn + 'uas_19790101-20201231.npy')
    v = np.load(pathIn + 'vas_19790101-20201231.npy')
    w = np.sqrt(u**2 + v**2)
    w[:, 0] = u[:, 0]
    np.savetxt(pathIn+'sfwcWind_19790101-20201231.txt', w, fmt=['%6d'] + ['%6.2f']*(w.shape[1]-1))
    np.save(pathIn+'sfwcWind_19790101-20201231', w)



########################################################################################################################
def prepare_reanalysis():

    pathIn = '../input_data/reanalysis/'
    aux = read.one_direct_predictor('uas', grid='ext')
    times = aux['times']
    u = aux['data']
    v = read.one_direct_predictor('vas', grid='ext')['data']
    print(u.shape, v.shape, len(ext_lats))
    w = np.sqrt(u ** 2 + v ** 2)
    write.netCDF(pathIn, 'sfcWind_ERA5_19790101-20201231.nc', 'sfcWind', w, 'm/s', ext_lats, ext_lons, times)

########################################################################################################################
def prepare_models(pathOut):

    pathIn = '../input_data/models/'
    for scene in scene_names_list:
        scene = scene.replace('.', '').replace('-', '').lower()
        if scene == 'historical':
            periodFilename = historicalPeriodFilename
        else:
            periodFilename = sspPeriodFilename
        for model in model_names_list:
            print(scene, model)
            modelName, modelRun = model.split('_')[0], model.split('_')[1]
            aux = read.one_direct_predictor('uas', grid='ext', model=model, scene=scene)
            times = aux['times']
            u = aux['data']
            v = read.one_direct_predictor('vas', grid='ext', model=model, scene=scene)['data']
            w = np.sqrt(u**2 + v**2)
            write.netCDF(pathIn, 'sfcWind'+ '_' + modelName + '_' + scene + '_' + modelRun + '_' + periodFilename + '.nc',
                         'sfcWind', w, 'm/s', ext_lats, ext_lons, times)




########################################################################################################################
def main():


    prepare_hres()
    prepare_reanalysis()
    prepare_models()




if __name__ == "__main__":
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print("\n------------------------------\nElapsed time: " + str(end-start))