import sys
import shutil
shutil.copyfile('../config/manual_settings.py', '../config/settings.py')
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

def main():
    path = '/MASIVO/clq/para_github_datos/hres/'
    n = 16
    for var in ('tmax', 'tmin', 'pcp'):
        print(var)
        data = np.loadtxt(path+var+'_19510101-20201231.txt')
        print(data)
        print(data.shape)
        data = data[:, :n]
        np.savetxt(path+var+'_19510101-20201231_new.txt', data, fmt=['%.i'] + ['%.2f'] * (n - 1))

    exit()

    preprocess.preprocess()
    preprocess.train_methods()
    process.downscale()
    postprocess.bias_correction_projections()
    postprocess.get_climdex()
    postprocess.plot_results()
    postprocess.nc2ascii()

if __name__=="__main__":
    start = datetime.datetime.now()
    aux_lib.initial_checks()
    main()
    end = datetime.datetime.now()
    print("\n------------------------------\nElapsed time: " + str(end-start))
