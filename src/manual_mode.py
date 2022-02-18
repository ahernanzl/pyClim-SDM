import sys
import os
os.system('cp ../config/manual_settings.py ../config/settings.py')
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

def main():

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
