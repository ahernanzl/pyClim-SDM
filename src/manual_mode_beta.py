import sys
import shutil
shutil.copyfile('../config/manual_settings.py', '../config/settings.py')
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

def main():

    #---------------------- PRECONTROL ---------------------------------------------------------------------------------

    precontrol.predictors_strength()
    precontrol.GCMs_availability()
    precontrol.GCMs_reliability()
    precontrol.GCMs_uncertainty()


    #---------------------- EVALUATION / PROJECTIONS -------------------------------------------------------------------

    aux_lib.initial_checks()
    preprocess.preprocess()
    preprocess.train_methods()
    process.downscale()
    postprocess.bias_correction_projections()
    postprocess.get_climdex()
    postprocess.plot_results()
    postprocess.nc2ascii()



if __name__=="__main__":
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print("\n------------------------------\nElapsed time: " + str(end-start))