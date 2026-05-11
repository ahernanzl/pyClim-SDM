import sys
import shutil
import os

version = ''
# Save version last execution
for file in os.listdir('../doc/'):
    if file.startswith('User_Manual_'):
        version = str(file.split('.')[0].split('_')[-1])
text_file = open('../config/.version_last_execution.txt', "w")
text_file.write(version)
text_file.close()

shutil.copyfile('../config/manual_settings.py', '../config/settings.py')
# shutil.copyfile('../config/default_gui_settings.py', '../config/settings.py')
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *


def main():

    #---------------------- INITIAL CHECKS ---------------------------------------------------------------------------------
    aux_lib.initial_checks()
    #
    # #---------------------- experiment = PRECONTROL --------------------------------------------------------------------
    #
    # aux_lib.check_var_units()
    # preprocess.preprocess()
    # precontrol.missing_data_check()
    # precontrol.predictors_correlation()
    # precontrol.GCMs_evaluation()

    # preprocess.preprocess()
    # precontrol.kolmogorov_smirnov_calculate()
    # precontrol.kolmogorov_smirnov_plot()
    # precontrol.pca_calculate()
    # precontrol.pca_plot()
    # precontrol.spread_GCM_fut_calculate()
    # precontrol.spread_GCM_fut_plot()

    #-------------------- experiment = EVALUATION / PROJECTIONS ------------------------------------------------------
    # preprocess.preprocess()
    # preprocess.train_methods()
    # process.downscale()
    # postprocess.trend_injection()
    # postprocess.bias_correction()
    # postprocess.get_climdex()
    # postprocess.plot_results()
    # postprocess.nc2ascii()
    # postprocess.nc1D_to_nc2D()



if __name__=="__main__":
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print('-------------------------------------------------------------------')
    print("Elapsed time: " + str(end-start))
    if running_at_HPC == True:
        print("pyClim-SDM has finished, but submited jobs can be still running")
        print("Do not launch more jobs until they have succesfully finished")
    print('-------------------------------------------------------------------')
