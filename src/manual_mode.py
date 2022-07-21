import sys
import shutil
shutil.copyfile('../config/manual_settings.py', '../config/settings.py')
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *



def main():

    # for (path, subdirs, files) in os.walk('../results/'):
    #     for name in files:
    #         for var in ('sfcWind', 'fwi'):
    #             for climdex in ('m', 'n', 'x', '90p'):
    #                 # print(var+climdex)
    #                 if var+climdex in name:
    #                     print('----------------')
    #                     # print(name)
    #                     file_path = os.path.join(path,name)
    #                     new_name = os.path.join(path,name.replace(var+climdex,var.upper()+climdex))
    #                     print(file_path)
    #                     print(new_name)
    #                     # os.rename(file_path, new_name)
    # exit()



    # aux_lib.check_var_units()
    # exit()
    #
    #---------------------- INITIAL CHECKS ---------------------------------------------------------------------------------
    aux_lib.initial_checks()
    #
    # #---------------------- experiment = PRECONTROL --------------------------------------------------------------------
    #
    # preprocess.preprocess()
    # precontrol.missing_data_check()
    # precontrol.predictors_correlation()
    # precontrol.GCMs_evaluation()

    #-------------------- experiment = EVALUATION / PROJECTIONS ------------------------------------------------------
    # preprocess.preprocess()
    # preprocess.train_methods()
    # process.downscale()
    # postprocess.bias_correction()
    # postprocess.get_climdex()
    postprocess.plot_results()
    # postprocess.nc2ascii()


if __name__=="__main__":
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print("\n------------------------------\nElapsed time: " + str(end-start))