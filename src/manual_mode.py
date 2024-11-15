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

# shutil.copyfile('../config/manual_settings.py', '../config/settings.py')
# shutil.copyfile('../config/default_gui_settings.py', '../config/settings.py')
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *


def main():

    aux = read.netCDF('/MASIVO/clq/AEMET-5KM-regular/from_CSIC_allYears/', 'tmax_1951-2022.nc', 'maxtemp')
    times = aux['times']
    data = aux['data'][:, 0, :, :]
    data = (100 * data).astype('int32')
    print(len(times), data.shape)
    train_ivalid = [i for i in range(len(times)) if times[i].year>=1979 and times[i].year <=2005]
    test_ivalid = [i for i in range(len(times)) if times[i].year>=200 and times[i].year <=2020]
    Y_train = np.zeros((3, len(train_ivalid), data.shape[1], data.shape[2]), dtype=np.int32)
    # Y_test = np.zeros((3, len(test_ivalid), data.shape[1], data.shape[2]), dtype=np.int32)

    Y_train[0] = data[train_ivalid]
    # Y_test[0] = data[test_ivalid]

    data = read.netCDF('/MASIVO/clq/AEMET-5KM-regular/from_CSIC_allYears/', 'tmin_1951-2022.nc', 'mintemp')['data'][:, 0, :, :]
    data = (100 * data).astype('int32')
    Y_train[1] = data[train_ivalid]
    # Y_test[1] = data[test_ivalid]

    data = read.netCDF('/MASIVO/clq/AEMET-5KM-regular/from_CSIC_allYears/', 'pcp_1951-2022.nc', 'precipitation')['data'][:, 0, :, :]
    data = (100 * data).astype('int32')
    Y_train[2] = data[train_ivalid]
    # Y_test[2] = data[test_ivalid]

    Y_train = np.swapaxes(Y_train, 0, 1)
    # Y_test = np.swapaxes(Y_test, 0, 1)
    print(Y_train.shape)
    # print(Y_test.shape)

    path = '/home/sclim/clq/gen-esd/data/'
    np.save(path + 'Y_train', Y_train)
    # np.save(path + 'Y_test', Y_test)
    exit()

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

    #-------------------- experiment = EVALUATION / PROJECTIONS ------------------------------------------------------
    preprocess.preprocess()
    # preprocess.train_methods()
    # process.downscale()
    # postprocess.bias_correction()
    # postprocess.get_climdex()
    # postprocess.plot_results()
    # postprocess.nc2ascii()



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