import sys
import os
sys.path.append('../lib/')


########################################################################################################################
def write_settings_file(showWelcomeMessage, experiment, steps, methods, reaNames, modNames, preds_t_list, preds_p_list,
                        saf_list, climdex_names, calibration_years, reference_years, historical_years, ssp_years, biasCorr_years,
                        bc_method, single_split_testing_years, fold1_testing_years, fold2_testing_years,
                        fold3_testing_years, fold4_testing_years, fold5_testing_years, hresPeriodFilename_t, hresPeriodFilename_p,
                        reanalysisName, reanalysisPeriodFilename, historicalPeriodFilename,
                        sspPeriodFilename, split_mode, grid_res, saf_lat_up, saf_lon_left, saf_lon_right,
                        saf_lat_down, model_names_list, scene_names_list):

    """This function prepares a new settings file with the user selected options"""

    # Open f for writing
    f = open('../config/settings.py', "w")

    # Write new settings
    f.write("showWelcomeMessage = " + str(showWelcomeMessage) + "\n")
    f.write("experiment = '" + str(experiment) + "'\n")
    f.write("methods = " + str(methods) + "\n")
    f.write("reaNames = " + str(reaNames) + "\n")
    f.write("modNames = " + str(modNames) + "\n")
    f.write("preds_t_list = " + str(preds_t_list) + "\n")
    f.write("preds_p_list = " + str(preds_p_list) + "\n")
    f.write("saf_list = " + str(saf_list) + "\n")
    f.write("calibration_years = (" + str(calibration_years[0]) + ", " + str(calibration_years[1]) + ")\n")
    f.write("reference_years = (" + str(reference_years[0]) + ", " + str(reference_years[1]) + ")\n")
    f.write("historical_years = (" + str(historical_years[0]) + ", " + str(historical_years[1]) + ")\n")
    f.write("ssp_years = (" + str(ssp_years[0]) + ", " + str(ssp_years[1]) + ")\n")
    f.write("biasCorr_years = (" + str(biasCorr_years[0]) + ", " + str(biasCorr_years[1]) + ")\n")
    if bc_method == None:
        f.write("bc_method = " + str(bc_method) + "\n")
    else:
        f.write("bc_method = '" + str(bc_method) + "'\n")

    f.write("single_split_testing_years = (" + str(single_split_testing_years[0]) + ", " + str(single_split_testing_years[1]) + ")\n")
    f.write("fold1_testing_years = (" + str(fold1_testing_years[0]) + ", " + str(fold1_testing_years[1]) + ")\n")
    f.write("fold2_testing_years = (" + str(fold2_testing_years[0]) + ", " + str(fold2_testing_years[1]) + ")\n")
    f.write("fold3_testing_years = (" + str(fold3_testing_years[0]) + ", " + str(fold3_testing_years[1]) + ")\n")
    f.write("fold4_testing_years = (" + str(fold4_testing_years[0]) + ", " + str(fold4_testing_years[1]) + ")\n")
    f.write("fold5_testing_years = (" + str(fold5_testing_years[0]) + ", " + str(fold5_testing_years[1]) + ")\n")


    f.write("hresPeriodFilename = {}\n")
    f.write("hresPeriodFilename.update({'t': '" + str(hresPeriodFilename_t) + "'})\n")
    f.write("hresPeriodFilename.update({'p': '" + str(hresPeriodFilename_p) + "'})\n")
    f.write("reanalysisName = '" + str(reanalysisName) + "'\n")
    f.write("reanalysisPeriodFilename = '" + str(reanalysisPeriodFilename) + "'\n")
    f.write("historicalPeriodFilename = '" + str(historicalPeriodFilename) + "'\n")
    f.write("sspPeriodFilename = '" + str(sspPeriodFilename) + "'\n")
    f.write("split_mode = '" + str(split_mode) + "'\n")
    f.write("grid_res = " + str(grid_res) + "\n")
    f.write("saf_lat_up = " + str(saf_lat_up) + "\n")
    f.write("saf_lon_left = " + str(saf_lon_left) + "\n")
    f.write("saf_lon_right = " + str(saf_lon_right) + "\n")
    f.write("saf_lat_down = " + str(saf_lat_down) + "\n")


    f.write("model_names_list = " + str(model_names_list) + "\n")
    f.write("scene_names_list = " + str(scene_names_list) + "\n")
    f.write("climdex_names = " + str(climdex_names) + "\n")

    # Close f
    f.close()


########################################################################################################################
def write_tmpMain_file(steps):

    """This function prepares a tmp main file with the user selected options"""

    # Open f for writing
    f = open('.tmp_main.py', "w")


    f.write("import sys\n")
    f.write("sys.path.append('../config/')\n")
    f.write("from imports import *\n")
    f.write("from settings import *\n")
    f.write("from advanced_settings import *\n")

    f.write("\n")
    f.write("def main():\n")

    # Steps
    noSteps = True
    if len(steps) > 0:
        f.write("    aux_lib.initial_checks()\n")
    if 'preprocess' in steps:
        noSteps = False
        f.write("    preprocess.preprocess()\n")
    if 'missing_data_check' in steps:
        noSteps = False
        f.write("    precontrol.missing_data_check()\n")
    if 'predictors_correlation' in steps:
        noSteps = False
        f.write("    precontrol.predictors_correlation()\n")
    if 'GCMs_evaluation' in steps:
        noSteps = False
        f.write("    precontrol.GCMs_evaluation()\n")
    if 'train_methods' in steps:
        noSteps = False
        f.write("    preprocess.train_methods()\n")
    if 'downscale' in steps:
        noSteps = False
        f.write("    process.downscale()\n")
    if 'calculate_climdex' in steps:
        noSteps = False
        f.write("    postprocess.get_climdex()\n")
    if 'plot_results' in steps:
        noSteps = False
        f.write("    postprocess.plot_results()\n")
    if 'bias_correct_projections' in steps:
        noSteps = False
        f.write("    postprocess.bias_correction_projections()\n")
    if 'nc2ascii' in steps:
        noSteps = False
        f.write("    postprocess.nc2ascii()\n")
    if noSteps == True:
        print('-----------------------------------------------')
        print('At least one step must be selected.')
        print('-----------------------------------------------')
        exit()

    f.write("\n")
    f.write("if __name__ == '__main__':\n")
    f.write("    start = datetime.datetime.now()\n")
    f.write("    main()\n")
    f.write("    end = datetime.datetime.now()\n")
    f.write("    print('Elapsed time: ' + str(end - start))")

    # Close f
    f.close()


########################################################################################################################
def main():
    """
    This function shows a graphical dialog to select settings and launch the main program.
    """

    # Welcome message
    run, showWelcomeMessage = gui_lib.welcomeMessage().get()
    if run == False:
        exit()

    # Seletcion window
    run, experiment, steps, methods, reaNames, modNames, preds_t_list, preds_p_list, saf_list, climdex_names, \
        calibration_years, reference_years, historical_years, ssp_years, biasCorr_years, bc_method, \
        single_split_testing_years, fold1_testing_years, fold2_testing_years, fold3_testing_years, fold4_testing_years, \
        fold5_testing_years, hresPeriodFilename_t, hresPeriodFilename_p, reanalysisName, reanalysisPeriodFilename, \
        historicalPeriodFilename, sspPeriodFilename, split_mode, grid_res, \
        saf_lat_up, saf_lon_left, saf_lon_right, saf_lat_down, model_names_list, scene_names_list, \
        = gui_lib.selectionWindow().get()
    if run == False:
        exit()

    # Write settings file
    write_settings_file(showWelcomeMessage, experiment, steps, methods, reaNames, modNames, preds_t_list, preds_p_list,
                        saf_list, climdex_names, calibration_years, reference_years, historical_years, ssp_years, biasCorr_years,
                        bc_method, single_split_testing_years, fold1_testing_years, fold2_testing_years,
                        fold3_testing_years, fold4_testing_years, fold5_testing_years, hresPeriodFilename_t, hresPeriodFilename_p,
                        reanalysisName, reanalysisPeriodFilename, historicalPeriodFilename,
                        sspPeriodFilename, split_mode, grid_res, saf_lat_up, saf_lon_left, saf_lon_right,
                        saf_lat_down, model_names_list, scene_names_list)

    # Write tmp_main file
    write_tmpMain_file(steps)

    # Launch tmp_main
    os.system('python3 .tmp_main.py')

    # Delete tmp_main
    os.remove('.tmp_main.py')

if __name__=="__main__":
    main()
