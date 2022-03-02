input_data folder must contain the following three subdirectories:
- hres/
- reanalysis/
- models/

Some example datasets have been included. In order to test pyClim-SDM using the example datasets, rename 
input_data_template/ as input_data/, run src/gui_mode.py and limit your selection to experiment=EVALUATION
using the default sets of predictors.
This will allow you to get familiar with pyClim-SDM with no need from your side to prepare any input data.

In order to use your own datasets, spatial domain, etc., prepare your input_data directory as explained hereafter.


# hres/

This folder must contain predictands information with the following format:

- hres_metadata.txt: one row per grid point with four columns: id, lon, lat, h. 
- tmax_$hresPeriodFilename.txt
- tmin_$hresPeriodFilename.txt
- pcp_$hresPeriodFilename.txt 

Example: pcp_19510101-20191231.txt

tmax/tmin in degrees and precipitation in mm. One row per date. 
The first column corresponds to the date yyyymmdd, and the other rows (as many as grid points) containing data. 
Missing data must be coded as -999.




# reanalysis/

This folder must contain reanalysis data. One netCDF file per variable, with all pressure levels. Filenames must 
follow the following format: $var_$reanalysisName_$reanalysisPeriodFilename.nc

Example: t_ERA5_19790101-20201231.nc ...




# models/ 

This folder must contain models data.
One netCDF file per variable, models and scene, with all pressure levels. 
Filenames must follow the following format: $var_$modelName_$scene_$modelRealizationFilename_$scenePeriodFilename.nc

Example: 
ua_EC-Earth3_historical_r1i1p1f1_19500101-20141231.nc 
ua_EC-Earth3_ssp585_r1i1p1f1_20150101-21001231.nc
...

