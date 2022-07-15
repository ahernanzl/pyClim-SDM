input_data folder must contain the following three subdirectories:
- hres/
- reanalysis/
- models/

Some example datasets have been included. In order to test pyClim-SDM using the example datasets, rename 
input_data_template/ as input_data/, run src/gui_mode.py and try different methods and steps
This will allow you to get familiar with pyClim-SDM with no need from your side to prepare any input data.

In order to use your own datasets, spatial domain, etc., prepare your input_data directory as explained hereafter.
Beware that the targetVariables themselves, given by reanalysis/GCMs are mandatory files for some methods and purposes.


# hres/

This folder must contain predictands information with the following format:

For maximum temperature:
- tasmax_hres_metadata.txt: one row per grid point with columns: id, lon, lat, in order.
- tasmax_$hresPeriodFilename['taxmas'].txt

For minimum temperature:
- tasmin_hres_metadata.txt: one row per grid point with columns: id, lon, lat, in order.
- tasmin_$hresPeriodFilename['tasmin'].txt

For precipitation:
- pr_hres_metadata.txt: one row per grid point with columns: id, lon, lat, in order.
- pr_$hresPeriodFilename['pr'].txt 

Example: pr_19510101-20191231.txt

tas/tasmax/tasmin in degrees, pr in mm, uas/vas in m/s, hurs in %, clt in %. One row per date. 
The first column corresponds to the date yyyymmdd, and the other rows (as many as grid points) containing data. 
Missing data must be coded as -999.




# reanalysis/

This folder must contain reanalysis data. One netCDF file per variable, with all pressure levels. Filenames must 
follow the following format: $var_$reanalysisName_$reanalysisPeriodFilename.nc

Example: t_ERA5_19790101-20201231.nc ...

Target variables in low resolution given by the reanalysis are mandatory files for many purposes



# models/ 

This folder must contain models data.
One netCDF file per variable, models and scene, with all pressure levels. 
Filenames must follow the following format: $var_$modelName_$scene_$modelRun_$scenePeriodFilename.nc

Example: 
ua_EC-Earth3_historical_r1i1p1f1_19500101-20141231.nc 
ua_EC-Earth3_ssp585_r1i1p1f1_20150101-21001231.nc
...

