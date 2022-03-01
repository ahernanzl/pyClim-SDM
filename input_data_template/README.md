input_data folder must contain the following three subdirectories:
- hres/
- reanalysis/
- models/

Some example datasets have been included. In order to test pyClim-SDM using the example datasets, rename 
input_data_template/ as input_data/, run src/gui_mode.py and limit your selection to the following options:
- experiment: EVALUATION
- predictors:
  - Maximum/minimum Temperature: tmax
  - Precipitation: mslp
  - Synoptic Analogy Fields: mslp

This will allow you to test how pyClim-SDM works with no need from your side to prepare any input data. 
Nevertheless, datasets here included are only meant for tests, downscaling a reanalysis over a set of 15 locations
and using a very poor set of predictors. 

In order to expand to other options/predictors, additional datasets should be included and prepared as explained 
hereafter.



#------------------------------------------------------------------------------------------------------------------<br/>
#--------------------------------- hres/ --------------------------------------------------------------------------<br/>
#------------------------------------------------------------------------------------------------------------------<br/>

This folder must contain predictands information with the following format:

- hres_metadata.txt: one row per grid point with four columns: id, lon, lat, h. 
- tmax_$hresPeriodFilename.txt
- tmin_$hresPeriodFilename.txt
- pcp_$hresPeriodFilename.txt 

Example: pcp_19510101-20191231.txt

tmax/tmin in degrees and precipitation in mm. One row per date. 
The first column corresponds to the date yyyymmdd, and the other rows (as many as grid points) containing data. 
Missing data must be coded as -999.



#------------------------------------------------------------------------------------------------------------------<br/>
#--------------------------------- reanalysis/ --------------------------------------------------------------------<br/>
#------------------------------------------------------------------------------------------------------------------<br/>

This folder must contain reanalysis data. One netCDF file per variable, with all pressure levels. Filenames must 
follow the following format: $var_$reanalysisName_$reanalysisPeriodFilename.nc

Example: t_ERA5_19790101-20201231.nc ...



#------------------------------------------------------------------------------------------------------------------<br/>
#--------------------------------- models/ ------------------------------------------------------------------------<br/>
#------------------------------------------------------------------------------------------------------------------<br/>

This folder must contain models data.
One netCDF file per variable, models and scene, with all pressure levels. 
Filenames must follow the following format: $var_$modelName_$scene_$modelRealizationFilename_$scenePeriodFilename.nc

Example: 
ua_EC-Earth3_historical_r1i1p1f1_19500101-20141231.nc 
ua_EC-Earth3_ssp585_r1i1p1f1_20150101-21001231.nc
...

