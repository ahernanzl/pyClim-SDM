This folder must contain predictands information with the following format:

- hres_metadata.txt: one row per grid point with four columns: id, lon, lat, h. 
- tmax_$hresPeriodFilename.txt
- tmin_$hresPeriodFilename.txt
- pcp_$hresPeriodFilename.txt 

Example: pcp_19510101-20191231.txt

tmax/tmin in degrees and precipitation in mm. One row per date. 
The first column corresponds to the date yyyymmdd, and the other rows (as many as grid points) containing data. 
Missing data must be coded as -999.

