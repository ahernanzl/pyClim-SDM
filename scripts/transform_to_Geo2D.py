import pandas as pd
import numpy as np
import netCDF4 as nc
import os

'''
Script to transform 1D NetCDF files into Geo2D NetCDF files.
The goal is to create Geo2D NetCDFs that can be used with geographical tools like CDO, GIS, Panolpy, GDAL, etc.
This script processes projection data from pyClim-SDM and saves it in a format that can be used for further analysis.
Note: The limitation applies to geographical tools, as they do not support 1D NetCDFs. In CDO, temporal operations like 'timmean' can be applied to 1D NetCDFs, but spatial operations like 'sellonlatbox' require a Geo2D format.
For more information about pyClim-SDM, visit: https://gitlab.aemet.es/aemc/scenarios/pyClim-SDM
And for the associated paper, check: https://doi.org/10.1016/j.cliser.2023.100408
'''

########################################################################################################################
def dataframe_to_matrix(df, latitudes, longitudes, var):
    """Convert a DataFrame to a 2D matrix based on latitude and longitude."""
    # Pivot the DataFrame by latitude and longitude, then reindex according to the provided latitudes and longitudes
    return df.pivot(index='latitude', columns='longitude', values=var).reindex(index=latitudes, columns=longitudes).values

########################################################################################################################
def process_variable(var, bias, model_names_list, scene_list, pathIn, pathOut, method_dict, institution_name, grid, domain):
    """Process a variable and create Geo2D NetCDF files with metadata."""
    # Extract method details for the variable from the method dictionary
    method_info = method_dict[var]
    method = method_info['method']
    varname = method_info['varname']

    # Iterate over each model and scene
    for model_name in model_names_list:
        for scene in scene_list:
            # Define input and output file paths
            nc_pathIn = f'{pathIn}/results/PROJECTIONS/{var.upper()}/{method}{bias}/daily_data/{model_name}_{scene}.nc'
            nc_pathOut = f'{pathOut}/results/PROJECTIONS/{var.upper()}/{method}{bias}/daily_data/{model_name}_{scene}.nc'

            # Skip if output file already exists
            if os.path.exists(nc_pathOut):
                print(f"Skipping existing file: {nc_pathOut}")
                continue  

            # Check if input file exists
            if not os.path.exists(nc_pathIn):
                print(f"Input file not found: {nc_pathIn}")
                continue

            # Open the input NetCDF file for reading
            with nc.Dataset(nc_pathIn, 'r') as file:
                targetvar = file.variables[var]
                lon = np.round(file.variables['lon'][:], decimals=5)  # Longitude values
                lat = np.round(file.variables['lat'][:], decimals=5)  # Latitude values
                time = file.variables['time'][:]  # Time values
                lat_unique = np.sort(np.unique(lat))  # Unique latitudes
                lon_unique = np.sort(np.unique(lon))  # Unique longitudes

                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(nc_pathOut), exist_ok=True)

                # Create a new NetCDF file for writing
                with nc.Dataset(nc_pathOut, 'w') as ncfile:
                    # Create dimensions for time, latitude, and longitude
                    ncfile.createDimension('time', None)
                    ncfile.createDimension('lat', len(lat_unique))
                    ncfile.createDimension('lon', len(lon_unique))

                    # Create variables for target variable, latitude, longitude, and time
                    targetvar_var = ncfile.createVariable(var, 'f4', dimensions=('time', 'lat', 'lon'))
                    lat_var = ncfile.createVariable('lat', 'f4', dimensions=('lat',))
                    lon_var = ncfile.createVariable('lon', 'f4', dimensions=('lon',))
                    time_var = ncfile.createVariable('time', 'f4', dimensions=('time',))

                    # Store data in variables
                    lat_var[:] = lat_unique
                    lon_var[:] = lon_unique
                    time_var[:] = time

                    # Process data for each time step and write it to the NetCDF file
                    for i in range(len(time)):
                        print(f"Processing {var} - Model: {model_name}, Scenario: {scene}, Time {i+1}/{len(time)}")
                        df = pd.DataFrame({var: targetvar[i, :], 'longitude': lon, 'latitude': lat})
                        matrix_targetvar = dataframe_to_matrix(df, lat_unique, lon_unique, var)
                        targetvar_var[i, :, :] = matrix_targetvar  

                    # Copy global attributes and variable-specific attributes from the input to the output file
                    ncfile.setncatts(file.__dict__)  # Copy global attributes
                    ncfile.variables['time'].setncatts(file.variables['time'].__dict__)  
                    ncfile.variables['lat'].setncatts(file.variables['lat'].__dict__)  
                    ncfile.variables['lon'].setncatts(file.variables['lon'].__dict__)  
                    ncfile.variables[var].setncatts(file.variables[var].__dict__)  

                    # Add or overwrite specific metadata attributes
                    ncfile.setncattr("Conventions", "CF-1.8")
                    ncfile.institution = institution_name  
                    ncfile.title = f'Dataset ref: {grid} / Domain: {domain} / GCM: {model_name}_{scene} CMIP6 / ESD: {method}{bias} / Variable: {varname} / Daily data'

                    # Ensure all data is written to the file
                    ncfile.sync()

            print(f"File {nc_pathOut} saved successfully with metadata.")


########################################################################################################################
def main():
    """Main function to process NetCDF files."""
    
    # Institution name for metadata
    institution_name = "Your Institution Name"  # Replace with actual institution name

    # Paths for input and output files
    pathIn = "/path/to/input" # path where your pyClim-SDM directory is located 
    pathOut = "/path/to/output" # path where transformed netCDFs will be saved
    
    # Define area details (grid and domain)
    area_dict = {
        'peninsulaybaleares': {'grid': 'Grid 5x5 km - AEMET', 'domain': 'Iberian Peninsula and Balearic Islands'},
        'canarias': {'grid': 'Grid 2.5x2.5 km - AEMET', 'domain': 'Canary Islands'}
    }
    
    # Select the area for processing
    area = 'peninsulaybaleares'  # or 'canarias'
    grid = area_dict[area]['grid']
    domain = area_dict[area]['domain']
    
    # Method details for variables (e.g., precipitation, temperature): Downscaling method and variable name
    method_dict = {
        'pr': {'method': 'XGB', 'varname': 'precipitation'},
        'tasmax': {'method': 'MLR-ANA', 'varname': 'maximum temperature'},
        'tasmin': {'method': 'MLR-ANA', 'varname': 'minimum temperature'}
    }

    # List of model names 
    model_names_list = [
        'ACCESS-CM2_r1i1p1f1',
        'CMCC-CM2-SR5_r1i1p1f1',
        'EC-Earth3-Veg_r1i1p1f1',
        'IITM-ESM_r1i1p1f1',
        'KACE-1-0-G_r1i1p1f1',
        'MIROC6_r1i1p1f1',
        'MRI-ESM2-0_r1i1p1f1',
        'MPI-ESM1-2-HR_r1i1p1f1',
        'NorESM2-MM_r1i1p1f1',
        'CNRM-ESM2-1_r1i1p1f2',
        'UKESM1-0-LL_r2i1p1f2',
    ]
    
    # List of scenarios
    scene_list = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    # List of variables
    var_list = ['pr', 'tasmax', 'tasmin']
    # List of bias correction methods (leave an empty string if no bias correction is applied)
    bias_list = ['', '+QDMs']

    # Loop through each bias correction method and variable to process the data
    for bias in bias_list:
        for var in var_list:
            process_variable(var, bias, model_names_list, scene_list, pathIn, pathOut, method_dict, institution_name, grid, domain)

# Run the main function
if __name__ == "__main__":
    main()
