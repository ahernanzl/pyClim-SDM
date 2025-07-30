# pyClim-SDM: Statistical Downscaling for Climate Change Projections with a Graphical User Interface  

<br/><br/>
![](doc/gui_lr.gif)
<br/><br/>
![](doc/app_lr.gif)
<br/><br/>

**Institution:** Spanish Meteorological Agency (AEMET)

**License:** GNU General Public License v3.0

**Citation:** Hernanz, A., Correa, C., García-Valero, J. A., Domínguez, M., Rodríguez-Guisado, E., & Rodríguez-Camino, E. (2023). pyClim-SDM: Service for generation of statistical downscaled climate change projections supporting national adaptation strategies. Climate Services. https://doi.org/10.1016/j.cliser.2023.100408
___

pyClim-SDM is a software for statistical downscaling of climate change projections for the following daily surface variables: mean, maximum and minimum temperature, precipitation, zonal and meridional wind components, relative and specific humidity, cloud cover, surface downwelling shortwave and longwave radiation, evaporation, potential evaporation, sea level pressure, surface pressure, total runoff and soil water content.
Additionally, it is prepared for downscaling any other user defined variable. pyClim-SDM incorporates the following utilities:
- evaluation of Global Climate Models (GCMs).
- downscaling of both reanalysis (for evaluation) and GCMs.
- bias correction of downscaled climate projections.
- post-processing: ETCCDI extreme climate indices (Karl *et al*., 1999; https://www.climdex.org/learn/indices/).
- visualization of downscaled projections and evaluation metrics.
- integrated dynamic map viewer


# Methods

### Raw:
- **RAW-BIL**: no downscaling (bilinear interpolation).
### Analogs / Weather Typing:
- **MLR-ANA**: multiple linear regression based on analogs. See Petisco de Lara (2008b), Amblar-Francés *et al*. (2017) and Hernanz *et al.* (2021).
- **ANA-SYN-1NN**: Analog based on synoptic analogy. Nearest analogSee Hernanz *et al.* (2021).      
### Linear:
- **MLR**: multiple linear regression. See Amblar-Francés *et al*., (2017) and Hernanz *et al.* (2021). Based on SDSM (Wilby *et al.*, 2002).
- **GLM**: Generalized Linear Model. Logistic + MLR (**LIN**), or over transformed data (**EXP** for exponential). See Amblar-Francés *et al*. (2017) and Hernanz *et al.* (2021). Based on SDSM (Wilby *et al.*, 2002).   
### Weather Generators:
- **WG-PDF**: Downscaling parameters of the distributions instead of downscaling daily data. See Erlandsen *et al.* (2020) and Benestad (2021).
### Machine Learning:
- **XGB**: eXtreme Gradient Boost. Non-linear machine learning classification/regression. This method is combined with a MLR to extrapolate to values out of the observed range (configurable).
- **DeepESD**: Convolutional Neural Networks. See Baño-Medina *et al*., (2022)


# How to use
- Download and install pyClim-SDM (see Installation section)
- Prepare your input data (see Input data section)
- Run pyClim-SDM:
  - cd src
  - python gui_mode.py
A graphical window will open. Make your selection and press the Run button. The graphical window will close and 
pyClim-SDM will start the process in the terminal, where information messages will be shown.

# Map viewer
pyClim-SDM has an integrated dynamic map viewer. After having generated downscaled data for experiment PROJECTIONS (with or without bias correction), launch the map viewer by:
 - cd map_viewer
 - python app.py

Open an internet browser and enter the following url: 127.0.0.1/8050


# Installation
pyClim-SDM has been originally designed for **Linux** and might present problems over a different OS.

Clone the repository or download the source code from gitlab.

In order to use pyClim-SDM, **python3.10** is required. pyClim-SDM makes use of the python libaries listed at 
requirements.txt. You can install them by following these steps: 
- Install Miniconda 3 (6Gb aprox. needed): https://docs.conda.io/en/latest/miniconda.html
- Create a virtual environment:
  - conda update -n base -c defaults conda
  - basic installation: **conda create -n env_pyClim-SDM python=3.10 absl-py=2.1.0 Bottleneck=1.4.2 numpy=1.26.4 pandas=2.2.3 geopandas=1.0.1 geopy=2.4.1 matplotlib=3.10.1 netCDF4=1.6.0 scikit-learn=1.6.1 scipy=1.15.2 seaborn=0.13.2 shapely=2.0.6 statsmodels=0.14.4 tensorflow=2.17.0 xarray=2025.1.2 xgboost=2.1.4 cartopy=0.24.0 torchvision=0.14.1 torchaudio=2.5.1 pytorch=2.5.1 dash=2.10.2 dash_bootstrap_components=1.7.1 plotly=5.6.0 -y -c conda-forge**
  - HPC additional libraries: mpi4py 
  - GPU additional libraries: pytorch-cuda -c pytorch -c nvidia
After installation:
- Activate your environment: **conda activate env_pyClim-SDM**
- Deactivate your environment: **conda deactivate**
**Warning**: Alternatively, if the basic installation fails, you can try **conda create -f environment.yml**

# Input data
Three types of datasets are needed:
hres: high-resolution observations. 
reanalysis: predictors from a reanalysis
models: predictors from GCMs
Do not split years, prepare your files only with complete years.

Format:
- hres format (high-resolution observations):
  One row per date. The first column corresponds to the date yyyymmdd, and the other rows (as many as target points) contain data (if observations come from a regular 2D grid, they need to be flattened to a 1D list of points). Missing data must be coded as -999.
  - Temperature (tas/tasmax/tasmin) in degrees
  - Precipitation (pr) in mm
  - Wind (uas/vas/sfcWind) in m/s
  - Relative humidity (hurs) in %
  - Specific humidity (huss) dimensionless
  - Cloud cover (clt) in %
  - Radiation (shortwave rsds and longwave rlds) in W/m2
  - Evaporation (evspsbl) and potential evaporation (evspsblpot) in kg m-2 s-1
  - Sea level and surface pressure (psl and ps) in Pa
  - Total runoff (mrro) in kg m-2 s-1
  - Soil water content (mrso) in kg m-2
- Reanalysis and models format (low resolution predictors for calibration and downscaling): One netCDF file per variable, models and scene, with all pressure levels, in a regular 2D grid.
Filenames: filenames are composed of specific fields separated by ‘_’, so the use of this symbol inside a field (the reanalysis name, for example) must be avoided.


# References

- Amblar-Francés, P., Casado-Calle, M.J., Pastor-Saavedra, M.A., Ramos-Calzado, P., and Rodríguez-Camino, E. (2017). Guía de escenarios regionalizados de cambio climático sobre España a partir de los resultados del IPCC-AR5. Available at: https://www.aemet.es/documentos/es/conocermas/recursos_en_linea/publicaciones_y_estudios/publicaciones/Guia_escenarios_AR5/Guia_escenarios_AR5.pdf

- Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad, J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model Dev., 15, 6747–6758, https://doi.org/10.5194/gmd-15-6747-2022, 2022.

- Benestad, R.E. (2021) A Norwegian approach to downscaling. Geoscientific Model Development Discussion (in review). https://doi.org/10.5194/gmd-2021-176. 

- Cannon, A.J., S.R. Sobie, and T.Q. Murdock (2015). Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes?. J. Climate, 28, 6938–6959, https://doi.org/10.1175/JCLI-D-14-00754.1

- Erlandsen, H.B., Parding, K.M., Benestad, R., Mezghani, A. and Pontoppidan, M. (2020). A hybrid downscaling approach for future temperature and precipitation change. Journal of Applied Meteorology and Climatology, 59(11), 1793–1807. https://doi.org/10.1175/JAMC-D-20-0013.1

- García-Valero, J.A. (2021). Redes neuronales artificiales. Aplicación a la regionalización de la precipitación y temperaturas diarias. In: AEMET Nota Técnica, Área de Evaluación y Modelización del Cambio Climático. Spain: AEMET. https://www.aemet.es/documentos/es/conocermas/recursos_en_linea/publicaciones_y_estudios/publicaciones/NT_34_Redes_neuronales_artificiales/NT_34_Redes_neuronales_artificiales.pdf

- Hernanz, A., García-Valero, J. A., Domínguez, M., Ramos-Calzado, P., Pastor-Saavedra, M. A. and Rodríguez-Camino, E. (2021). Evaluation of statistical downscaling methods for climate change projections over Spain: present conditions with perfect predictors. International Journal of Climatology, 42( 2), 762– 776. https://doi.org/10.1002/joc.7271

- Karl, T.R., Nicholls, N., Ghazi, A. (1999). CLIVAR/GCOS/WMO workshop on indices and indicators for climate extremes. Workshop summary. Climatic Change 42, 3–7. https://doi.org/10.1007/978-94-015-9265-9_2

- Petisco de Lara, S.E. (2008a). Método de regionalización de precipitación basado en análogos. Explicación y Validación. In: AEMET Nota Técnica 3A, Área de Evaluación y Modelización del Cambio Climático. Spain: AEMET. 

- Petisco de Lara, S.E. (2008b). Método de regionalización de temperatura basado en análogos. Explicación y Validación. In: AEMET Nota Técnica 3B, Área de Evaluación y Modelización del Cambio Climático. Spain: AEMET.

- Richardson, C. W. (1981), Stochastic simulation of daily precipitation, temperature, and solar radiation, Water Resour. Res., 17( 1), 182– 190, https://doi.org/10.1029/WR017i001p0018

- Switanek, M.B., Troch, P.A., Castro, C.L., Leuprecht, A., Chang, H.-I., Mukherjee, R., and Demaria, E.M.C. (2017). Scaled distribution mapping: a bias correction method that preserves raw climate model projected changes, Hydrol. Earth Syst. Sci., 21, 2649–2666, https://doi.org/10.5194/hess-21-2649-2017

- Themeßl, M.J., Gobiet, A. and Leuprecht, A. (2011). Empirical-statistical downscaling and error correction of daily precipitation from regional climate models. Int. J. Climatol., 31: 1530-1544. https://doi.org/10.1002/joc.2168

- Wilby, R., Dawson, C. and Barrow, E.M. (2002). SDSM—a decision support tool for the assessment of regional climate change impacts. Environmental Modelling & Software, 17, 145–157. https://doi.org/10.1016/S1364-8152(01)00060-3



# pyClim-SDM has been used for the following studies:

- Sandel, B., Merow, C., Serra‐Diaz, J. M., & Svenning, J. C. (2025). Disequilibrium in plant distributions: Challenges and approaches for species distribution models. Journal of Ecology, 113(4), 782-794.

- Hernanz, A., Correa, C., Domínguez, M., Rodríguez‐Guisado, E., & Rodríguez‐Camino, E. (2023). Comparison of machine learning statistical downscaling and regional climate models for temperature, precipitation, wind speed, humidity and radiation over Europe under present conditions. International Journal of Climatology, 43(13), 6065-6082.

- Andimuthu, R., Lakshminarayanan, B., Ramaswamy, M., & Joseph, K. (2024). Multivariate drought risk assessment of tropical river basin in South India under SSP scenarios. Theoretical and Applied Climatology, 155(7), 6843-6861.

- Hassan, M., Ejaz, Z., Abbas, S., Mahmood, R., Chishtie, F. A., Shi, X., ... & Naqvi, S. A. A. (2025). Projected climate regime over Pakistan and its implications for hydrology in the Hunza River Basin using CMIP6 GCMs. Climate Dynamics, 63(7), 285.
