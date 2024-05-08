import sys
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import MOS_lib
import aux_lib
import MOS_lib
import down_scene_ANA
import down_scene_MOS
import down_scene_RAW
import down_scene_TF
import down_scene_WG
import down_day
import down_point
import evaluate_methods
import grids
import launch_jobs
import plot
import postpro_lib
import postprocess

import derived_predictors
import precontrol
import preprocess
import process
import read
import transform
import TF_lib
import val_lib
import WG_lib
import write


########################################################################################################################
def netCDF(path, filename, varName, data, units, lats, lons, times, calendar, regular_grid=True, level=None,
		   level_name='level', lat_name='lat', lon_name='lon', time_name='time'):
	"""
	This function writes data to netCDF file.
	"""
	try:
		os.makedirs(path)
	except:
		pass

	if filename[-3:] != '.nc':
		filename += '.nc'

	# Define dataset and dimensions
	nc=Dataset(path+filename, 'w', format='NETCDF4')
	nc.Conventions = "CF-1.8"
	nc.createDimension(time_name, len(times))

	if regular_grid == True:
		nc.createDimension(lat_name, len(lats))
		nc.createDimension(lon_name, len(lons))
	else:
		nc.createDimension('point', len(lats))

	if level is not None:
		nc.createDimension(level_name, 1)
		# levelVar = nc.createVariable(level_name, 'f4', (level_name))
		# levelVar.units='degrees north'
		# levelVar[:] = level

	# # Create time variable
	timeVar = nc.createVariable(varname=time_name, dimensions=(time_name,),datatype='float64')
	timeVar.calendar = calendar
	timeVar.long_name = "Time variable"
	timeVar.units = 'hours since 1900-01-01 00:00:0.0'
	timeVar[:] = date2num(times, units=timeVar.units, calendar=timeVar.calendar)

	# Create lat/lon and data variable
	varName_hres_metadata = varName.split('_')[0]
	
	if regular_grid == True:
		latitude = nc.createVariable(lat_name, 'f4', (lat_name))
		longitude = nc.createVariable(lon_name, 'f4', (lon_name))
		if level is None:
			var = nc.createVariable(varName, 'f4', (time_name, lat_name, lon_name,))
		else:
			var = nc.createVariable(varName, 'f4', (time_name, level_name, lat_name, lon_name,))
	else:
		point = nc.createVariable('point', 'i', 'point')
		point.units = ' '
		point.long_name = ""
		point[:] = range(len(lats))
		ids = list(read.hres_metadata(varName_hres_metadata)['id'].values)
		ids = [str(i) for i in ids]
		point.id_names = ids

		latitude = nc.createVariable(lat_name, 'f4', 'point')
		longitude = nc.createVariable(lon_name, 'f4', 'point')
		var = nc.createVariable(varName, 'f4', (time_name, 'point'))

	var.fill_value = fill_value
	data[np.isnan(data)] = fill_value

	latitude.units = 'degrees_north'
	latitude.long_name = "latitude"
	latitude[:] = lats
	longitude.units = 'degrees_east'
	longitude.long_name = "longitude"
	longitude[:] = lons
	var.units = units
	var.long_name = varName
	var[:] = data

	# # print(nc)
	# for var in nc.variables:
	# 	print(var)
	# 	print(nc.variables[var])
	# 	print(nc.variables[var][:].shape)

	# Write to file
	nc.close()


########################################################################################################################
def netCDF_rotated(path, filename, varName, data, dates):
	"""
	This function writes data to netCDF file.
	"""

	try:
		os.makedirs(path)
	except:
		pass

	df = pd.read_csv('../input_data/hres/rotated_coordinates.csv')
	# print(df)

	nrlons, nrlats = 280, 240
	if varName == 'pcp':
		varName = 'precipitation'
		long_name = "precipitation amount"
		units = "k g m-2"
	elif varName == 'tmax':
		varName = 'maximum temperature'
		long_name = "maximum daily temperature"
		units = "degrees"
	elif varName == 'tmin':
		varName = 'minimum temperature'
		long_name = "minimum daily temperature"
		units = "degrees"

	# Global atributes
	if filename[-3:] != '.nc':
		filename += '.nc'
	nc = Dataset(path+filename, 'w', format='NETCDF4_CLASSIC')
	nc.Conventions = "CF-1.7"
	nc.title = "AEMET High-resolution (0.05 deg) daily gridded " + varName + " climate projections dataset for Peninsular Spain and Balearic Islands"
	nc.institution = "Agencia Estatal de Meteorologia (AEMET, www.aemet.es)"
	nc.history = "Creation year 2020"
	nc.references = ''
	nc.comment = ''

	# Dimensions
	nc.createDimension('rlat', nrlats)
	nc.createDimension('rlon', nrlons)
	nc.createDimension('height', 1)
	nc.createDimension('time', len(dates))

	# Variable time
	times = [datetime.datetime(dates[i].year, dates[i].month,dates[i].day)+datetime.timedelta(hours=12)
			 for i in range(len(dates))]
	timeVar = nc.createVariable(varname='time', dimensions='time', datatype='float64')
	timeVar.long_name = "Time variable"
	timeVar.calendar = "proleptic_gregorian"
	timeVar.units = 'hours since 1900-01-01 00:00:0.0'
	timeVar[:] = date2num(times, units=timeVar.units, calendar="proleptic_gregorian")

	# Variable rlon
	rlonVar = nc.createVariable(varname='rlon', dimensions='rlon', datatype='float64')
	rlonVar.long_name = "longitude in rotated pole grid"
	rlonVar.units = "degrees"
	rlonVar.standard_name = "grid_longitude"
	rlonVar.axis = "X"
	rlonVar[:] = np.arange(-5, 9, .05)

	# Variable rlat
	rlatVar = nc.createVariable(varname='rlat', dimensions='rlat', datatype='float64')
	rlatVar.long_name = "latitude in rotated pole grid"
	rlatVar.units = "degrees"
	rlatVar.standard_name = "grid_latitude"
	rlatVar.axis = "Y"
	rlatVar[:] = np.arange(-6.45, 5.55, .05)

	# Variable rotated_pole
	rotated_pole = nc.createVariable(varname='rotated_pole', datatype='c')
	rotated_pole.grid_mapping_name = "rotated_latitude_longitude"
	rotated_pole.grid_north_pole_latitude = 49.5
	rotated_pole.grid_north_pole_longitude = -186.

	# Variable height
	heightVar = nc.createVariable(varname='height', dimensions='height', datatype='float64')
	heightVar.long_name = "height"
	heightVar.units = "m"
	heightVar.axis = "Z"
	heightVar[:] = 0

	# Variable lat
	latVar = nc.createVariable(varname='lat', dimensions=('rlat', 'rlon'), datatype='float64')
	latVar.long_name = "latitude"
	latVar.units = "degrees_north"
	latVar[:] = np.round(df['lat'].values.reshape(nrlats, nrlons), 3)

	# Variable lon
	lonVar = nc.createVariable(varname='lon', dimensions=('rlat', 'rlon'), datatype='float64')
	lonVar.long_name = "latitude"
	lonVar.units = "degrees_east"
	lonVar[:] = np.round(df['lon'].values.reshape(nrlats, nrlons), 3)

	# Variable with data
	dataVar = nc.createVariable(varname=varName, dimensions=('time', 'height', 'rlat', 'rlon'), datatype='float64')
	dataVar.long_name = long_name
	dataVar.table = 1
	dataVar.units = units
	dataVar.grid_mapping = "rotated_pole"
	dataVar.missing_value = -9999.

	# # Establish relation between rotated and normal coordinates
	# pathIn = '../input_data/hres/'
	# fileIn = 'sfcan20100101a20101231_rot_mask.nc'
	# nc = Dataset(pathIn + fileIn)
	# rlats = np.round(nc.variables['rlat'][:], 2)
	# rlons = np.round(nc.variables['rlon'][:], 2)
	# lats = np.round(nc.variables['lat'][:], 3)
	# lons = np.round(nc.variables['lon'][:], 3)
	# nrlats, nrlons, nlats, nlons = rlats.size, rlons.size, lats.size, lons.size
	# coords = np.zeros((nrlats * nrlons, 6))
	# for irlat in range(nrlats):
	# 	for irlon in range(nrlons):
	# 		lat = lats[irlat, irlon]
	# 		lon = lons[irlat, irlon]
	# 		rlat = rlats[irlat]
	# 		rlon = rlons[irlon]
	# 		print(irlat, irlon, lat, lon, rlat, rlon)
	# 		coords[irlat * nrlons + irlon] = np.array([irlat, irlon, lat, lon, rlat, rlon])
	# rlats = []
	# rlons = []
	# lats = read.hres_metadata(var0)['lats'].values
	# lons = read.hres_metadata(var0)['lons'].values
	# npoints = len(lats)
	# err = .0001
	# for ipoint in range(npoints):
	# 	lat, lon = lats[ipoint], lons[ipoint]
	# 	i = int(np.where((abs(coords[:, 2] - lat) < err) * (abs(coords[:, 3] - lon) < err))[0])
	# 	rlat = round(coords[i, 4], 2)
	# 	rlon = round(coords[i, 5], 2)
	# 	print(ipoint, lat, lon, rlat, rlon, i)
	# 	rlats.append(rlat)
	# 	rlons.append(rlon)
	# rlats = np.array(rlats)
	# rlons = np.array(rlons)
	# np.save('../input_data/hres/rlats', rlats)
	# np.save('../input_data/hres/rlons', rlons)
	rlats = np.load('../input_data/hres/rlats.npy')
	rlons = np.load('../input_data/hres/rlons.npy')
	irlats = (rlats + 6.45) / 0.05
	irlons = (rlons + 5) / 0.05
	irlats = (np.rint(irlats)).astype(int)
	irlons = (np.rint(irlons)).astype(int)

	print('---------------')
	ndates, npoints = data.shape[0], data.shape[1]
	aux = dataVar[:, 0, :, :].T
	data = data.T
	for i in range(npoints): # When doing it without for loop memory cannot handle it
		if i % 1000 == 0:
			print(i, round(100*i/npoints, 2), '%')
		aux[irlons[i], irlats[i]] = data[i]
	dataVar[:] = aux.T[:, np.newaxis, :, :]
	data = data.T

	# print(nc)
	# print('------------------------------------------------------------')
	# for var in nc.variables:
	# 	print('------------------------------------------------------------')
	# 	print(var)
	# 	print(nc.variables[var])
	# 	print(nc.variables[var][:].shape)
	# 	print(nc.variables[var][:])
	# # exit()

	# Write to file
	nc.close()


########################################################################################################################
def netCDF_rotated_seasonal_forecast(path, filename, varName, data, year):
	"""
	This function writes data to netCDF with the format required for S-ClimWare.
	"""

	try:
		os.makedirs(path)
	except:
		pass

	df = pd.read_csv('../input_data/hres/rotated_coordinates.csv')
	# print(df)
	# exit()

	# print(year, data.shape[0])

	nrlons, nrlats = 280, 240
	nmem, npoints = data.shape[0], data.shape[1]

	if varName == 'prlr':
		varName = 'pr'
		long_name = "precipitation ammount"
		units = "kg m-2"
	elif varName == 'tas':
		varName = 'tas'
		long_name = "temperature"
		units = "degrees"

	# Global atributes
	nc = Dataset(path + filename, 'w', format='NETCDF4_CLASSIC')
	nc.CDI = "Climate Data Interface version 1.9.8 (https://mpimet.mpg.de/cdi)"
	nc.Conventions = "CF-1.7"
	nc.title = "AEMET downscalled Seasonal Forecast S5 (ECMWF) accumulated NDJFM " + varName + " (0.05 deg) for Iberia,  Balearic Islands and Southern France"
	nc.institution = "Agencia Estatal de Meteorologia (AEMET, www.aemet.es)"
	nc.references = "https://meetingorganizer.copernicus.org/EMS2019/EMS2019-570.pdf";
	nc.perturbationNumber = 0
	nc.version = "1.0"
	nc.nco_input_file_number = nmem

	# Dimensions
	nc.createDimension('member', nmem)
	nc.createDimension('rlat', nrlats)
	nc.createDimension('rlon', nrlons)
	nc.createDimension('time', 1)
	nc.createDimension('time_bnds', 2)

	# Variable member
	mems = [i for i in range(nmem)]
	memVar = nc.createVariable(varname='member', dimensions='member', datatype='float64')
	memVar.long_name = "member"
	memVar.units = ''
	memVar[:] = mems

	# Variable rlon
	rlonVar = nc.createVariable(varname='rlon', dimensions='rlon', datatype='float64')
	rlonVar.long_name = "longitude in rotated pole grid"
	rlonVar.units = "degrees"
	rlonVar.standard_name = "grid_longitude"
	rlonVar.axis = "X"
	rlonVar[:] = np.arange(-5, 9, .05)

	# Variable rlat
	rlatVar = nc.createVariable(varname='rlat', dimensions='rlat', datatype='float64')
	rlatVar.long_name = "latitude in rotated pole grid"
	rlatVar.units = "degrees"
	rlatVar.standard_name = "grid_latitude"
	rlatVar.axis = "Y"
	rlatVar[:] = np.arange(-6.45, 5.55, .05)

	# # Create time variable
	# times = [datetime.datetime(dates[i].year, dates[i].month,dates[i].day)+datetime.timedelta(hours=12)
	# 		 for i in range(len(dates))]
	timeVar = nc.createVariable(varname='time', dimensions=('time',),datatype='float64')
	timeVar.units = 'hours since '+str(year)+'-11-01 06:00:0.0'
	timeVar.calendar = 'proleptic gregorian'
	# timeVar[:] = date2num(times, units=timeVar.units, calendar=timeVar.calendar)
	timeVar[:] = 1

	# # Create time_bnds variable
	time_bndsVar = nc.createVariable(varname='time_bnds', dimensions=('time_bnds',),datatype='float64')
	first_date = datetime.datetime(year, 11, 1, 6, 0)
	last_date = datetime.datetime(year+1, 4, 1, 6, 0)
	ndays = (last_date-first_date).days+1
	time_bndsVar[:] = [0, ndays]

	# Variable rotated_pole
	rotated_pole = nc.createVariable(varname='rotated_pole', datatype='c')
	rotated_pole.grid_mapping_name = "rotated_latitude_longitude"
	rotated_pole.grid_north_pole_latitude = 49.5
	rotated_pole.grid_north_pole_longitude = -186.

	# Variable lat
	latVar = nc.createVariable(varname='lat', dimensions=('rlat', 'rlon'), datatype='float64')
	latVar.long_name = "latitude"
	latVar.units = "degrees_north"
	latVar[:] = np.round(df['lat'].values.reshape(nrlats, nrlons), 3)

	# Variable lon
	lonVar = nc.createVariable(varname='lon', dimensions=('rlat', 'rlon'), datatype='float64')
	lonVar.long_name = "latitude"
	lonVar.units = "degrees_east"
	lonVar[:] = np.round(df['lon'].values.reshape(nrlats, nrlons), 3)

	# Variable with data
	dataVar = nc.createVariable(varname=varName, dimensions=('member', 'time', 'rlat', 'rlon'),
								datatype='float64')
	dataVar.long_name = long_name
	dataVar.table = 1
	dataVar.units = units
	dataVar.grid_mapping = "rotated_pole"
	dataVar.missing_value = -9999.

	# Read rlats and rlons (or create them)
	try:
		rlats = np.load('../input_data/hres/rlats.npy')
		rlons = np.load('../input_data/hres/rlons.npy')
	except:
		# Establish relation between rotated and normal coordinates
		pathIn = '../input_data/hres/'
		fileIn = 'sfcan20100101a20101231_rot_mask.nc'
		nc = Dataset(pathIn + fileIn)
		rlats = np.round(nc.variables['rlat'][:], 2)
		rlons = np.round(nc.variables['rlon'][:], 2)
		lats = np.round(nc.variables['lat'][:], 3)
		lons = np.round(nc.variables['lon'][:], 3)
		nrlats, nrlons, nlats, nlons = rlats.size, rlons.size, lats.size, lons.size
		coords = np.zeros((nrlats * nrlons, 6))
		for irlat in range(nrlats):
			for irlon in range(nrlons):
				lat = lats[irlat, irlon]
				lon = lons[irlat, irlon]
				rlat = rlats[irlat]
				rlon = rlons[irlon]
				print(irlat, irlon, lat, lon, rlat, rlon)
				coords[irlat * nrlons + irlon] = np.array([irlat, irlon, lat, lon, rlat, rlon])
		rlats = []
		rlons = []
		lats = read.hres_metadata(var0)['lats'].values
		lons = read.hres_metadata(var0)['lons'].values
		npoints = len(lats)
		err = .0001
		for ipoint in range(npoints):
			lat, lon = lats[ipoint], lons[ipoint]
			i = int(np.where((abs(coords[:, 2] - lat) < err) * (abs(coords[:, 3] - lon) < err))[0])
			rlat = round(coords[i, 4], 2)
			rlon = round(coords[i, 5], 2)
			print(ipoint, lat, lon, rlat, rlon, i)
			rlats.append(rlat)
			rlons.append(rlon)
		rlats = np.array(rlats)
		rlons = np.array(rlons)
		np.save('../input_data/hres/rlats', rlats)
		np.save('../input_data/hres/rlons', rlons)


	irlats = (rlats + 6.45) / 0.05
	irlons = (rlons + 5) / 0.05
	irlats = (np.rint(irlats)).astype(int)
	irlons = (np.rint(irlons)).astype(int)

	aux = dataVar[:, 0, :, :].T
	data = data.T
	for i in range(npoints):  # When doing it without for loop memory cannot handle it
		# if i % 1000 == 0:
		# 	print(i, round(100*i/npoints, 2), '%')
		aux[irlons[i], irlats[i]] = data[i]
	dataVar[:] = aux.T[:, np.newaxis, :, :]

	# print(nc)
	# print('------------------------------------------------------------')
	# for var in nc.variables:
	# 	print('------------------------------------------------------------')
	# 	print(var)
	# 	print(nc.variables[var])
	# 	print(nc.variables[var][:].shape)
	# 	print(np.min(nc.variables[var][:]), np.max(nc.variables[var][:]))
	# print(np.nanmin(nc.variables[var][:]), np.nanmax(nc.variables[var][:]))
	# print(nc.variables[var][:])
	# exit()

	# Write to file
	nc.close()
