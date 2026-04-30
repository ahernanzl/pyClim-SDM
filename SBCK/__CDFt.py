# -*- coding: utf-8 -*-

## Copyright(c) 2021 / 2025 Yoann Robin
## 
## This file is part of SBCK.
## 
## SBCK is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## SBCK is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with SBCK.  If not, see <https://www.gnu.org/licenses/>.

##################################################################################
##################################################################################
##                                                                              ##
## Original author  : Mathieu Vrac                                              ##
## Contact          : mathieu.vrac@lsce.ipsl.fr                                 ##
##                                                                              ##
## Notes   : CDFt is the re-implementation of the function CDFt of R package    ##
##           "CDFt" developped by Mathieu Vrac, available at                    ##
##           https://cran.r-project.org/web/packages/CDFt/index.html            ##
##           This code is governed by the GNU-GPL3 license with the             ##
##           authorization of Mathieu Vrac                                      ##
##                                                                              ##
##################################################################################
##################################################################################


###############
## Libraries ##
###############

import numpy       as np
import scipy.stats as sc
import scipy.interpolate as sci

from .tools.__Dist import _Dist


###########
## Class ##
###########

class CDFt:
	"""
	SBCK.CDFt
	=========
	
	Description
	-----------
	Quantile Mapping bias corrector, taking account of an evolution of the
	distribution, see [1].
	
	References
	----------
	[1] Michelangeli, P.-A., Vrac, M., and Loukos, H.: Probabilistic downscaling
	approaches: Application to wind cumulative distribution functions, Geophys.
	Res. Lett., 36, L11708, https://doi.org/10.1029/2009GL038401, 2009.
	
	Notes
	-----
	CDFt is the re-implementation of the function CDFt of R package "CDFt"
	developped by Mathieu Vrac, available at
	https://cran.r-project.org/web/packages/CDFt/index.htmm
	"""
	def __init__( self , **kwargs ):##{{{
		"""
		Initialisation of CDFt bias corrector. All arguments must be named.
		
		Parameters
		----------
		distY0 : A statistical distribution from scipy.stats or SBCK.tools.rv_*
			The distribution of references in calibration period. Default is rv_histogram.
		distX0 : A statistical distribution from scipy.stats or SBCK.tools.rv_*
			The distribution of biased dataset in calibration period. Default is rv_histogram.
		distY1 : A statistical distribution from scipy.stats or SBCK.tools.rv_*
			The distribution of references in projection period. Default is rv_histogram, and Y1 is estimated during fit
		distX1 : A statistical distribution from scipy.stats or SBCK.tools.rv_*
			The distribution of biased dataset in projection period. Default is rv_histogram.
		kwargsY0 : dict
			Arguments passed to distY0
		kwargsX0 : dict
			Arguments passed to distX0
		kwargsY1 : dict
			Arguments passed to distY1
		kwargsX1 : dict
			Arguments passed to distX1
		n_features: None or integer
			Numbers of features, optional because it is determined during fit
			if X0 and Y0 are not None.
		tol : float
			Numerical tolerance, default 1e-6
		version: int, optional
			...
		normalize_cdf: bool or list of bool
			If a normalization is applied to the data to maximize the overlap
			of the support. Can be a bool (True or False, applied for all
			colums), or a list of bool of size 'n_features' to distinguished
			each columns.
		
		"""
		self.n_features  = kwargs.get("n_features")
		self._tol        = kwargs.get( "tol"       , 1e-6  )
		self._dsupp      = kwargs.get( "dsupp"     , 1000  )
		self._samples_Y1 = kwargs.get("samples_Y1" , 10000 )
		self._version    = kwargs.get("version"    , 2     )
		self._v3_e       = kwargs.get("v3_e"       , "auto"  )
		
		self._distY0 = _Dist( dist = kwargs.get("distY0") , kwargs = kwargs.get("kwargsY0") )
		self._distY1 = _Dist( dist = kwargs.get("distY1") , kwargs = kwargs.get("kwargsY1") )
		self._distX0 = _Dist( dist = kwargs.get("distX0") , kwargs = kwargs.get("kwargsX0") )
		self._distX1 = _Dist( dist = kwargs.get("distX1") , kwargs = kwargs.get("kwargsX1") )
		self._normalize_cdf    = kwargs.get("normalize_cdf")
		if ~(type(self._normalize_cdf) in [bool,list]):
			self._normalize_cdf = True
		self._p_left  = 0
		self._p_right = 1
	##}}}
	
	def fit( self , Y0 , X0 , X1 ):##{{{
		"""
		Fit of CDFt model
		
		Parameters
		----------
		Y0	: np.array[ shape = (n_samples,n_features) ]
			Reference dataset during calibration period
		X0	: np.array[ shape = (n_samples,n_features) ]
			Biased dataset during calibration period
		X1	: np.array[ shape = (n_samples,n_features) ]
			Biased dataset during projection period
		
		Note
		----
		The fit is performed margins by margins (without taking into account the dependance structure, see R2D2 or dOTC)
		"""
		
		## Reshape data in matrix form
		if Y0 is not None and Y0.ndim == 1 : Y0 = Y0.reshape(-1,1)
		if X0 is not None and X0.ndim == 1 : X0 = X0.reshape(-1,1)
		if X1 is not None and X1.ndim == 1 : X1 = X1.reshape(-1,1)
		
		## Find n_features
		if self.n_features is None:
			if Y0 is None and X0 is None and X1 is None:
				print( "n_features must be set during initialization if Y0 = X0 = X1 = None" )
			elif Y0 is not None: self.n_features = Y0.shape[1]
			elif X0 is not None: self.n_features = X0.shape[1]
			else:                self.n_features = X1.shape[1]
		
		## Set normalizations
		if type(self._normalize_cdf) == bool:
			self._normalize_cdf = [self._normalize_cdf for _ in range(self.n_features)]
		
		## Find laws
		self._distY0.set_features(self.n_features)
		self._distY1.set_features(self.n_features)
		self._distX0.set_features(self.n_features)
		self._distX1.set_features(self.n_features)
		
		## Start fit itself
		for i in range(self.n_features):
			self._distY0.fit( Y0[:,i] , i )
			self._distX0.fit( X0[:,i] , i )
			self._distX1.fit( X1[:,i] , i )
			## Fit Y1
			if self._distY1.is_frozen(i):
				self._distY1.law.append(self._dist.distY1[i])
			else:
				if self._distY0.is_parametric(i) and self._distX0.is_parametric(i) and self._distX1.is_parametric(i):
					Y1 = self._distX1.law[i].ppf( self._distX0.law[i].cdf( self._distY0.law[i].ppf( self._distX1.law[i].cdf(X1[:,i].squeeze()) ) ) )
				else:
					Y0uni = Y0[:,i] if Y0 is not None else self._distY0.law[-1].rvs(10000)
					X0uni = X0[:,i] if X0 is not None else self._distX0.law[-1].rvs(10000)
					X1uni = X1[:,i] if X1 is not None else self._distX1.law[-1].rvs(10000)
					Y1 = self._infer_Y1( Y0uni , X0uni , X1uni , i )
				self._distY1.fit( Y1 , i )
	##}}}
	
	def predict( self , X1 , X0 = None ):##{{{
		"""
		Perform the bias correction
		Return Z1 if X0 is None, else return a tuple Z1,Z0
		
		Parameters
		----------
		X1 : np.array[ shape = (n_sample,n_features) ]
			Array of value to be corrected in projection period
		X0 : np.array[ shape = (n_sample,n_features) ] or None
			Array of value to be corrected in calibration period, optional
		
		Returns
		-------
		Z1 : np.array[ shape = (n_sample,n_features) ]
			Return an array of correction in projection period
		Z0 : np.array[ shape = (n_sample,n_features) ] or None
			Return an array of correction in calibration period
		
		Note
		----
		The correction is performed margins by margins (without taking into account the dependance structure, see R2D2 or dOTC)
		"""
		if X1.ndim == 1 : X1 = X1.reshape(-1,1)
		Z1 = np.zeros_like(X1)
		for i in range(self.n_features):
			cdf = self._distX1.law[i].cdf(X1[:,i])
			cdf[np.logical_not(cdf < 1)] = 1 - self._tol
			cdf[np.logical_not(cdf > 0)] = self._tol
			Z1[:,i] = self._distY1.law[i].ppf( cdf )
		
		if X0 is not None:
			if X0.ndim == 1 : X0 = X0.reshape(-1,1)
			Z0 = np.zeros_like(X0)
			for i in range(self.n_features):
				cdf = self._distX0.law[i].cdf(X0[:,i])
				cdf[np.logical_not(cdf < 1)] = 1 - self._tol
				cdf[np.logical_not(cdf > 0)] = self._tol
				Z0[:,i] = self._distY0.law[i].ppf( cdf )
			return Z1,Z0
		return Z1
	##}}}
	
	def _CDFt_V1( self , Y0 , X0 , X1 , idist ):##{{{
		
		## CDF
		rvY0 = self._distY0.law[idist]
		rvX0 = self._distX0.dist[idist]( *self._distX0.dist[idist].fit( X0.squeeze()) , **self._distX0.kwargs )
		rvX1 = self._distX1.dist[idist]( *self._distX1.dist[idist].fit( X1.squeeze()) , **self._distX1.kwargs )
		
		hY1 = rvX1.ppf( rvX0.cdf( rvY0.ppf( rvX1.cdf( X1 ) ) ) )
		
		return hY1
	##}}}
	
	def _CDFt_V2( self , Y0 , X0 , X1 , idist ):##{{{
		
		dsupp = self._dsupp
		
		## Normalization
		if self._normalize_cdf[idist]:
			mY0 = np.mean(Y0)
			mX0 = np.mean(X0)
			mX1 = np.mean(X1)
			sY0 = np.std(Y0)
			sX0 = np.std(X0)
			
			X0s = (X0 - mX0) * sY0 / sX0 + mY0
			X1s = (X1 - mX1) * sY0 / sX0 + mX1 + mY0 - mX0
		
		## CDF
		rvY0  = self._distY0.law[idist]
		rvX0s = self._distX0.dist[idist]( *self._distX0.dist[idist].fit( X0s.squeeze()) , **self._distX0.kwargs )
		rvX1s = self._distX1.dist[idist]( *self._distX1.dist[idist].fit( X1s.squeeze()) , **self._distX1.kwargs )
		
		## Support
		## Here the support is such that the CDF of Y0, X0s and X1s start from 0
		## and go to 1
		x_min = min([T.min() for T in [Y0,X0s,X1s,X0,X1]])
		x_max = max([T.max() for T in [Y0,X0s,X1s,X0,X1]])
		x_eps = 0.05 * (x_max - x_min)
		x_fac = 1
		x = np.linspace( x_min - x_fac * x_eps , x_max + x_fac * x_eps , dsupp )
		
		def support_test( rv , x ):
			if not abs(rv.cdf(x[0])) < self._tol:
				return False
			if not abs(rv.cdf(x[-1])-1) < self._tol:
				return False
			return True
		
		while (not support_test(rvY0,x)) or (not support_test(rvX0s,x)) or (not support_test(rvX1s,x)):
			x_fac *= 2
			x = np.linspace( x_min - x_fac * x_eps , x_max + x_fac * x_eps , dsupp )
		x_fac /= 2
		
		## Loop to check the support
		extend_support = True
		p_min = 0
		p_max = 1
		while extend_support:
			extend_support = False
			
			## Inference of the CDF of Y1
			cdfY1 = rvY0.cdf(rvX0s.ppf(rvX1s.cdf(x)))
			
			## Correction of the CDF, we want that the CDF of Y1 start from 0 and goto 1
			if cdfY1[0] > p_min:
				## CDF not start at 0
				idx  = np.max(np.argwhere(np.abs(cdfY1[0] - cdfY1) < self._tol))
				if idx == 0:
					extend_support = True
				else:
					supp_l_X0s = rvX0s.ppf(cdfY1[0]) - rvX0s.ppf(p_min)
					supp_l_X1s = rvX1s.ppf(cdfY1[0]) - rvX1s.ppf(p_min)
					supp_l_Y0  = rvY0.ppf(cdfY1[0])  - rvY0.ppf(p_min)
					supp_l_Y1  = supp_l_Y0
					if x[idx] - supp_l_Y1 < x[0]:
						extend_support = True
					else:
						idxl = np.argmin(np.abs(x - (x[idx] - supp_l_Y1)))
						cdfY1[:idxl] = 0
						cdfY1[idxl:idx] = rvY0.cdf( np.linspace( rvY0.ppf(p_min) , rvY0.ppf(cdfY1[idx]) , idx - idxl ) )
			
			if cdfY1[-1] < p_max:
				## CDF not finished at 1
				idx = np.min(np.argwhere(np.abs(cdfY1[-1] - cdfY1) < self._tol))
				if idx == dsupp -1:
					extend_support = True
				else:
					supp_r_Y0  = rvY0.ppf(p_max)  - rvY0.ppf(cdfY1[-1])
					supp_r_X0s = rvX0s.ppf(p_max) - rvX0s.ppf(cdfY1[-1]) 
					supp_r_X1s = rvX1s.ppf(p_max) - rvX1s.ppf(cdfY1[-1]) 
					supp_r_Y1  = supp_r_Y0
					if x[idx] + supp_r_Y1 > x[-1]:
						extend_support = True
					else:
						idxr = np.argmin(np.abs(x - (x[idx] + supp_r_Y1)))
						cdfY1[idxr:] = 1
						cdfY1[idx:idxr] = rvY0.cdf( np.linspace( rvY0.ppf(cdfY1[idx]) , rvY0.ppf(p_max) , idxr - idx ) )
			
			## Support
			if extend_support:
				dsupp  = int(dsupp*1.2)
				x_fac *= 2
				x      = np.linspace( x_min - x_fac * x_eps , x_max + x_fac * x_eps , dsupp )
		
		## Cut the support to remove identical values
		try:
			idxl  = np.max( np.argwhere( np.abs( cdfY1 - cdfY1[0] ) < self._tol ) )
			x     = x[idxl:]
			cdfY1 = cdfY1[idxl:]
		except:
			pass
		try:
			idxr  = np.min( np.argwhere( np.abs( cdfY1 - cdfY1[-1] ) < self._tol ) ) + 1
			x     = x[:idxr]
			cdfY1 = cdfY1[:idxr]
		except:
			pass
		
		## Inverse of the CDF
		icdfY1 = sci.interp1d( cdfY1 , x , fill_value = (x[0],x[-1]) , bounds_error = False )
		
#		## Now find cut
#		lsuppl_Y0  = np.median(Y0) - np.quantile(Y0,self._p_left)
#		lsuppl_X0  = np.median(X0) - np.quantile(X0,self._p_left)
#		lsuppl_X1  = np.median(X1) - np.quantile(X1,self._p_left)
#		lsuppl_Y1  = lsuppl_Y0 * lsuppl_X1 / lsuppl_X0
#		lsuppl_pY1 = icdfY1(0.5) - icdfY1(self._p_left)
#		lsuppr_Y0  = np.quantile(Y0,self._p_right) - np.median(Y0)
#		lsuppr_X0  = np.quantile(X0,self._p_right) - np.median(X0)
#		lsuppr_X1  = np.quantile(X1,self._p_right) - np.median(X1)
#		lsuppr_Y1  = lsuppr_Y0 * lsuppr_X1 / lsuppr_X0
#		lsuppr_pY1 = icdfY1(self._p_right) - icdfY1(0.5)
#		
#		if lsuppl_pY1 > lsuppl_Y1 or lsuppr_pY1 > lsuppr_Y1:
#			
#			## Find p_min
#			p_min = 0
#			if lsuppl_pY1 > lsuppl_Y1:
#				pl  = np.linspace( 0 , 0.5 , 10000 )
#				ql  = icdfY1(pl)
#				ql  = ql[-1] - ql
#				idxl = np.argmin( np.abs( ql - lsuppl_Y1 ) )
#				p_min = pl[idxl]
#			
#			## Find p_max
#			p_max = 1
#			if lsuppr_pY1 > lsuppr_Y1:
#				pr  = np.linspace( 0.5 , 1 , 10000 )
#				qr  = icdfY1(pr)
#				qr  = qr - qr[0]
#				idxr = np.argmin( np.abs( qr - lsuppr_Y1 ) )
#				p_max = pr[idxr]
#			
#			## Final: Replace by 0 / 1 bellow / behind p_min / p_max
#			cdfY1[cdfY1 < p_min] = 0
#			cdfY1[cdfY1 > p_max] = 1
#			
#			## Cut values and new icdf
#			try:
#				idxl  = np.max(np.argwhere(cdfY1 < p_min))
#				x     = x[idxl:]
#				cdfY1 = cdfY1[idxl:]
#			except:
#				pass
#			try:
#				idxr  = np.min(np.argwhere(cdfY1 > p_max)) + 1
#				x     = x[:idxr]
#				cdfY1 = cdfY1[:idxr]
#			except:
#				pass
#			icdfY1 = sci.interp1d( cdfY1 , x , fill_value = (x[0],x[-1]) , bounds_error = False )
#		print(cdfY1[0])
#		print(cdfY1[-1])
		
		## Draw hY1
		rvX1 = self._distX1.dist[idist]( *self._distX1.dist[idist].fit( X1.squeeze()) , **self._distX1.kwargs )
		hY1  = icdfY1( rvX1.cdf(X1) )
		
		return hY1
	##}}}
	
	def _CDFt_V3( self , Y0 , X0 , X1 , idist ):##{{{
		
		if (X0.min() <= Y0.min()) and (X0.max() >= Y0.max()):
			return self._CDFt_V1( Y0 , X0 , X1 , idist )
		
		if not type(self._v3_e) is float:
			self._v3_e = 5 * max( 1 / Y0.size , 1 / X0.size , 1 / X1.size )
		e   = self._v3_e
		
		lX0 = np.quantile( X0 , e )
		lY0 = np.quantile( Y0 , e )
		lX1 = np.quantile( X1 , e )
		
		uX0 = np.quantile( X0 , 1 - e )
		uY0 = np.quantile( Y0 , 1 - e )
		uX1 = np.quantile( X1 , 1 - e )
		
		X0s = ( X0 - lX0 ) / ( uX0 - lX0 ) * ( uY0 - lY0 ) + lY0 
		X1s = ( X1 - lX1 ) / ( uX0 - lX0 ) * ( uY0 - lY0 ) + lY0 + lX1 - lX0
		
		rvY0  = self._distY0.law[idist]
		rvX0s = self._distX0.dist[idist]( *self._distX0.dist[idist].fit( X0s.squeeze()) , **self._distX0.kwargs )
		rvX1s = self._distX1.dist[idist]( *self._distX1.dist[idist].fit( X1s.squeeze()) , **self._distX1.kwargs )
		
		hY1 = rvX1s.ppf( rvX0s.cdf( rvY0.ppf( rvX1s.cdf( X1s ) ) ) )
		
		return hY1
	##}}}
	
	def _infer_Y1( self , Y0 , X0 , X1 , idist ):##{{{
		
		if self._version == 1:
			return self._CDFt_V1( Y0 , X0 , X1 , idist )
		elif self._version == 2:
			return self._CDFt_V2( Y0 , X0 , X1 , idist )
		else:
			return self._CDFt_V3( Y0 , X0 , X1 , idist )
	##}}}
	

