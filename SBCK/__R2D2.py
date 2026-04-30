# -*- coding: utf-8 -*-

## Copyright(c) 2021 Yoann Robin
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
## Original authors : Mathieu Vrac and Soulivanh Thao                           ##
## Contact          : mathieu.vrac@lsce.ipsl.fr                                 ##
## Contact          : soulivanh.thao@lsce.ipsl.fr                               ##
##                                                                              ##
## Notes   : R2D2 is the re-implementation of the function R2D2 of R package    ##
##           "R2D2" developped by Mathieu Vrac and Soulivanh Thao, available at ##
##                                                                              ##
##           This code is governed by the GNU-GPL3 license with the             ##
##           authorization of Mathieu Vrac                                      ##
##                                                                              ##
##################################################################################
##################################################################################

###############
## Libraries ##
###############

import numpy as np
from .__CDFt          import CDFt
from .tools.__shuffle import SchaakeShuffleRef


###########
## Class ##
###########

class R2D2(CDFt):
	"""
	SBCK.R2D2
	=========
	
	Description
	-----------
	Non stationnary Quantile Mapping bias corrector with multivariate rankshuffle, as described in [1]
	
	References
	----------
	[1] Vrac, M.: Multivariate bias adjustment of high-dimensional climate simulations: the Rank Resampling for Distributions and Dependences (R2 D2 ) bias correction, Hydrol. Earth Syst. Sci., 22, 3175–3196, https://doi.org/10.5194/hess-22-3175-2018, 2018.
	"""
	
	
	def __init__( self , refs = [0] , **kwargs ):##{{{
		"""
		Initialisation of R2D2.
		
		Parameters
		----------
		refs     : list
			Index of reference for SchaakeShuffleRef, see SBCK.tools.SchaakeShuffleRef
		**kwargs : see SBCK.CDFt
			All others arguments are passed to SBCK.CDFt class.
		"""
		CDFt.__init__( self , **kwargs )
		self._refs = refs
		self._ssr  = SchaakeShuffleRef( refs[0] )
	##}}}
	
	def fit( self , Y0 , X0 , X1 ):##{{{
		"""
		Fit of the R2D2 model
		
		Parameters
		----------
		Y0	: np.array[ shape = (n_samples,n_features) ]
			Reference dataset during calibration period
		X0	: np.array[ shape = (n_samples,n_features) ]
			Biased dataset during calibration period
		X1	: np.array[ shape = (n_samples,n_features) ]
			Biased dataset during projection period
		"""
		CDFt.fit( self , Y0 , X0 , X1 )
		self._ssr.fit(Y0)
	##}}}
	
	def predict( self , X1 , X0 = None ):##{{{
		"""
		Perform the bias correction
		Return Z1 if X0 is None, else return a tuple Z1,Z0
		
		Parameters
		----------
		X1  : np.array[ shape = (n_samples,n_features) ]
			Array of value to be corrected in projection period
		X0  : np.array[ shape = (n_samples,n_features) ] or None
			Array of value to be corrected in calibration period
		
		Returns
		-------
		Z1 : np.array[ shape = (n_sample,n_features) ]
			Return an array of correction in projection period
		Z0 : np.array[ shape = (n_sample,n_features) ] or None
			Return an array of correction in calibration period
		"""
		Zu = CDFt.predict( self , X1 , X0 )
		
		if X0 is not None:
			Z1u,Z0u = Zu
			Z1 = np.zeros( Z1u.shape + (len(self._refs),) )
			Z0 = np.zeros( Z0u.shape + (len(self._refs),) )
			for i,r in enumerate(self._refs):
				self._ssr._ref = r
				Z1[:,:,i] = self._ssr.predict(Z1u)
				Z0[:,:,i] = self._ssr.predict(Z0u)
			return Z1.squeeze(),Z0.squeeze()
		
		Z1u = Zu
		Z1  = np.zeros( Z1u.shape + (len(self._refs),) )
		for i,r in enumerate(self._refs):
			self._ssr._ref = r
			Z1[:,:,i] = self._ssr.predict(Z1u)
		return Z1.squeeze()
	##}}}

