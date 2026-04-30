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
## Notes   : SchaakeShuffleRef is the re-implementation of the rank shuffle     ##
##           funtions of R package "R2D2" developped by Mathieu Vrac and        ##
##           Soulivanh Thao, available at                                       ##
##                                                                              ##
##           This code is governed by the GNU-GPL3 license with the             ##
##           authorization of Mathieu Vrac                                      ##
##                                                                              ##
##################################################################################
##################################################################################

###############
## Libraries ##
###############

import numpy        as np
import scipy.stats  as sc
import scipy.linalg as scl
import scipy.spatial.distance as ssd
from .__rv_extend import mrv_histogram

#############
## Classes ##
#############

class SchaakeShuffle:##{{{
	"""
	SBCK.tools.SchaakeShuffle
	=========================
	Match the rank structure of X with them of Y by reordering X. Work in multivariate case, but rank  of each 
	features are reordered independantly.
	"""
	
	def __init__( self , Y0 = None ):##{{{
		"""
		Initialization
		
		Parameters
		----------
		Y0 : np.array[n_samples,n_features] or None
			The target dataset, use fit function if Y0 is None
		"""
		self._Y0 = None
		if Y0 is not None: self.fit(Y0)
	##}}}
	
	def fit( self , Y0 ):##{{{
		"""
		Definine the reference ranks structure
		
		Parameters
		----------
		Y0 : np.array[n_samples,n_features]
			The target dataset
		"""
		self._Y0 = Y0
		if self._Y0.ndim == 1: self._Y0 = self._Y0.reshape(-1,1)
	##}}}
	
	def _predict( self , Y0 , X0 ):##{{{
		X0 = X0.squeeze()
		Y0 = Y0.squeeze()
		
		rank_X0 = sc.rankdata( X0 , method = "ordinal" )
		rank_Y0 = sc.rankdata( Y0 , method = "ordinal" )
		
		arank_X0 = np.argsort(rank_X0)
		Z0 = X0[arank_X0][rank_Y0-1]
		
		return Z0
	##}}}
	
	def predict( self , X0 ):##{{{
		"""
		Apply the rank structure to X0
		
		Parameters
		----------
		X0 : np.array[n_samples,n_features]
			The dataset to reorder
		
		Returns
		-------
		Z0 : np.array[n_samples,n_features]
			Reordered dataset
		"""
		if X0.ndim == 1: X0 = X0.reshape(-1,1)
		
		## If n_samples of X/Y differs, we complet the sequence by drawing uniformly in X/Y to have the same shape.
		if self._Y0.shape[0] < X0.shape[0]:
				YY = np.zeros_like(X0)
				YY[:self._Y0.shape[0],:] = self._Y0
				YY[self._Y0.shape[0]:,:] = self._Y0[np.random.choice( self._Y0.shape[0] , X0.shape[0] - self._Y0.shape[0] , replace = True ),:]
				XX = X0
		elif X0.shape[0] < self._Y0.shape[0]:
			XX = np.zeros_like(self._Y0)
			XX[:X0.shape[0],:] = X0
			XX[X0.shape[0]:,:] = X0[np.random.choice( X0.shape[0] , self._Y0.shape[0] - X0.shape[0] , replace = True ),:]
			YY = self._Y0
		else:
			XX,YY = X0,self._Y0
		
		n_features = X0.shape[1]
		ZZ = np.zeros_like(XX)
		for i in range(n_features):
			ZZ[:,i] = self._predict( YY[:,i] , XX[:,i] )
		
		Z0 = ZZ[:X0.shape[0],:]
		
		return Z0
	##}}}
##}}}

def schaake_shuffle( Y0 , X0 ):##{{{
	"""
	SBCK.tools.schaake_shuffle
	==========================
	Match the rank structure of X0 with them of Y0 by reordering X0. Work in multivariate case, but ranks of each 
	features are reordered independantly.
	
	Note: This function just call the class SBCK.tools.SchaakeShuffle
	
	Parameters
	----------
	X0 : np.array[n_samples,n_features]
		The dataset to reorder
	Y0 : np.array[n_samples,n_features]
		The target dataset
	
	Returns
	-------
	Z0 : np.array[n_samples,n_features]
		Reordered dataset
	
	"""
	ss = SchaakeShuffle(Y0)
	return ss.predict(X0)
##}}}

class SchaakeShuffleRef(SchaakeShuffle):##{{{
	"""
	SBCK.tools.SchaakeShuffleRef
	============================
	Match the rank structure of X with them of Y by reordering X, but fix one features to keep the structure of X.
	"""
	
	def __init__( self , ref , Y0 = None ):##{{{
		"""
		Initialization
		
		Parameters
		----------
		ref : int
			features kept.
		Y0 : np.array[n_samples,n_features] or None
			The target dataset, use fit function if Y0 is None
		"""
		self._ref = ref
		SchaakeShuffle.__init__( self , Y0 )
	##}}}
	
	def fit( self , Y0 ):##{{{
		"""
		Definine the reference ranks structure
		
		Parameters
		----------
		Y0 : np.array[n_samples,n_features]
			The target dataset
		"""
		SchaakeShuffle.fit( self , Y0 )
	##}}}
	
	def predict( self , X0 ):##{{{
		"""
		Apply the rank structure to X0
		
		Parameters
		----------
		X0 : np.array[n_samples,n_features]
			The dataset to reorder
		
		Returns
		-------
		Z0 : np.array[n_samples,n_features]
			Reordered dataset
		"""
		if X0.ndim == 1: X0 = X0.reshape(-1,1)
		Z0 = SchaakeShuffle.predict( self , X0 )
		
		rank_ref_X0  = sc.rankdata( X0[:,self._ref] , method = "ordinal" )
		rank_ref_Z0  = sc.rankdata( Z0[:,self._ref] , method = "ordinal" )
		arank_ref_Z0 = np.argsort(rank_ref_Z0)
		Z0 = Z0[arank_ref_Z0,:][rank_ref_X0-1,:]
		
		return Z0
	##}}}
##}}}

class MVQuantilesShuffle: ##{{{
	"""
	SBCK.tools.MVQuantilesShuffle
	=============================
	Multivariate Schaake shuffle using the quantiles.
	Used to reproduce the dependence structure of a dataset to another dataset
	"""
	
	def __init__( self , col_cond = [1] , lag_search = 1 , lag_keep = 1 ): ##{{{
		"""
		Initialization
		
		Parameters
		----------
		col_cond : list[int]
			Conditioning columns
		lag_search: int
			Number of lags to transform the dependence structure
		lag_keep: int
			Number of lags to keep
		"""
		self.col_cond   = col_cond
		self.lag_search = lag_search
		self.lag_keep   = lag_keep
		self._w         = 1
		
	##}}}
	
	def fit( self , Y ):##{{{
		"""
		Fit the reference structure
		
		Parameters
		----------
		Y : np.array[n_samples,n_features]
			The target dataset
		"""
		
		## Parameters
		n_samplesY = Y.shape[0]
		self.n_features = Y.shape[1]
		self.col_ucond  = [ i for i in range(self.n_features) if i not in self.col_cond ]
		
		## Build non-parametric marginal distribution of Y
		rvY = mrv_histogram().fit(Y)
		
		## Index to build block search matrix
		tiY = (n_samplesY - 1 - scl.toeplitz(range(n_samplesY)))[::-1,:][:self.lag_search,:(n_samplesY-self.lag_search+1)]
		
		## Find quantiles (i.e. ranks)
		self.qY = rvY.cdf(Y)
		
		## Build conditionning block search
		qYc = self.qY[:,self.col_cond]
		self.bsYc = np.array( [ qYc[tiY[:,i],:].ravel() for i in range(n_samplesY-self.lag_search+1) ] )
		
	##}}}
	
	def transform( self , X ): ##{{{
		"""
		Apply the quantiles structure to X
		
		Parameters
		----------
		X : np.array[n_samples,n_features]
			The dataset to reorder
		
		Returns
		-------
		Z : np.array[n_samples,n_features]
			Reordered dataset
		"""
		
		## Parameters
		n_samplesX = X.shape[0]
		
		## Build non-parametric marginal distribution of X
		rvX  = mrv_histogram().fit(X)
		
		## Index to build block search matrix
		tiX  = (n_samplesX - 1 - scl.toeplitz(range(n_samplesX)))[::-1,:][:self.lag_search,:(n_samplesX-self.lag_search+1)]
		
		## Find quantiles (i.e. ranks)
		qX = rvX.cdf(X)
		
		## Build conditionning block search
		## NOTE: in bsXc, the tiX[:,-1] column is added, otherwise the last values
		## are missing
		qXc = qX[:,self.col_cond]
		bsXc = np.array( [ qXc[tiX[:,i],:].ravel() for i in range(0,n_samplesX-self.lag_search+1,self.lag_keep) ] + [qXc[tiX[:,-1],:].ravel()] )
		
		## Now pairwise dist between cond. X / Y block search
		bsdistc = ssd.cdist( self._w * bsXc , self._w * self.bsYc )
		idx_bsc = np.argmin( bsdistc , axis = 1 )
		
		## Find associated quantiles in unconditioning Y
		## NOTE: Here we split into lag_keep values, and some last missing values
		## lag_search - n_last is the numbers of last missing values.
		n_last = self.lag_search - (n_samplesX - (bsXc.shape[0] - 1) * self.lag_keep)
		
		## ===> Saved
#		qZuc = np.vstack( [ self.qY[:,self.col_ucond][i:(i+self.lag_keep),:] for i in idx_bsc[:-1] ] + [self.qY[:,self.col_ucond][(idx_bsc[-1]+n_last):(idx_bsc[-1]+self.lag_search),:]] )
#		
#		## Now build qZ
#		qZ_unordered = np.hstack( (qXc,qZuc) )
#		qZ = np.zeros_like( qZ_unordered )
#		qZ[:,self.col_cond + self.col_ucond] = qZ_unordered
		## <===
		
		## ===> New
		qZ = np.vstack( [ self.qY[i:(i+self.lag_keep),:] for i in idx_bsc[:-1] ] + [self.qY[(idx_bsc[-1]+n_last):(idx_bsc[-1]+self.lag_search),:]] )
		## <===
		
		## And finaly inverse quantiles
		Z = rvX.ppf(qZ)
		
		return Z
	##}}}
	
##}}}

class MVRanksShuffle: ##{{{
	"""
	SBCK.tools.MVRanksShuffle
	=============================
	Multivariate Schaake shuffle using the ranks.
	Used to reproduce the dependence structure of a dataset to another dataset
	"""
	
	def __init__( self , col_cond = [1] , lag_search = 1 , lag_keep = 1 ): ##{{{
		"""
		Initialization
		
		Parameters
		----------
		col_cond : list[int]
			Conditioning columns
		lag_search: int
			Number of lags to transform the dependence structure
		lag_keep: int
			Number of lags to keep
		"""
		self.col_cond   = col_cond
		self.lag_search = lag_search
		self.lag_keep   = lag_keep
	##}}}
	
	def fit( self , Y ):##{{{
		"""
		Fit the reference structure
		
		Parameters
		----------
		Y : np.array[n_samples,n_features]
			The target dataset
		"""
		
		## Parameters
		n_samplesY = Y.shape[0]
		self.n_features = Y.shape[1]
		self.col_ucond  = [ i for i in range(self.n_features) if i not in self.col_cond ]
		
		## Index to build block search matrix
		tiY = (n_samplesY - 1 - scl.toeplitz(range(n_samplesY)))[::-1,:][:self.lag_search,:(n_samplesY-self.lag_search+1)]
		
		## Find quantiles (i.e. ranks)
		self.qY = sc.rankdata( Y , axis = 0 , method = "ordinal" )
		
		## Build conditionning block search
		qYc = self.qY[:,self.col_cond]
		self.bsYc = np.array( [ qYc[tiY[:,i],:].ravel() for i in range(n_samplesY-self.lag_search+1) ] )
	##}}}
	
	def transform( self , X ): ##{{{
		"""
		Apply the ranks structure to X
		
		Parameters
		----------
		X : np.array[n_samples,n_features]
			The dataset to reorder
		
		Returns
		-------
		Z : np.array[n_samples,n_features]
			Reordered dataset
		"""
		
		## Parameters
		n_samplesX = X.shape[0]
		
		## Build non-parametric marginal distribution of X
		
		## Index to build block search matrix
		tiX  = (n_samplesX - 1 - scl.toeplitz(range(n_samplesX)))[::-1,:][:self.lag_search,:(n_samplesX-self.lag_search+1)]
		
		## Find quantiles (i.e. ranks)
		qX = sc.rankdata( X , axis = 0 , method = "ordinal" )
		
		## Shrink
		qY = self.qY
		if qY.shape[0] < qX.shape[0]:
			qX = np.round( qX * qY.shape[0] / qX.shape[0] , 0 ).astype(int)
		elif qX.shape[0] < qY.shape[0]:
			qY = np.round( qY * qX.shape[0] / qY.shape[0] , 0 ).astype(int)
		
		## Build conditionning block search
		## NOTE: in bsXc, the tiX[:,-1] column is added, otherwise the last values
		## are missing
		qXc = qX[:,self.col_cond]
		bsXc = np.array( [ qXc[tiX[:,i],:].ravel() for i in range(0,n_samplesX-self.lag_search+1,self.lag_keep) ] + [qXc[tiX[:,-1],:].ravel()] )
		
		## Now pairwise dist between cond. X / Y block search
		bsdistc = ssd.cdist( bsXc , self.bsYc )
		idx_bsc = np.argmin( bsdistc , axis = 1 )
		
		## Find associated quantiles in unconditioning Y
		## NOTE: Here we split into lag_keep values, and some last missing values
		## lag_search - n_last is the numbers of last missing values.
		n_last = self.lag_search - (n_samplesX - (bsXc.shape[0] - 1) * self.lag_keep)
		## ===> Saved
#		qZuc = np.vstack( [ self.qY[:,self.col_ucond][i:(i+self.lag_keep),:] for i in idx_bsc[:-1] ] + [self.qY[:,self.col_ucond][(idx_bsc[-1]+n_last):(idx_bsc[-1]+self.lag_search),:]] )
#		
#		## Now build qZ
#		qZ_unordered = np.hstack( (qXc,qZuc) )
#		qZ = np.zeros_like( qZ_unordered )
#		qZ[:,self.col_cond + self.col_ucond] = qZ_unordered
		## <===
		
		## ===> New
		qZ = np.vstack( [ self.qY[i:(i+self.lag_keep),:] for i in idx_bsc[:-1] ] + [self.qY[(idx_bsc[-1]+n_last):(idx_bsc[-1]+self.lag_search),:]] )
		## <===
		
		## And finaly inverse ranks
		Xs = np.sort( X , axis = 0 )
		Z  = np.array( [ Xs[qZ[:,i]-1,i] for i in range(Xs.shape[1]) ] ).T.copy()
		
		return Z
	##}}}
	
##}}}


