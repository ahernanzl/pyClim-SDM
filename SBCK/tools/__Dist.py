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

###############
## Libraries ##
###############

import numpy as np
import scipy.stats as sc
from .__rv_extend import rv_histogram


#############
## Classes ##
#############

class _Dist:
	def __init__( self , dist , kwargs ):
		self.dist   = dist   if dist is not None else rv_histogram
		self.kwargs = kwargs if kwargs is not None else {}
		self.law    = []
	
	def set_features( self , n_features ):
		if type(self.dist) is not list:
			self.dist = [ self.dist for _ in range(n_features) ]
	
	def is_frozen( self , i ):
		return isinstance(self.dist[i],sc._distn_infrastructure.rv_frozen)
	
	def is_parametric( self , i ):
		ispar = self.is_frozen(i)
		if len(self.law) >= i:
			ispar = ispar or isinstance(self.law[i],sc._distn_infrastructure.rv_frozen)
		return ispar
	
	def fit( self , X , i ):
		if self.is_frozen(i):
			self.law.append(self.dist[i])
		else:
			self.law.append( self.dist[i]( *self.dist[i].fit( X.squeeze() ) , **self.kwargs ) )
		 
