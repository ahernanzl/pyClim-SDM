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
import scipy.interpolate as sci


#############
## Classes ##
#############

class MonotoneInverse:##{{{
	
	def __init__( self , xminmax , yminmax , transform ):##{{{
		self.xmin  = xminmax[0]
		self.xmax  = xminmax[1]
		self.ymin  = yminmax[0]
		self.ymax  = yminmax[1]
		delta = 0.05 * (self.xmax - self.xmin)
		nstepmin,nstepmax = 0,0
		while transform(self.xmin) > self.ymin:
			self.xmin -= delta
			nstepmin += 1
		while transform(self.xmax) < self.ymax:
			self.xmax += delta
			nstepmax += 1
		self.nstep = 100 + max(nstepmin,nstepmax)
		x = np.linspace(self.xmin,self.xmax,self.nstep)
		y = transform(x)
		self._inverse = sci.interp1d( y , x )
	##}}}
	
	def __call__( self , y ):##{{{
		return self._inverse(y)
	##}}}

##}}}

#class rv_histogram(sc.rv_histogram):##{{{
#	"""
#	SBCK.tools.rv_histogram
#	=======================
#	Wrapper on scipy.stats.rv_histogram adding a fit method.
#	"""
#	def __init__( self , *args , **kwargs ):##{{{
#		sc.rv_histogram.__init__( self , *args , **kwargs )
#	##}}}
#	
#	def fit( X , bins = 100 ):##{{{
#		return (np.histogram( X , bins = bins ),)
#	##}}}
#	
###}}}

class rv_histogram:##{{{
	"""
	SBCK.tools.rv_histogram
	=======================
	Empirical histogram class. The difference with scipy.stats.rv_histogram
	is the way to infer the cdf and the icdf. Here:
	
	>>> X ## Input
	>>> Xs = np.sort(X)
	>>> Xr = sc.rankdata(Xs,method="max")
	>>> p  = np.unique(Xr) / X.size
	>>> q  = Xs[np.unique(Xr)-1]
	>>> p[0] = 0
	>>>
	>>> icdf = scipy.interpolate.interp1d( p , q )
	>>> cdf  = scipy.interpolate.interp1d( q , p )
	"""
	
	def __init__( self , cdf = None , icdf = None , pdf = None , *args , X = None , **kwargs ):
		self._cdf  = None
		self._icdf = None
		self._pdf  = None
		if cdf is not None and icdf is not None and pdf is not None:
			self._cdf  = cdf
			self._icdf = icdf
			self._pdf  = pdf
		elif X is not None:
			cdf,icdf,pdf = rv_histogram.fit(X)
			self._cdf  = cdf
			self._icdf = icdf
			self._pdf  = pdf
	
	def fit( X , *args , **kwargs ):
		
		Xs = np.sort(X.squeeze())
		Xr = sc.rankdata(Xs,method="max")
		p  = np.unique(Xr) / X.size
		q  = Xs[np.unique(Xr)-1]
		
		p[0] = 0
#		p  = np.hstack( (0,p) )
#		q  = np.hstack( (X.min(),q) )
#		if q[0] == q[1]:
#			eps  = np.sqrt(np.finfo(float).resolution)
#			q[1] = (1-eps) * q[0] + eps * q[2]
		
		icdf = sci.interp1d( p , q , bounds_error = False , fill_value = (q[0],q[-1]) )
		cdf  = sci.interp1d( q , p , bounds_error = False , fill_value = (0,1) )
		
		h,c = np.histogram( X , int(0.1*X.size) , density = True )
		c = (c[1:] + c[:-1]) / 2
		pdf = sci.interp1d( c , h , bounds_error = False , fill_value = (0,0) ) 
		
		return (cdf,icdf,pdf)
	
	def rvs( self , size ):
		return self._icdf( np.random.uniform( size = size ) )
	
	def cdf( self , q ):
		return self._cdf(q)
	
	def icdf( self , p ):
		return self._icdf(p)
	
	def sf( self , q ):
		return 1 - self._cdf(q)
	
	def isf( self , p ):
		return self._icdf(1-p)
	
	def ppf( self , p ):
		return self.icdf(p)
	
	def pdf( self , x ):
		return self._pdf(x)
##}}}

class rv_ratio_histogram(sc.rv_histogram):##{{{
	"""
	SBCK.tools.rv_ratio_histogram
	=============================
	Extension of SBCK.tools.rv_histogram taking into account of a "ratio" part, i.e., instead of fitting:
	P( X < x )
	We fit separatly the frequency of 0 and:
	P( X < x | X > 0 )
	"""
	def __init__( self , *args , **kwargs ):##{{{
		eargs = ()
		if len(args) > 0:
			eargs = (args[0],)
		sc.rv_histogram.__init__( self , *eargs , **kwargs )
		self.p0 = 0
		if len(args) > 1:
			self.p0 = args[1]
	##}}}
	
	def fit( X , bins = 100 ):##{{{
		Xp = X[X>0]
		p0 = np.sum(np.logical_not(X>0)) / X.size
		return (np.histogram( Xp , bins = bins ),p0)
	##}}}
	
	def cdf( self , x ):##{{{
		cdf = np.zeros_like(x)
		idxp = x > 0
		idx0 = np.logical_not(x>0)
		cdf[idxp] = (1-self.p0) * sc.rv_histogram.cdf( self , x[idxp] ) + self.p0
		cdf[idx0] = self.p0 / 2
		return cdf
	##}}}
	
	def ppf( self , p ):##{{{
		idxp = p > self.p0
		idx0 = np.logical_not(p > self.p0 )
		ppf = np.zeros_like(p)
		ppf[idxp] = sc.rv_histogram.ppf( self , (p[idxp] - self.p0) / (1-self.p0) )
		ppf[idx0] = 0
		return ppf
	##}}}
	
	def sf( self , x ):##{{{
		return 1 - self.cdf(x)
	##}}}
	
	def isf( self , p ):##{{{
		return self.ppf( 1 - p )
	##}}}

##}}}

class rv_density:##{{{
	
	def __init__( self , *args , **kwargs ):##{{{
		self._kernel = None
		if kwargs.get("X") is not None:
			X = kwargs.get("X")
			self._kernel = sc.gaussian_kde( X.squeeze() , bw_method = kwargs.get("bw_method") )
			self._init_icdf( [X.min(),X.max()] )
		elif len(args) > 0:
			self._kernel = args[0]
			self._init_icdf( [args[1],args[2]] )
	##}}}
	
	def rvs( self , size ):##{{{
		p = np.random.uniform( size = size )
		return self.icdf(p)
	##}}}
	
	def fit( X , bw_method = None ):##{{{
		kernel = sc.gaussian_kde( X , bw_method = bw_method )
		return (kernel,X.min(),X.max())
	##}}}
	
	def pdf( self , x ):##{{{
		return self._kernel.pdf(x)
	##}}}
	
	def cdf( self , x ):##{{{
		x = np.array([x]).squeeze().reshape(-1,1)
		cdf = np.apply_along_axis( lambda z: self._kernel.integrate_box_1d( -np.Inf , z ) , 1 , x )
		cdf[cdf < 0] = 0
		cdf[cdf > 1] = 1
		return cdf.squeeze()
	##}}}
	
	def sf( self , x ):##{{{
		return 1 - self.cdf(x)
	##}}}
	
	def ppf( self , q ):##{{{
		return self.icdf(q)
	##}}}
	
	def icdf( self , q ):##{{{
		return self._icdf_fct(q)
	##}}}
	
	def isf( self , q ):##{{{
		return self.icdf(1-q)
	##}}}
	
	def _init_icdf( self , xminmax ):##{{{
		self._icdf_fct = MonotoneInverse( xminmax , [0,1] , self.cdf )
	##}}}

##}}}

class rv_mixture:##{{{
	
	def __init__( self , l_dist , weights = None ):##{{{
		self._l_dist  = l_dist
		self._n_dist  = len(l_dist)
		self._weights = np.array([weights]).squeeze() if weights is not None else np.ones(self._n_dist)
		self._weights /= self._weights.sum()
		self._init_icdf()
	##}}}
	
	def rvs( self , size ):##{{{
		out = np.zeros(size)
		ib,ie = 0,int(self._weights[0]*size)
		for i in range(self._n_dist-1):
			out[ib:ie] = self._l_dist[i].rvs( size = ie - ib )
			next_size = int(self._weights[i+1]*size)
			ib,ie = ie,min(ie+next_size,size)
		out[ib:] = self._l_dist[-1].rvs( size = size - ib )
		
		return out[np.random.choice(size,size,replace = False)]
	##}}}
	
	def pdf( self , x ):##{{{
		x = np.array([x]).reshape(-1,1)
		dens = np.zeros_like(x)
		for i in range(self._n_dist):
			dens += self._l_dist[i].pdf(x) * self._weights[i]
		return dens
	##}}}
	
	def cdf( self , x ):##{{{
		x = np.array([x]).reshape(-1,1)
		cdf = np.zeros_like(x)
		for i in range(self._n_dist):
			cdf += self._l_dist[i].cdf(x) * self._weights[i]
		return cdf.squeeze()
	##}}}
	
	def sf( self , x ):##{{{
		return 1 - self.cdf(x)
	##}}}
	
	def ppf( self , q ):##{{{
		return self.icdf(q)
	##}}}
	
	def icdf( self , q ):##{{{
		q = np.array([q]).reshape(-1,1)
		return self._icdf_fct(q)
	##}}}
	
	def _init_icdf(self):##{{{
		rvs = self.rvs(10000)
		self._icdf_fct = MonotoneInverse( [rvs.min(),rvs.max()] , [0,1] , self.cdf )
	##}}}
	
	def isf( self , q ):##{{{
		return self.icdf(1-q)
	##}}}
	
##}}}

class mrv_histogram:##{{{
	"""
	SBCK.tools.mrv_histogram
	========================
	Multidimensional rv_histogram. Each margins is fitted separately.
	"""
	
	def __init__( self ):
		self.n_features = 0
		self._law = []
	
	def fit( self , X ):
		if X.ndim == 1:
			X = X.reshape(-1,1)
		self.n_features = X.shape[1]
		for i in range(self.n_features):
			self._law.append( rv_histogram( *rv_histogram.fit( X[:,i] ) ) )
		
		return self
	
	def rvs( self , size = 1 ):
		return np.array( [ self._law[i].rvs(size=size) for i in range(self.n_features) ] ).T.copy()
	
	def cdf( self , q ):
		q = q.reshape(-1,self.n_features)
		return np.array( [ self._law[i].cdf(q[:,i]) for i in range(self.n_features) ] ).T.copy()
	
	def sf( self , q ):
		q = q.reshape(-1,self.n_features)
		return np.array( [ self._law[i].sf(q[:,i]) for i in range(self.n_features) ] ).T.copy()
	
	def ppf( self , p ):
		p = p.reshape(-1,self.n_features)
		return np.array( [ self._law[i].ppf(p[:,i]) for i in range(self.n_features) ] ).T.copy()
	
	def icdf( self , p ):
		return self.ppf(p)
	
	def isf( self , p ):
		p = p.reshape(-1,self.n_features)
		return np.array( [ self._law[i].isf(p[:,i]) for i in range(self.n_features) ] ).T.copy()
	
##}}}

class rv_empirical_gpd(rv_histogram):##{{{
	"""
	SBCK.tools.rv_empirical_gpd
	===========================
	Empirical histogram class where tails are given by the fit of Generalized
	Pareto Distribution. In dev., so use with caution.
	
	"""
	
	def __init__( self , *args , X = None , p = 0.9 , **kwargs ):##{{{
		
		if len(args) in [5,6]:
			super().__init__( *args )
			self._gpd_l   = args[3]
			self._gpd_r   = args[4]
			if len(args) == 5:
				self._p = p
			else:
				self._p = args[5]
		elif X is not None:
			cdf,icdf,pdf,gpd_l,gpd_r,p = rv_empirical_gpd.fit( X , p = p )
			super().__init__( cdf , icdf , pdf )
			self._gpd_l   = gpd_l
			self._gpd_r   = gpd_r
			self._p       = p
	##}}}
	
	def _gpd(X):##{{{
		m     = np.mean(X)
		s     = np.std(X)
		scale = m * ( m**2 / s**2 + 1 ) / 2
		shape = 1 - scale / m
		return scale,shape
	##}}}
	
	def fit( X , *args , p = 0.9 , **kwargs ):##{{{
		
		## Empirical part
		cdf,icdf,pdf = rv_histogram.fit(X)
		
		## Location parameter
		loc_l = icdf(1-p)
		loc_r = icdf(p)
		
		## GPD left fit
		Xl = -(X[X<loc_l] - loc_l)
		sc_l,sh_l = rv_empirical_gpd._gpd(Xl)
		gpd_l = sc.genpareto( loc = - loc_l , scale = sc_l , c = sh_l )
		
		## GPD right fit
		Xr = X[X>loc_r] - loc_r
		sc_r,sh_r = rv_empirical_gpd._gpd(Xr)
		gpd_r = sc.genpareto( loc = loc_r , scale = sc_r , c = sh_r )
		
		return cdf,icdf,pdf,gpd_l,gpd_r,p
	##}}}
	
	def rvs( self , size = 1 ):##{{{
		return self.icdf( np.random.uniform( size = size ) )
	##}}}
	
	def cdf( self , q ):##{{{
		q = np.array([q]).reshape(-1)
		p = np.zeros_like(q) + np.nan
		
		## Empirical
		idx = (q > self.loc_l) & (q < self.loc_r)
		p[idx] = self._cdf(q[idx])
		
		## Left part
		idx = q < self.loc_l
		ql  = -(q[idx] - self.loc_l) - self.loc_l
		p[idx] = (1-self._p) * self._gpd_l.sf(ql)
		
		## Right part
		idx = q > self.loc_r
		p[idx] = self._p + (1-self._p) * self._gpd_r.cdf(q[idx])
		
		return p
	##}}}
	
	def icdf( self , p ):##{{{
		p = np.array([p]).reshape(-1)
		
		q = np.zeros_like(p) + np.nan
		
		## Empirical part
		idx = ( p > (1-self._p) ) & (p < self._p)
		q[idx] = self._icdf(p[idx])
		
		## Left part
		idx = p < 1 - self._p
		q[idx] = - self._gpd_l.isf( p[idx] / (1-self._p) )
		
		## Right part
		idx = p > self._p
		q[idx] = self._gpd_r.isf( ( 1 - p[idx] ) / (1-self._p) )
		
		return q
	##}}}
	
	def sf( self , q ):##{{{
		return 1 - self.cdf(q)
	##}}}
	
	def isf( self , p ):##{{{
		return self.icdf(1-p)
	##}}}
	
	def ppf( self , p ):##{{{
		return self.icdf(p)
	##}}}
	
	## Attributes ##{{{
	
	@property
	def p(self):
		return self._p
	
	@property
	def loc_l(self):
		return -float(self._gpd_l.kwds["loc"])
	
	@property
	def loc_r(self):
		return float(self._gpd_r.kwds["loc"])
	
	@property
	def scale_l(self):
		return float(self._gpd_l.kwds["scale"])
	
	@property
	def scale_r(self):
		return float(self._gpd_r.kwds["scale"])
	
	@property
	def shape_l(self):
		return float(self._gpd_l.kwds["c"])
	
	@property
	def shape_r(self):
		return float(self._gpd_r.kwds["c"])
	
	##}}}
	
##}}}

