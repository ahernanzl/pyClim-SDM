import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='No protocol specified')
import sys
import os
import shutil
import datetime
import numpy as np
import time
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
import math
import seaborn as sns
from sys import exit
import collections
from multiprocessing import Pool
from os import listdir
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import sklearn
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gamma
from scipy.stats import norm
from scipy.signal import detrend
from scipy.optimize import fsolve
from netCDF4 import Dataset
from netCDF4 import date2num
from netCDF4 import num2date
import glob
import random
import subprocess
import platform
import geopandas as gpd
from geopy.distance import distance as dist
from shapely.geometry import Point, Polygon
from statsmodels.distributions.empirical_distribution import ECDF


sys.path.append('../lib/')
import ANA_lib
import aux_lib
import derived_predictors
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
import MOS_lib
import plot
import postpro_lib
import postprocess
import precontrol
import preprocess
import process
import read
import standardization
import TF_lib
import val_lib
import WG_lib
import write

