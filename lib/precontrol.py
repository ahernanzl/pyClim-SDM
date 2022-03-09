import sys
sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import BC_lib
import down_scene_ANA
import down_scene_BC
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
import standardization
import TF_lib
import val_lib
import WG_lib
import write

########################################################################################################################
def predictors_strength():
    """
    Test the strength of the predictors/predictand relationships. It can be used to select the most relevant predictors
    for the downscaling.
    """
    print('Work in progress')
    exit()

########################################################################################################################
def GCMs_availability():
    """
    Check for missing data in predictors by GCMs. It can be used to discard some predictors/levels.
    """
    print('Work in progress')
    exit()

########################################################################################################################
def GCMs_reliability():
    """
    Test the reliability of GCMs in a historical period comparing them with a reanalysis, analysing all predictors,
    models, synoptic analogy fields... It can be used to discard models/predictors and to detect outliers.
    """
    print('Work in progress')
    exit()

########################################################################################################################
def GCMs_uncertainty():
    """
    Test the uncertainty in GCMs in the future, given by the multimodel spread, analysing all predictors, models,
    synoptic analogy fields... It can be used to discard models/predictors and to detect outliers.
    """
    print('Work in progress')
    exit()
