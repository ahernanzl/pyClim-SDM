import sys

sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

sys.path.append('../deep4downscaling/')
import deep.loss as deep_loss
import deep.train as deep_train
import deep.models as deep_models
import deep.pred as deep_pred
import deep.utils as deep_utils

sys.path.append('../lib/')
import ANA_lib
import aux_lib
import derived_predictors
import DeepESD_lib
import down_scene_ANA
import down_scene_DeepESD
import down_scene_MOS
import down_scene_RAW
import down_scene_TF
import down_scene_WG
import down_day
import down_point
import evaluate_methods
import grids
import launch_jobs
import launch_jobs_GPU
import MOS_lib
import plot
import postpro_lib
import postprocess
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
def info_msg():
    print('----------------------------------------------------------------')
    print('You are about to launch a job.')
    print('Do not launch next steps until current jobs have finished')
    print('Check stdout and stderr files at the jobs/ directory')
    print('----------------------------------------------------------------')


########################################################################################################################
def training(targetVar, methodName, family, mode, fields):
    """
    Launch a job for training chunks of grid-points in parallel
    """

    # Display info message
    info_msg()


    job_file = '../lib/job.sh'

    # Define number of cores and memory
    n = 256
    mem = 250000

    f = open(job_file, 'w')
    f.write('#!/bin/bash\n')
    f.write('#SBATCH -o ../job/%j.out\n')
    f.write('#SBATCH -e ../job/%j.err\n')
    f.write('#SBATCH --qos=ng\n')
    f.write('#SBATCH --gpus=1\n')
    f.write('#SBATCH -t 2880\n')
    f.write('#SBATCH --mem=' + str(mem) + '\n')
    f.write('module load conda\n')
    f.write('conda activate env_pyClim-SDM\n')
    f.write('module load python3/new cuda\n')

    # Diagnóstico de GPU y entorno
    f.write('nvidia-smi\n')
    f.write('which python\n')
    f.write('python --version\n')

    # Verificación de CUDA en Python (corrigiendo comillas)
    f.write('python -c "import torch; print(\'CUDA disponible:\', torch.cuda.is_available())"\n')
    f.write('python -c "import torch; print(\'Versión CUDA:\', torch.version.cuda)"\n')
    f.write('python -c "import torch; print(\'Dispositivo:\', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'No GPU detectada\')"\n')

    f.write('SECONDS=0\n')
    f.write('srun python3 ../lib/DeepESD_lib.py $1 $2 $3 $4 $5 $6\n')
    f.write('duration=$SECONDS\n')
    f.write('hours=$(($duration/3600))\n')
    f.write('duration=$(($duration%3600))\n')
    f.write('minutes=$(($duration/60))\n')
    f.write('seconds=$(($duration%60))\n')
    f.write('echo "$hours h $minutes m $seconds s"\n')
    f.write('sleep 5\n')
    f.write('echo "end"\n')
    f.close()

    os.system('sbatch --job-name=' + targetVar + '_' + methodName + ' ' +
              ' '.join((job_file, targetVar, methodName, family, mode, fields)))


########################################################################################################################
def process(targetVar, methodName, family, mode, fields, scene, model):
    """
    Launch a job for training chunks of dates (for ANA/WT) or grid-points (for the rest) in parallel
    """

    # Display info message
    info_msg()

    job_file = '../lib/job.sh'

    # Define number of cores and memory
    mem = 250000
    n = 256

    f = open(job_file, 'w')
    f.write('#!/bin/bash\n')
    f.write('#SBATCH -o ../job/%j.out\n')
    f.write('#SBATCH -e ../job/%j.err\n')
    f.write('#SBATCH --qos=ng\n')
    f.write('#SBATCH --gpus=1\n')
    f.write('#SBATCH -t 2880\n')
    f.write('#SBATCH --mem=' + str(mem) + '\n')
    f.write('module load conda\n')
    f.write('conda activate env_pyClim-SDM\n')
    f.write('module load python3/new cuda\n')

    # Diagnóstico de GPU y entorno
    f.write('nvidia-smi\n')
    f.write('which python\n')
    f.write('python --version\n')

    # Verificación de CUDA en Python (corrigiendo comillas)
    f.write('python -c "import torch; print(\'CUDA disponible:\', torch.cuda.is_available())"\n')
    f.write('python -c "import torch; print(\'Versión CUDA:\', torch.version.cuda)"\n')
    f.write('python -c "import torch; print(\'Dispositivo:\', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'No GPU detectada\')"\n')

    f.write('SECONDS=0\n')
    f.write('srun python3 ../lib/down_scene_$3.py $1 $2 $3 $4 $5 $6 $7 $8\n')
    f.write('duration=$SECONDS\n')
    f.write('hours=$(($duration/3600))\n')
    f.write('duration=$(($duration%3600))\n')
    f.write('minutes=$(($duration/60))\n')
    f.write('seconds=$(($duration%60))\n')
    f.write('echo "$hours h $minutes m $seconds s"\n')
    f.write('sleep 5\n')
    f.write('echo "end"\n')
    f.close()

    os.system('sbatch --job-name=' + scene + '_' + model + ' ' +
              ' '.join((job_file, targetVar, methodName, family, mode, fields, scene, model)))





