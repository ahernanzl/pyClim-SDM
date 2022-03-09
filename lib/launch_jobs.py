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


def info_msg():
    print('----------------------------------------------------------------')
    print('You are about to launch a job.')
    print('Different methods might need different number of nodes and memory.')
    print('You can optimize your configuration in lib/launch_jobs.py.')
    print('Do not launch next steps until current step jobs are finished')
    print('Check stdout and stderr files at the jobs/ directory')
    print('----------------------------------------------------------------')


########################################################################################################################
def standardize(var0, grid, model, scene):
    """
    Launch a job for calculating mean and std of each model and scene in parallel
    """

    # Display info message
    info_msg()

    job_file = '../lib/job.sh'

    # Define number of cores and memory
    n = 1
    mem = 250000

    f = open(job_file, 'w')
    f.writelines('#!/bin/bash\n')
    f.writelines('#SBATCH -p ' + HPC_partition + '\n')
    f.writelines('#SBATCH -o ../job/%j.out\n')
    f.writelines('#SBATCH -e ../job/%j.err\n')
    f.writelines('#SBATCH -N 1\n')  # nodes requested
    f.writelines('#SBATCH -n ' + str(n) + '\n')  # tasks requested (256xN...)
    f.writelines('#SBATCH -c 1\n')  # cores per task
    f.writelines('#SBATCH --mem=' + str(mem) + '\n')
    f.writelines('SECONDS=0\n')
    f.writelines('srun -n $SLURM_NTASKS --mpi=pmi2 python3 ../lib/standardization.py $1 $2 $3 $4\n')
    f.writelines('duration=$SECONDS\n')
    f.writelines('hours=$(($duration/3600))\n')
    f.writelines('duration=$(($duration%3600))\n')
    f.writelines('minutes=$(($duration/60))\n')
    f.writelines('seconds=$(($duration%60))\n')
    f.writelines('echo "$hours h $minutes m $seconds s"\n')
    f.writelines('sleep 5\n')
    f.writelines('echo "end"\n')
    f.close()

    os.system('sbatch --job-name=' + var0 + '_' + grid + '_' + model + ' ' +
              ' '.join((job_file, var0, grid, model, scene)))


########################################################################################################################
def cluster(var, methodName, mode, cluster_option):
    """
    Launch a job for training chunks of clusters / weather types in parallel
    """

    # Display info message
    info_msg()

    job_file = '../lib/job.sh'

    # Define number of cores and memory
    n = 256
    mem = 250000

    f = open(job_file, 'w')
    f.writelines('#!/bin/bash\n')
    f.writelines('#SBATCH -p ' + HPC_partition + '\n')
    f.writelines('#SBATCH -o ../job/%j.out\n')
    f.writelines('#SBATCH -e ../job/%j.err\n')
    f.writelines('#SBATCH -N 1\n')  # nodes requested
    f.writelines('#SBATCH -n ' + str(n) + '\n')  # tasks requested (256xN...)
    f.writelines('#SBATCH -c 1\n')  # cores per task
    f.writelines('#SBATCH --mem=' + str(mem) + '\n')
    # f.writelines('# SBATCH --exclusive\n')
    f.writelines('SECONDS=0\n')
    f.writelines('srun -n $SLURM_NTASKS --mpi=pmi2 python3 ../lib/ANA_lib.py $1 $2 $3 $4\n')
    f.writelines('duration=$SECONDS\n')
    f.writelines('hours=$(($duration/3600))\n')
    f.writelines('duration=$(($duration%3600))\n')
    f.writelines('minutes=$(($duration/60))\n')
    f.writelines('seconds=$(($duration%60))\n')
    f.writelines('echo "$hours h $minutes m $seconds s"\n')
    f.writelines('sleep 5\n')
    f.writelines('echo "end"\n')
    f.close()

    os.system('sbatch --job-name=' + var + '_' + methodName + ' ' +
              ' '.join((job_file, var, methodName, mode, cluster_option)))


########################################################################################################################
def training(var, methodName, family, mode, fields):
    """
    Launch a job for training chunks of grid-points in parallel
    """

    # Display info message
    info_msg()

    job_file = '../lib/job.sh'

    # Define number of cores and memory
    n = 256
    mem = 250000

    if methodName == 'WG-PDF':
        n = 16
    elif methodName == 'RF':
        n = 32
    elif methodName == 'ANN':
        n = 256
    if methodName[:3] == 'GLM':
        n = 128

    f = open(job_file, 'w')
    f.writelines('#!/bin/bash\n')
    f.writelines('#SBATCH -p ' + HPC_partition + '\n')
    f.writelines('#SBATCH -o ../job/%j.out\n')
    f.writelines('#SBATCH -e ../job/%j.err\n')
    f.writelines('#SBATCH -N 1\n')  # nodes requested
    f.writelines('#SBATCH -n ' + str(n) + '\n')  # tasks requested (256xN...)
    f.writelines('#SBATCH -c 1\n')  # cores per task
    f.writelines('#SBATCH --mem=' + str(mem) + '\n')
    # f.writelines('# SBATCH --exclusive\n')
    f.writelines('SECONDS=0\n')
    if family == 'TF':
        f.writelines('srun -n $SLURM_NTASKS --mpi=pmi2 python3 ../lib/TF_lib.py $1 $2 $3 $4 $5 $6\n')
    elif family == 'WG':
        f.writelines('srun -n $SLURM_NTASKS --mpi=pmi2 python3 ../lib/WG_lib.py $1 $2 $3 $4 $5 $6\n')
    f.writelines('duration=$SECONDS\n')
    f.writelines('hours=$(($duration/3600))\n')
    f.writelines('duration=$(($duration%3600))\n')
    f.writelines('minutes=$(($duration/60))\n')
    f.writelines('seconds=$(($duration%60))\n')
    f.writelines('echo "$hours h $minutes m $seconds s"\n')
    f.writelines('sleep 5\n')
    f.writelines('echo "end"\n')
    f.close()

    os.system('sbatch --job-name=' + var + '_' + methodName + ' ' +
              ' '.join((job_file, var, methodName, family, mode, fields)))


########################################################################################################################
def process(var, methodName, family, mode, fields, scene, model):
    """
    Launch a job for training chunks of dates (for ANA/WT) or grid-points (for the rest) in parallel
    """

    # Display info message
    info_msg()

    job_file = '../lib/job.sh'

    # Define number of cores and memory
    mem = 250000
    n = 256

    if family == 'ANA':
        n = 80
    elif family == 'BC':
        n = 80

    f = open(job_file, 'w')
    f.writelines('#!/bin/bash\n')
    f.writelines('#SBATCH -p ' + HPC_partition + '\n')
    f.writelines('#SBATCH -o ../job/%j.out\n')
    f.writelines('#SBATCH -e ../job/%j.err\n')
    f.writelines('#SBATCH -N 1\n')  # nodes requested
    f.writelines('#SBATCH -n ' + str(n) + '\n')  # tasks requested (256xN...)
    f.writelines('#SBATCH -c 1\n')  # cores per task
    f.writelines('#SBATCH --mem=' + str(mem) + '\n')
    # f.writelines('# SBATCH --exclusive\n')
    f.writelines('SECONDS=0\n')
    f.writelines('srun -n $SLURM_NTASKS --mpi=pmi2 python3 ../lib/down_scene_$3.py $1 $2 $3 $4 $5 $6 $7 $8\n')
    f.writelines('duration=$SECONDS\n')
    f.writelines('hours=$(($duration/3600))\n')
    f.writelines('duration=$(($duration%3600))\n')
    f.writelines('minutes=$(($duration/60))\n')
    f.writelines('seconds=$(($duration%60))\n')
    f.writelines('echo "$hours h $minutes m $seconds s"\n')
    f.writelines('sleep 5\n')
    f.writelines('echo "end"\n')
    f.close()

    os.system('sbatch --job-name=' + scene + '_' + model + ' ' +
              ' '.join((job_file, var, methodName, family, mode, fields, scene, model)))


########################################################################################################################
def climdex(model, var, methodName):
    """
    Launch a job for calclulating climdex for each GCM in parallel
    """

    # Display info message
    info_msg()

    job_file = '../lib/job.sh'

    # Define number of cores and memory
    n = 1
    mem = 250000

    f = open(job_file, 'w')
    f.writelines('#!/bin/bash\n')
    f.writelines('#SBATCH -p ' + HPC_partition + '\n')
    f.writelines('#SBATCH -o ../job/%j.out\n')
    f.writelines('#SBATCH -e ../job/%j.err\n')
    f.writelines('#SBATCH -N 1\n')  # nodes requested
    f.writelines('#SBATCH -n ' + str(n) + '\n')  # tasks requested (256xN...)
    f.writelines('#SBATCH -c 1\n')  # cores per task
    f.writelines('#SBATCH --mem=' + str(mem) + '\n')
    # f.writelines('# SBATCH --exclusive\n')
    f.writelines('SECONDS=0\n')
    f.writelines('srun -n $SLURM_NTASKS --mpi=pmi2 python3 ../lib/postprocess.py climdex $1 $2 $3\n')
    f.writelines('duration=$SECONDS\n')
    f.writelines('hours=$(($duration/3600))\n')
    f.writelines('duration=$(($duration%3600))\n')
    f.writelines('minutes=$(($duration/60))\n')
    f.writelines('seconds=$(($duration%60))\n')
    f.writelines('echo "$hours h $minutes m $seconds s"\n')
    f.writelines('sleep 5\n')
    f.writelines('echo "end"\n')
    f.close()

    os.system('sbatch --job-name=' + model + ' ' + job_file + ' ' + ' ' + model + ' ' + var + ' ' + methodName)


########################################################################################################################
def biasCorrection(model, var, methodName):
    """
    Launch a job for bias correcting each model in parallel
    """

    # Display info message
    info_msg()

    job_file = '../lib/job.sh'

    # Define number of cores and memory
    n = 1
    mem = 250000

    f = open(job_file, 'w')
    f.writelines('#!/bin/bash\n')
    f.writelines('#SBATCH -p ' + HPC_partition + '\n')
    f.writelines('#SBATCH -o ../job/%j.out\n')
    f.writelines('#SBATCH -e ../job/%j.err\n')
    f.writelines('#SBATCH -N 1\n')  # nodes requested
    f.writelines('#SBATCH -n ' + str(n) + '\n')  # tasks requested (256xN...)
    f.writelines('#SBATCH -c 1\n')  # cores per task
    f.writelines('#SBATCH --mem=' + str(mem) + '\n')
    # f.writelines('# SBATCH --exclusive\n')
    f.writelines('SECONDS=0\n')
    f.writelines('srun -n $SLURM_NTASKS --mpi=pmi2 python3 ../lib/postprocess.py bias_correction $1 $2 $3 $4\n')
    f.writelines('duration=$SECONDS\n')
    f.writelines('hours=$(($duration/3600))\n')
    f.writelines('duration=$(($duration%3600))\n')
    f.writelines('minutes=$(($duration/60))\n')
    f.writelines('seconds=$(($duration%60))\n')
    f.writelines('echo "$hours h $minutes m $seconds s"\n')
    f.writelines('sleep 5\n')
    f.writelines('echo "end"\n')
    f.close()

    # Set bc_method None to empty string
    local_bc_method = bc_method
    if local_bc_method == None:
        local_bc_method = ''

    os.system('sbatch --job-name=' + model + ' ' + job_file + ' ' + ' ' + model +
              ' ' ' ' + var + ' ' + methodName + ' ' + local_bc_method)

