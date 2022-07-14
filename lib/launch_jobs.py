import sys

sys.path.append('../config/')
from imports import *
from settings import *
from advanced_settings import *

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

########################################################################################################################
def wait_maxJobs():


    message_printed = False

    while 1:
        # Check number of living jobs
        os.system('squeue -u ' + user + ' | wc -l > ../log/nJobs.txt')
        f = open('../log/nJobs.txt', 'r')
        nJobs = int(f.read()) - 1
        f.close()
        time.sleep(1)
        if nJobs == max_nJobs and message_printed == False:
            print('max_nJobs', max_nJobs, 'reached. Waiting for jobs to finish...')
            message_printed = True

        if nJobs < max_nJobs:
            break

########################################################################################################################
def info_msg():
    print('----------------------------------------------------------------')
    print('You are about to launch a job.')
    print('Different methods might need different number of nodes and memory.')
    print('You can optimize your configuration in lib/launch_jobs.py.')
    print('Do not launch next steps until current jobs have finished')
    print('Check stdout and stderr files at the jobs/ directory')
    print('----------------------------------------------------------------')

########################################################################################################################
def standardize(targetGroup, grid, model, scene):
    """
    Launch a job for calculating mean and std of each model and scene in parallel
    """


    # Wait max jobs allowed
    wait_maxJobs()

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

    os.system('sbatch --job-name=' + targetGroup + '_' + grid + '_' + model + ' ' +
              ' '.join((job_file, targetGroup, grid, model, scene)))


########################################################################################################################
def cluster(targetVar, methodName, mode, cluster_option):
    """
    Launch a job for training chunks of clusters / weather types in parallel
    """


    # Wait max jobs allowed
    wait_maxJobs()

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

    os.system('sbatch --job-name=' + targetVar + '_' + methodName + ' ' +
              ' '.join((job_file, targetVar, methodName, mode, cluster_option)))


########################################################################################################################
def training(targetVar, methodName, family, mode, fields):
    """
    Launch a job for training chunks of grid-points in parallel
    """

    # Wait max jobs allowed
    wait_maxJobs()

    # Display info message
    info_msg()


    job_file = '../lib/job.sh'

    # Define number of cores and memory
    n = 256
    mem = 250000

    if methodName == 'WG-PDF':
        n = 16
    elif methodName == 'RF' and targetVar == 'pr':
        n = 32
    elif methodName == 'RF' and targetVar != 'pr':
        n = 64
    elif methodName == 'XGB':
        n = 32
    elif methodName in ('ANN', 'CNN', ):
        n = 128
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

    os.system('sbatch --job-name=' + targetVar + '_' + methodName + ' ' +
              ' '.join((job_file, targetVar, methodName, family, mode, fields)))


########################################################################################################################
def process(targetVar, methodName, family, mode, fields, scene, model):
    """
    Launch a job for training chunks of dates (for ANA/WT) or grid-points (for the rest) in parallel
    """

    # Wait max jobs allowed
    wait_maxJobs()

    # Display info message
    info_msg()

    job_file = '../lib/job.sh'

    # Define number of cores and memory
    mem = 250000
    n = 256

    if family == 'ANA':
        n = 80
    elif family == 'MOS':
        n = 80
    if methodName == 'XGB':
        n = 80
    elif methodName == 'RF' and targetVar != 'pr':
        n = 128
    elif methodName == 'ANN':
        n = 80
    elif methodName in ('CNN', ):
        n = 80
    elif methodName == 'LS-SVM':
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
              ' '.join((job_file, targetVar, methodName, family, mode, fields, scene, model)))


########################################################################################################################
def climdex(model, targetVar, methodName):
    """
    Launch a job for calclulating climdex for each GCM in parallel
    """

    # Wait max jobs allowed
    wait_maxJobs()

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

    os.system('sbatch --job-name=' + model + ' ' + job_file + ' ' + ' ' + model + ' ' + targetVar + ' ' + methodName)


########################################################################################################################
def biasCorrection(model, targetVar, methodName):
    """
    Launch a job for bias correcting each model in parallel
    """

    # Wait max jobs allowed
    wait_maxJobs()

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

    os.system('sbatch --job-name=' + model + ' ' + job_file + ' ' + ' ' + model + ' ' + targetVar + ' ' + methodName)

