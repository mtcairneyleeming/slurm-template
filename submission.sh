#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=vae_training_test
#SBATCH --time=00:60:00
#SBATCH --partition=short

# TODO: add GPUs

# set the correct paths/etc:
CONDA_ENV=$DATA/pvenv5              # the prefix of the conda environment
WORKING_DIR=$DATA/slurm-template    # where to load python from


module load Mamba # note we are not using Mamba to build the environment, we just need to load into it

# set the Anaconda environment, and activate it:
CONDA_ENV=$DATA/pvenv5   
source activate $CONDA_ENV

# change to the temporary $SCRATCH directory, where we can create whatever files we want
cd $SCRATCH 
mkdir output # create an output folder, which we will copy across to $DATA when done

# copy across whatever files we need
WORKING_DIR=$DATA/slurm-template 
cp $WORKING_DIR/train.py ./train.py

python train.py


# copy the output directory back across to $DATA
mkdir $WORKING_DIR/$SLURM_JOB_ID
rsync -av ./output/* $DATA/script_tests/$SLURM_JOB_ID/