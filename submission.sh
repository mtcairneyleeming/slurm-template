#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=vae_training_test
#SBATCH --time=00:60:00
#SBATCH --partition=short
#SBATCH -o ./reports/output.%a.out # STDOUT
######SBATCH --gres=gpu:1

# TODO: add GPUs

export WORKING_DIR=$DATA/slurm-template
export CONDA_PREFIX=$DATA/pv_env


module load Mamba # note we are not using Mamba to build the environment, we just need to load into it

# set the Anaconda environment, and activate it:
source activate $CONDA_PREFIX


# change to the temporary $SCRATCH directory, where we can create whatever files we want
cd $SCRATCH
mkdir output # create an output folder, which we will copy across to $DATA when done

# copy across whatever files we need
cp $WORKING_DIR/train.py ./train.py
cp $WORKING_DIR/config.py ./config.py
# copy the config to the output as well, so we know what this run was setup with
cp $WORKING_DIR/train.py ./output/config.py 

python train.py


# copy the output directory back across to $DATA
mkdir $WORKING_DIR/outputs/$SLURM_JOB_ID
tree ./output
rsync -av ./output $WORKING_DIR/outputs/$SLURM_JOB_ID
