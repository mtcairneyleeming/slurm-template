#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=vae_training_test
#SBATCH --time=00:10:00
#SBATCH --partition=short
#SBATCH -o ./reports/slurm-%j.out # STDOUT
#SBATCH --gres=gpu:1

# TODO: add GPUs



export WORKING_DIR=$DATA/slurm-template
export CONDA_PREFIX=/data/coml-hawkes/lady6235/.conda/envs/numpyro10_torch_env


module load Mamba # note we are not using Mamba to build the environment, we just need to load into it
module load CUDA/11.6.0

echo "CUDA Devices(s) allocated: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# set the Anaconda environment, and activate it:
source activate $CONDA_PREFIX


# change to the temporary $SCRATCH directory, where we can create whatever files we want
cd $SCRATCH
mkdir output # create an output folder, which we will copy across to $DATA when done
mkdir code

# copy across only what we need:
cp -R $WORKING_DIR/code   . # the code we've actually written
cp $WORKING_DIR/init.py   . # standard init file
cp $WORKING_DIR/config.py . # config file

cp -R $WORKING_DIR/data   . # any pre-downloaded data as well

# copy the config to the output as well, so we know what this run was setup with
cp $WORKING_DIR/config.py ./output/config.py

tree

python init.py


# copy the output directory back across to $DATA
mkdir $WORKING_DIR/outputs/$SLURM_JOB_ID
tree ./output
rsync -av ./output $WORKING_DIR/outputs/$SLURM_JOB_ID
