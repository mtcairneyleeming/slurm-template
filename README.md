# slurm-template
A basic example to run PyTorch (and hopefully JAX) code on SLURM (specifically Oxford's ARC systems) in a organised fashion.


## How to use

write all your code, like model definitions, the training and test routines, image processing, inference/etc. in the `code` directory.

change `init.py` to run the python code you want in the job, using the arguments passed from config.py

in `visualise.ipynb`, import anything you need from the `code` directory - e.g. the actual model definition, and use my code in the notebook to load the outputs of the training run for graphing/future use/etc.

set arguments for the job, like learning rate/ no. batches, etc. in config.py - this will be used in training, and copied together with the output for reproducibility.

`outputs` contains any files your job saves to the `output` directory while running, in separate directories per job id. `reports` is the same for the stdout printed by the job - at the moment it has some GPU info, and lists all the files copied across at the start and end.

edit `download.py` to download any training data/etc. you need to the `data` folder, so it can be used in the job - I don't believe you can download it in the job itself. Everything in `data` will be copied to `$SCRATCH/data` - the job is run in `$SCRATCH`

to submit, run `sbatch submission.sh` - you *can't* be in a Conda environment while doing this, as then all the python imports will fail.
