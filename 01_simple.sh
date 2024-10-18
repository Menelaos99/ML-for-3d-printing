#!/bin/bash

#SBATCH --partition=gpucloud
#SBATCH --ntasks=1              # Requested (MPI)tasks. Default=1
#SBATCH --cpus-per-task=1       # Requested CPUs per task. Default=1
#SBATCH --mem=10G                # Memory limit. [1-999][K|M|G|T]
#SBATCH --time=100:00:00         # Time limit. [[days-]hh:]mm[:ss]
#SBATCH --gpus=0

### configure file to store console output.
### Write output to /dev/null to discard output
#SBATCH --output=output.log

### configure email notifications
### mail types: BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90,TIME_LIMIT_80
#SBATCH --mail-user=ga27dus@tum.de
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

### give your job a name (and maybe a comment) to find it in the queue
#SBATCH --job-name=batch
#SBATCH --comment="simple sbatch script"

### run your program...
source /home/ga27dus/env/3dprint/bin/activate
srun python returnpoints.py