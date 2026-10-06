#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7                 # number of cores per task
#SBATCH -t 48:00:00                        # time (HH:MM:SS)
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output logs/predict_logs_%x.out
#SBATCH --error logs/predict_logs_%x.err
#SBATCH --mail-user <email>
#SBATCH --mail-type END,FAIL,time_limit_80

module load tools/Apptainer

srun --exact -G 1 -c 7 -n 1 apptainer exec --nv \
  --bind "path/to/project/":/tell_detection:rw \
  --bind "path/to/data/":/tell_detection/data/corona_images:rw \
  --bind /scratch/users/ralbrecht/corona_images:/tell_detection/data/corona_images:rw \
  --bind "path/to/results_dir/":/tell_detection/results:rw \
  tell.sif python3 /tell_detection/src/predict.py -m resnetv2_50 -b 50 -cp /tell_detection/checkpoints/resnetv2_50/epoch=02-val_loss=0.03279-fold_3_lr_0.003.ckpt --countries $1
