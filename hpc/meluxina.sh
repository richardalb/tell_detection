#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=4                         # number of tasks
#SBATCH --ntasks-per-node=4                # number of tasks per node
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --cpus-per-task=32                 # number of cores per task
#SBATCH -t 12:00:00                        # time (HH:MM:SS)
#SBATCH --partition=gpu                    # partition
#SBATCH --account=<account_id>                  # project account
#SBATCH --qos=default                      # SLURM qos
#SBATCH --output logs/train_logs.out
#SBATCH --error logs/train_logs.err
#SBATCH --mail-user <email>
#SBATCH --mail-type END,FAIL,time_limit_80

module load Apptainer

models=("resnetv2_50" "swinv2_base_window12to24_192to384" "vit_base_patch16_384" "vgg19")
checkpoint_names=("resnetv2_50" "swinv2_base_window12to24_192to384" "vit_base_patch16_384" "vgg19")
batch_sizes=(120 12 70 100)
lrs=(0.001 0.003 0.005 0.01)

for i in $(seq 0 3); do
  echo ${lrs[${i}]}
  srun --exact -G 1 -c 32 -n 1 apptainer exec --nv \
    --bind "path/to/project/":/tell_detection:rw \
    --bind "path/to/data/":/tell_detection/data:rw \
    tell.sif python3 /tell_detection/src/train.py -m ${models[0]} -b ${batch_sizes[0]} -e 100 -dpath /tell_detection/data -cn ${checkpoint_names[0]} -lv 0 -lr ${lrs[${i}]} -nw 31 &
  srun --exact -G 1 -c 32 -n 1 apptainer exec --nv \
    --bind "path/to/project/":/tell_detection:rw \
    --bind "path/to/data/":/tell_detection/data:rw \
    tell.sif python3 /tell_detection/src/train.py -m ${models[1]} -b ${batch_sizes[1]} -e 100 -dpath /tell_detection/data -cn ${checkpoint_names[1]} -lv 0 -lr ${lrs[${i}]} -nw 31 &
  srun --exact -G 1 -c 32 -n 1 apptainer exec --nv \
    --bind "path/to/project/":/tell_detection:rw \
    --bind "path/to/data/":/tell_detection/data:rw \
    tell.sif python3 /tell_detection/src/train.py -m ${models[2]} -b ${batch_sizes[2]} -e 100 -dpath /tell_detection/data -cn ${checkpoint_names[2]} -lv 0 -lr ${lrs[${i}]} -nw 31 &
  srun --exact -G 1 -c 32 -n 1 apptainer exec --nv \
    --bind "path/to/project/":/tell_detection:rw \
    --bind "path/to/data/":/tell_detection/data:rw \
    tell.sif python3 /tell_detection/src/train.py -m ${models[3]} -b ${batch_sizes[3]} -e 100 -dpath /tell_detection/data -cn ${checkpoint_names[3]} -lv 0 -lr ${lrs[${i}]} -nw 31
done
