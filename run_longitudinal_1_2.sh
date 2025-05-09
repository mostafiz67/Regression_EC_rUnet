#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --gres=gpu:v100:1  # on Cedar
#SBATCH --mem=120G  # memory
#SBATCH --cpus-per-task=08
#SBATCH --output=runet-1-%j.out  # %N for node name, %j for jobID
#SBATCH --err=runet-1-%j.err
#SBATCH --time=00-08:00     # time (DD-HH:MM)
#SBATCH --mail-user=x2020fpt@stfx.ca # used to send emails
#SBATCH --mail-type=ALL

module load python/3.8 cuda cudnn
SOURCEDIR=/home/x2020fpt/scratch/

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=^docker0,lo

# force to synchronization, can pinpoint the exact number of lines of error code where our memory operation is observed
CUDA_LAUNCH_BLOCKING=1

# Prepare virtualenv
source /home/x2020fpt/scratch/.venv/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"


echo -e '\n'
cd $SLURM_TMPDIR
mkdir work
echo "$(date +"%T"):  Copying data"
tar -xf /home/x2020fpt/scratch/Long_Data/ADNI_longitudinal_AD/ADNI.tar.xz -C work && echo "$(date +"%T"):  Copied data"

cd work


GPUS=1
BATCH_SIZE=3
LOSS=l2  # l1 l2 smoothl1

TASK=longitudinal  # diffusion longitudinal
ACTIVATION=LeakyReLU # LeakyReLU ReLU
WEIGHT_DECAY=1e-8
IN_CHANNELS=3
KFOLD_NUM=1
NUM_REP=2
LEARNING_RATE=1e-5
SEED=43

LOG_DIR=/home/x2020fpt/scratch/rUnet_CC_1

echo -e '\n\n\n'
echo "$(date +"%T"):  start running model!"
tensorboard --logdir="$LOG_DIR" --host 0.0.0.0 & python3 /home/x2020fpt/scratch/rUnet_CC_1/project/main.py \
       --gpus=$GPUS \
       --in_channels=$IN_CHANNELS \
       --loss="$LOSS" \
       --weight_decay=$WEIGHT_DECAY \
       --activation="$ACTIVATION" \
       --batch_size=$BATCH_SIZE \
       --kfold_num=$KFOLD_NUM \
       --num_rep=$NUM_REP \
       --seed=$SEED \
       --task="$TASK" \
       --learning_rate=$LEARNING_RATE \
       --tensor_board_logger="$LOG_DIR"
 echo "$(date +"%T"):  Finished running!"
