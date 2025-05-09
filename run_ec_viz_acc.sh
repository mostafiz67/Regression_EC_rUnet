#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mem=150G  # memory
#SBATCH --cpus-per-task=08
#SBATCH --output=runet-%j.out  # %N for node name, %j for jobID
#SBATCH --time=00-3:00     # time (DD-HH:MM)
#SBATCH --mail-user=x2020fpt@stfx.ca # used to send emails
#SBATCH --mail-type=ALL

module load python/3.8 cuda cudnn
# SOURCEDIR=/home/x2020fpt/home/

# Prepare virtualenv
source /home/x2020fpt/scratch/.venv/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"




echo -e '\n\n\n'
echo "$(date +"%T"):  start Collecting all residuals, saving them subjectwise and calculating EC subject wise and saving them!"
python3 /home/x2020fpt/scratch/rUnet_CC/project/calculate_ec.py \
echo "$(date +"%T"):  Finished running!"

echo -e '\n\n\n'
echo "$(date +"%T"):  Ploatting Parametric map images for middle part only!"
python3 /home/x2020fpt/scratch/rUnet_CC/project/visualize_ec.py \
echo "$(date +"%T"):  Finished running!"

echo -e '\n\n\n'
echo "$(date +"%T"): calculating Accuracy (Mean, Std) and saving them!"
python3 /home/x2020fpt/scratch/rUnet_CC/project/calculate_accuracy.py \
echo "$(date +"%T"):  Finished running!"
