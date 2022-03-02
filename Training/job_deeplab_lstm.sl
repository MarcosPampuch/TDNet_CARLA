#!/bin/bash

# Slurm submission script, 
# GPU job 
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

#shared ressources
##SBATCH --share


# Job name
#SBATCH -J "Pytorch_DeepLab_lstm_norm"

# Batch output file
#SBATCH --output tPytorch_Deeplab_lstm.o%J

# Batch error file
#SBATCH --error tPytorch_Deeplab_lstm.e%J


#SBATCH --partition gpu_k80

#SBATCH --time 48:00:00
#SBATCH --gres gpu:1


#SBATCH --cpus-per-task 4

#SBATCH --mem 20000
# -----
#SBATCH --mail-type ALL
# User e-mail address
##SBATCH --mail-user marcos.grassi@groupe-esigelec.org
# environments
# -----

module load python3-DL/3.7.6
source /home/2018015/mgrass01/.bashrc
conda activate venv
# ---------------------------------


cp -ar ./ $LOCAL_WORK_DIR
cd $LOCAL_WORK_DIR
echo Working directory : $PWD
CUDA_VISIBLE_DEVICES=0,1

#add wanted options on the next line
#srun python3 ./train.py --config configs/td4_psp18_cityscapes.yml
srun python3 ./train_carla.py --config configs/td4_psp18_cityscapes.yml
#python3 ./train_teacher.py --config configs/td4_psp18_carla.yml

# Move output data to target directory
mkdir $SLURM_SUBMIT_DIR/$SLURM_JOB_ID
mv *.pth *.txt $SLURM_SUBMIT_DIR/$SLURM_JOB_ID

sacct --format=AllocCPUs,AveCPU,MaxRSS,MaxVMSize,JobName -j $SLURM_JOB_ID
