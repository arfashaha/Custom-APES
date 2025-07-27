#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 12:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate APES

# SCRATCH_DISK=/disk/scratch_big
# dest_path=${SCRATCH_DISK}/${USER}/APES

# ====================
# Clean up scratch space
# ====================
# if [ -d "${dest_path}" ]; then
#     echo "Deleting scratch disk path: ${dest_path}"
#     rm -rf "${dest_path}"
# else
#     echo "Scratch disk path does not exist: ${dest_path}"
# fi

# mkdir -p ${dest_path}
# src_path=/home/${USER}/APES/
# rsync -azvP ${src_path} ${dest_path}
# rsync -azvP --exclude 'MiniMarket_raw/' --exclude 'object_segmentation_dataset/' ${src_path} ${dest_path}
# rsync -azvP "/home/s2737104/MiniMarket_dataset_processing/object_segmentation_dataset/hazelnut_cocoa_spread_nutella_350gm_1200_2048_segmentation_4096_20480" ${dest_path}

# python segmentation_dataset_prep.py
# python segmentation_dataset_prep_torch.py
bash utils/dist_train.sh configs/apes/apes_seg_local-custom-50epochs.py 2
# python segmentation_visualization.py
# python generate_hdf5_dataset_with_padding.py

# echo "Moving output data back to Home"

# rsync -azvP ${dest_path}runs/ ${src_path}/results/

# ====================
# Clean up scratch space
# ====================
# echo "Deleting scratch disk path: ${dest_path}"
# rm -rf ${dest_path}


echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
