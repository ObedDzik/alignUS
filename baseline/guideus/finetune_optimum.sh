#!/bin/bash

#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --job-name=guideus_optimum_f0
#SBATCH --output=logs/guideus_baseline/%x-%A-%a.log

export CHECKPOINT=/scratch/obed
export JOB_ID=$SLURM_JOB_ID
export MEDSAM_CHECKPOINT_DIR=/datasets/exactvu_pca/checkpoint_store
export MEDSAM_CHECKPOINT=/datasets/exactvu_pca/checkpoint_store/sam/medsam_vit_b_cpu.pth
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running (never commit a real key here)}"
export NCT_RAW_DATA_DIR=/datasets/exactvu_pca/nct2013
export NCT_METADATA_PATH=/datasets/exactvu_pca/nct2013/metadata.csv
export EXACTVU_PCA_DATA_ROOT=/datasets/exactvu_pca
export DINOV3_LIBRARY_PATH=/home/obed/projects/aip-medilab/obed/medproj/dinov3
export DINOV3_CHECKPOINTS_PATH=/datasets/exactvu_pca/checkpoint_store/dinov3

module load python/3.12 cuda/12.2  # cluster-specific; drop if you are not on this cluster
module load opencv/4.12.0        # cluster-specific; drop if you are not on this cluster

source venv/bin/activate  # your own venv -- pip install -r requirements.txt
srun python -m baseline.guideus.guideus_pnf_train -c baseline/guideus/guideus_cfg0_optimum.yaml "$@"
