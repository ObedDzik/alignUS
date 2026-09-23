#!/bin/bash

#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_paper_metrics_guideus
#SBATCH --output=logs/guideus_baseline/%x-%A.log

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
srun python -m baseline.guideus.eval_paper_metrics \
  --cfg baseline/guideus/guideus_cfg0_optimum.yaml \
  --ckpts \
    fold0=/scratch/obed/guideus_baseline/guideus_optimum_0/5567062/best.pth \
    fold1=/scratch/obed/guideus_baseline/guideus_optimum_1/5567063/best.pth \
    fold2=/scratch/obed/guideus_baseline/guideus_optimum_2/5567064/best.pth \
    fold3=/scratch/obed/guideus_baseline/guideus_optimum_3/5567065/best.pth \
    fold4=/scratch/obed/guideus_baseline/guideus_optimum_4/5567066/best.pth \
  --out baseline/guideus/results_paper_metrics.json
