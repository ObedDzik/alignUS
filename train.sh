#!/bin/bash

#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=26:30:00
#SBATCH --job-name=alignus_f0
#SBATCH --output=logs/%x-%A-%a.log

export CHECKPOINT=/scratch/obed
export JOB_ID=$SLURM_JOB_ID
export MEDSAM_CHECKPOINT_DIR=/datasets/exactvu_pca/checkpoint_store
export MEDSAM_CHECKPOINT=/datasets/exactvu_pca/checkpoint_store/sam/medsam_vit_b_cpu.pth
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running (never commit a real key here)}"
export NCT_RAW_DATA_DIR=/datasets/exactvu_pca/nct2013
export NCT_METADATA_PATH=/datasets/exactvu_pca/nct2013/metadata.csv
export DINOV3_LIBRARY_PATH=/home/obed/projects/aip-medilab/obed/medproj/dinov3
export EXACTVU_PCA_DATA_ROOT=/datasets/exactvu_pca
export DINOV3_CHECKPOINTS_PATH=/datasets/exactvu_pca/checkpoint_store/dinov3
export CUDA_LAUNCH_BLOCKING=1

module load python/3.12 cuda/12.2  # cluster-specific; drop if you are not on this cluster
module load opencv/4.12.0        # cluster-specific; drop if you are not on this cluster
source venv/bin/activate  # your own venv -- pip install -r requirements.txt

mkdir -p logs

# --- core alignUS method (propBCE + SupCon), and its backbone/loss ablations ---
srun python -m train_patched -c cfgs/cfg_alignus/alignus_f0.yaml
# srun python -m train_patched -c cfgs/cfg_dino/dino_cfg0.yaml
# srun python -m train_patched -c cfgs/cfg_medsam/medsam_cfgf0.yaml
# srun python -m train_patched -c cfgs/cfg_triplet/triplet_cfgf0.yaml

# --- baselines — see baseline/README.md ---
# srun python -m baseline.guideus.guideus_pnf_train -c baseline/guideus/guideus_cfg0.yaml
# srun python -m baseline.guideus.guideus_pnf_train -c baseline/pnf/cfg/pnf_cfg0.yaml
# srun python -m train_patched -c baseline/acmil/cfg/supcon_onlyf0.yaml
# srun python -m train_patched -c baseline/aem/cfg/aem_f0.yaml
# srun python -m train_patched -c baseline/microsegnet/cfg/microseg_cfg0.yaml
