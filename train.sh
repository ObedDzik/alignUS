#!/bin/bash

#SBATCH --account=aip-medilab  # cluster-specific SLURM allocation; change to your own
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=26:30:00
#SBATCH --job-name=alignus_f0
#SBATCH --output=logs/%x-%A-%a.log

# ----- environment -----
# Copy .env.example to .env and fill in your own paths/credentials first
# (see README.md's Data section for what each one should contain).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$REPO_ROOT/.env" ]; then
  echo "Missing $REPO_ROOT/.env -- copy .env.example to .env and fill in your paths." >&2
  exit 1
fi
set -a
source "$REPO_ROOT/.env"
set +a
export JOB_ID=$SLURM_JOB_ID
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
