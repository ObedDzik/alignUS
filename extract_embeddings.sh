#!/bin/bash
# extract_embeddings.sh — sets the training environment, then hands EVERY
# argument straight through to extract_embeddings.py.
#
# Use exactly the flags the Python script takes; nothing is positional:
#
#   bash extract_embeddings.sh \
#       --config  /path/to/cfg.yaml \
#       --checkpoint /path/to/best.pth \
#       --train_module train_patched \
#       --run_name us_histo_512supcon_only_0 --fold 0 --with_train \
#       --plot figures/embedding_supconf0.pdf
#
# Submit under slurm with:  sbatch extract_embeddings.sh <same flags>

#SBATCH --job-name=extract_emb
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/extract_%j.out

set -euo pipefail

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

# ----- project imports -----
# extract_embeddings.py imports the training module by dotted path; medAI is
# vendored inside this repo (medAI/, external_libs/), so only this repo's own
# root needs to be on PYTHONPATH.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# module load python/3.12 cuda/12.2                          # cluster-specific; uncomment if needed
# source venv/bin/activate                                    # your own venv -- pip install -r requirements.txt

if [ "$#" -eq 0 ]; then
    echo "usage: bash $0 --config <cfg.yaml> --checkpoint <best.pth> \\"
    echo "         --train_module <dotted.path> --run_name <name> [--fold N] \\"
    echo "         [--with_train] [--plot figures/x.pdf] [--out runs/predictions]"
    exit 1
fi

# Resolve the python script next to this wrapper, so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "args         : $*"
echo

exec python "${SCRIPT_DIR}/extract_embeddings.py" "$@"
