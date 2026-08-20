#!/bin/bash
# Translate the 426 prompts into es / fr / zh on ARC.
#
# NLLB-600M is small; this is well under an hour on any of the GPUs.
#
#   sbatch pipeline/arc_translate.slurm --lang es --lang fr --lang zh
#
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --account=engs-applied-machine-learning
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=smell-translate
#SBATCH --output=/home/some5943/logs/translate_%j.log

set -euo pipefail
source ~/smell-env.sh
export HF_HOME=/data/engs-applied-machine-learning/some5943/hf

REPO=${REPO:-~/code-smell-multilingual}
export PROMPT_CACHE=${PROMPT_CACHE:-/data/engs-applied-machine-learning/some5943/_prompts}

echo "node: $(hostname)"
cd "$REPO"
python -m pipeline.translate_prompts --out "$PROMPT_CACHE" "$@"
