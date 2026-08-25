#!/bin/bash
# Generation job for the ARC htc GPU cluster.
#
# Model and language are variables, so the same script produces every cell of the
# matrix under identical conditions -- which is what makes the cells comparable.
#
#   sbatch pipeline/arc_gen.slurm --model qwen2.5-coder-1.5b --lang en --limit 4
#   MODEL=yi-coder-1.5b LANG_CODE=en sbatch pipeline/arc_gen.slurm
#
# Resume-safe: prompts already in results.jsonl are skipped, so a job that hits
# its wall clock is simply resubmitted.
#
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --account=engs-applied-machine-learning
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=smell-ml
#SBATCH --output=/home/some5943/logs/mlgen_%j.log

set -euo pipefail
source ~/smell-env.sh
# Yi-Coder needs an older transformers: 5.x classifies its sentencepiece
# tokenizer.model as a tiktoken file and fails parsing it. GEN_VENV points a job
# at ~/yi-venv, which pins transformers 4.46 and inherits torch from smell-venv
# via a .pth file, so there is only one CUDA stack on disk.
if [ -n "${GEN_VENV:-}" ]; then
  source "$GEN_VENV/bin/activate"
  echo "venv: $GEN_VENV"
fi
export HF_HOME=/data/engs-applied-machine-learning/some5943/hf

REPO=${REPO:-~/code-smell-multilingual}
# Outputs are large and belong on the project filesystem, not in $HOME.
export OUT_ROOT=${OUT_ROOT:-/data/engs-applied-machine-learning/some5943/outputs_arc}
export PROMPT_CACHE=${PROMPT_CACHE:-/data/engs-applied-machine-learning/some5943/_prompts}
# A whole batch of KV cache at 2048 new tokens is what fragments the allocator.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "node:  $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO"
python -m pipeline.arc_generate "$@"
