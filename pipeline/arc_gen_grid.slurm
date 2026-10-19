#!/bin/bash
# Run a list of generation cells as one SLURM job array.
#
# Each line of the cells file is "model plang lang", and array task i runs line i
# through arc_gen.slurm, so every cell gets exactly the conditions a single job
# would. One array instead of one job per cell keeps a cell from ever being
# queued twice, and the %N throttle caps how many GPUs the grid holds at once.
#
#   sbatch --array=1-45%15 pipeline/arc_gen_grid.slurm pipeline/cells_xlang_multilingual.txt
#
# Resume-safe like arc_gen.slurm: a cell already complete exits at once.
#
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --account=engs-applied-machine-learning
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=smell-grid
#SBATCH --output=/home/some5943/logs/grid_%A_%a.log

set -euo pipefail
CELLS=${1:?usage: sbatch --array=1-N pipeline/arc_gen_grid.slurm CELLS_FILE}
read -r MODEL PLANG LANG_CODE < <(grep -v '^#' "$CELLS" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "cell $SLURM_ARRAY_TASK_ID: $MODEL / $PLANG / $LANG_CODE"

# Yi-Coder needs its own venv; see arc_gen.slurm.
if [ "$MODEL" = "yi-coder-9b" ] || [ "$MODEL" = "yi-coder-1.5b" ]; then
  export GEN_VENV=/home/some5943/yi-venv
fi
bash pipeline/arc_gen.slurm --model "$MODEL" --lang "$LANG_CODE" --plang "$PLANG"
