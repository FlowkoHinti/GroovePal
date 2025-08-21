#!/bin/bash
# ===== SLURM SETTINGS =====
#SBATCH --job-name=groovepal
#SBATCH --output=Log/%x_%j.out
#SBATCH --error=Log/%x_%j.err
#SBATCH --time=5-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --chdir=/data/fhinterberger/GroovePal  # alternative to manual cd

set -euo pipefail
mkdir -p Log

echo "[$(date)] Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "CPUs/task=${SLURM_CPUS_PER_TASK:-?}  GRES=${SLURM_JOB_GRES:-?}  CVD=${CUDA_VISIBLE_DEVICES:-unset}"

# Threading limits (avoid oversubscription)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1


# --- CUDA toolchain autodetect (conda first, then system) ---
# Prefer the conda toolchain if installed
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
  export CUDA_HOME="$CONDA_PREFIX"
else
  # Fallback: try to infer from whatever nvcc is on PATH
  if command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH="$(command -v nvcc)"
    export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
  fi
fi

# Set correct lib path
if [ -n "$CUDA_HOME" ]; then
  if [ -d "$CUDA_HOME/lib64" ]; then
    export CUDA_LIB="$CUDA_HOME/lib64"
  else
    export CUDA_LIB="$CUDA_HOME/lib"
  fi
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_LIB:${LD_LIBRARY_PATH:-}"
fi

# --- Diagnostics ---
echo "CONDA_PREFIX : $CONDA_PREFIX"
echo "CUDA_HOME    : ${CUDA_HOME:-<unset>}"
echo "CUDA_LIB     : ${CUDA_LIB:-<unset>}"
echo "which nvcc   : $(command -v nvcc || echo 'not found')"
[ -n "$CUDA_HOME" ] && ls -l "$CUDA_HOME/bin/nvcc" || true
[ -n "$CUDA_LIB" ]  && ls -l "$CUDA_LIB"/libcublas* 2>/dev/null || echo "cuBLAS not found in CUDA_LIB"

# Quick PyTorch GPU availability test
python - <<'PY'
import torch, os
print("Torch version:", torch.__version__)
print("Torch built for CUDA:", torch.version.cuda)
print("CUDA available?", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA_LIB env:", os.environ.get("CUDA_LIB"))
PY

# (Optional) traps for preemption/termination
on_sigusr1() { echo "[$(date)] SIGUSR1: save a checkpoint here."; }
on_term()    { echo "[$(date)] SIGTERM: graceful exit."; exit 0; }
trap on_sigusr1 USR1
trap on_term TERM

# ===== ENVIRONMENT =====
set +u   # allow unset vars temporarily
eval "$(conda shell.bash hook)"
conda activate /home2/fhinterberger/miniconda/envs/dna_xlstm/
set -u   # restore strict mode

# Working directory
cd /data/fhinterberger/GroovePal

# ===== RUN =====
python Main.py --train experiment_multitask_relgu_mixedloss