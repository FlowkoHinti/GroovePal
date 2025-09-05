#!/bin/bash
#SBATCH --job-name=seq_absgu
#SBATCH --output=Log/%x_%j.out
#SBATCH --error=Log/%x_%j.err
#SBATCH --time=6-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/data/fhinterberger/GroovePal

set -Eeuo pipefail
mkdir -p Log

echo "[$(date)] Job $SLURM_JOB_ID on ${SLURMD_NODENAME:-$HOSTNAME}"
echo "CPUs/task=${SLURM_CPUS_PER_TASK:-?}  Partition=${SLURM_JOB_PARTITION:-?}"
echo "AllocTRES=${SLURM_TRES_ALLOC_STR:-?}  ReqTRES=${SLURM_TRES_REQ_STR:-?}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Threading limits
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

# --- Conda ---
set +u
eval "$(conda shell.bash hook)"
conda activate /home2/fhinterberger/miniconda/envs/dna_xlstm
set -u

# --- Discover CUDA ---
# Prefer conda-provided toolkit if present; otherwise rely on system CUDA.
CUDA_HOME_CANDIDATES=()
command -v nvcc >/dev/null 2>&1 && CUDA_HOME_CANDIDATES+=("$(dirname "$(dirname "$(command -v nvcc)")")")
[ -d "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ] && CUDA_HOME_CANDIDATES+=("$CONDA_PREFIX")
for C in "${CUDA_HOME_CANDIDATES[@]}"; do
  if [ -d "$C/lib64" ] || [ -d "$C/lib" ]; then
    export CUDA_HOME="$C"
    break
  fi
done

if [ -n "${CUDA_HOME:-}" ]; then
  if [ -d "$CUDA_HOME/lib64" ]; then CUDA_LIB="$CUDA_HOME/lib64"; else CUDA_LIB="$CUDA_HOME/lib"; fi
  export CUDA_LIB
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_LIB:${LD_LIBRARY_PATH:-}"
fi

# Target GPU arch list (don’t fail if query unsupported)
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
export TORCH_CUDA_ARCH_LIST="${CAP:-6.1;7.5;8.6}"

# --- Diagnostics (non-fatal) ---
{
  echo "CONDA_PREFIX : ${CONDA_PREFIX:-<unset>}"
  echo "CUDA_HOME    : ${CUDA_HOME:-<unset>}"
  echo "CUDA_LIB     : ${CUDA_LIB:-<unset>}"
  echo "which nvcc   : $(command -v nvcc || echo 'not found')"
  [ -n "${CUDA_LIB:-}" ] && ls -l "$CUDA_LIB"/libcublas* 2>/dev/null || echo "cuBLAS not found in CUDA_LIB"
  command -v gcc >/dev/null 2>&1 && gcc --version | head -1 || echo "gcc not found"
  command -v g++ >/dev/null 2>&1 && g++ --version | head -1 || echo "g++ not found"
  echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
} || true

# Quick PyTorch GPU availability test (non-fatal)
python - <<'PY' || true
import torch, os
print("Torch version:", torch.__version__)
print("Torch built for CUDA:", torch.version.cuda)
print("CUDA available?", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA_LIB env:", os.environ.get("CUDA_LIB"))
PY

# Graceful traps (optional)
trap 'echo "[$(date)] SIGUSR1: checkpoint."' USR1
trap 'echo "[$(date)] SIGTERM: exit."; exit 0' TERM

# ===== RUN =====
# Use srun so cgroups/accounting apply to the task:
srun --cpu-bind=cores python Main.py --train experiment_sequential_absgu_pretrain