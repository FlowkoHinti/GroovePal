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

# Show which conda env is active
echo "CONDA_PREFIX: $CONDA_PREFIX"

# Show CUDA_HOME and CUDA_LIB
echo "CUDA_HOME:    $CUDA_HOME"
echo "CUDA_LIB:     $CUDA_LIB"

# Show PATH entries relevant to nvcc
echo "PATH includes CUDA bin?"
echo $PATH | tr ':' '\n' | grep -E "cuda|$CONDA_PREFIX"

# Show LD_LIBRARY_PATH entries relevant to CUDA libs
echo "LD_LIBRARY_PATH includes CUDA lib?"
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -E "cuda|$CONDA_PREFIX"

# Check if nvcc is found
echo "nvcc location: $(which nvcc 2>/dev/null || echo 'not found')"
nvcc --version || echo "nvcc not working"

# Check if libcublas is visible
echo "libcublas in CUDA_LIB?"
ls -l $CUDA_LIB/libcublas* 2>/dev/null || echo "no libcublas found"

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