#!/bin/bash
# ===== SLURM SETTINGS =====
#SBATCH --job-name=train_groove
#SBATCH --output=Log/%x_%j.out
#SBATCH --error=Log/%x_%j.err
#SBATCH --time=5-00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:4
#SBATCH --chdir=/data/fhinterberger/GroovePal  # alternative to manual cd

set -euo pipefail
mkdir -p Log

echo "[$(date)] Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "CPUs/task=${SLURM_CPUS_PER_TASK:-?}  GRES=${SLURM_JOB_GRES:-?}  CVD=${CUDA_VISIBLE_DEVICES:-unset}"

# Threading limits (avoid oversubscription)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

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