#!/usr/bin/bash
#SBATCH --job-name=BIS-MC
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%j.out
#SBATCH --constraint="CPU_GEN:MLN|CPU_GEN:GEN"

module load py-numpy/1.24.2_py39
module load py-scipy/1.10.1_py39

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"
echo

seed_args=()
if [[ -n "${SEED:-}" ]]; then
  echo "Using fixed --seed=$SEED for reproducibility only."
  echo "Do not use a fixed public seed for an operational EVR deployment."
  seed_args=(--seed "$SEED")
else
  echo "No SEED supplied; the verifier will use fresh OS entropy."
fi
echo

python3 -u bis_calibration.py \
  --eps "$EPS" \
  --delta "$DELTA" \
  --T "$TVAL" \
  --k "$KVAL" \
  --initial-sigma "$INITIAL_SIGMA" \
  --samples-per-sigma "$SAMPLES_PER_SIGMA" \
  --directions "${DIRECTIONS:-both}" \
  --sigma-step "$SIGMA_STEP" \
  "${seed_args[@]}"
