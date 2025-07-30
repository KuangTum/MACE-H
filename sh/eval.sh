#!/bin/bash
#SBATCH -J MACEH
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=3850
#SBATCH --cpus-per-task=1
#SBATCH --output=%j.out
#SBATCH --partition=compute
#SBATCH --account=su007-rjm

module purge
module load GCC/13.2.0 CUDA/11.8.0

python_path="/home/c/chenqian3/.conda/envs/DeepH/bin/python"

cd ..
${python_path} ./deephe3-eval.py ./configs/eval.ini | tee -a sh/log_eval.txt