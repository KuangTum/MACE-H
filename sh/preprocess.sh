#!/bin/bash
#SBATCH -J DeepH-E3
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=3850
#SBATCH --cpus-per-task=1
#SBATCH --output=%j.out
#SBATCH --partition=compute
#SBATCH --account=su007-rjm

module purge
module load GCC/13.2.0 CUDA/11.8.0 
python_path="/home/c/chenqian3/.conda/envs/DeepH/bin/python"

cd ..
${python_path} ./deephe3-preprocess.py ./configs/preprocess.ini | tee -a sh/log_preprocess.txt