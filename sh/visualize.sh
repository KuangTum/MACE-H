#!/bin/bash
#SBATCH -J MACEH
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

cd ../train/dataset_3/2024-07-18_13-25-29_Bi2Te3_dataset_nosoc/analyze_result/MAE_2024-07-23_17-16-41
${python_path} ../../../../../visual_tools/error_heatmap.py error_info_combined.json
