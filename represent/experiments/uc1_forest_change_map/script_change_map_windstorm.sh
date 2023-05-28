#!/bin/bash -x

singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 1 --out-dir represent/result/windstorm/1/1/ --output-layers 1 --n-trials 3 --threshold-steps 25 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 1 --out-dir represent/result/windstorm/1/2/ --output-layers 2 --n-trials 3 --threshold-steps 25 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 1 --out-dir represent/result/windstorm/1/3/ --output-layers 3 --n-trials 3 --threshold-steps 25 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 1 --out-dir represent/result/windstorm/1/4/ --output-layers 4 --n-trials 3 --threshold-steps 25 &

singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 2 --out-dir represent/result/windstorm/2/1/ --output-layers 1 --n-trials 3 --threshold-steps 25 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 2 --out-dir represent/result/windstorm/2/2/ --output-layers 2 --n-trials 3 --threshold-steps 25 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 2 --out-dir represent/result/windstorm/2/3/ --output-layers 3 --n-trials 3 --threshold-steps 25 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 2 --out-dir represent/result/windstorm/2/4/ --output-layers 4 --n-trials 3 --threshold-steps 25 &

singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 3 --out-dir represent/result/windstorm/3/1/ --output-layers 1 --n-trials 3 --threshold-steps 25 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 3 --out-dir represent/result/windstorm/3/2/ --output-layers 2 --n-trials 3 --threshold-steps 25 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 3 --out-dir represent/result/windstorm/3/3/ --output-layers 3 --n-trials 3 --threshold-steps 25 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm --input-type 3 --out-dir represent/result/windstorm/3/4/ --output-layers 4 --n-trials 3 --threshold-steps 25