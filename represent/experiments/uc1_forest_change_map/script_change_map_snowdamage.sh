#!/bin/bash -x

singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_snowdamage --input-type 1 --out-dir represent/result/snowdamage/1/1/ --output-layers 1 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_snowdamage --input-type 1 --out-dir represent/result/snowdamage/1/2/ --output-layers 2 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_snowdamage --input-type 1 --out-dir represent/result/snowdamage/1/3/ --output-layers 3 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_snowdamage --input-type 1 --out-dir represent/result/snowdamage/1/4/ --output-layers 4
