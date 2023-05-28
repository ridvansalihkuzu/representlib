#!/bin/bash -x


singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 1 --out-dir represent/result/clearcut/1/1/  --output-layers 1 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 1 --out-dir represent/result/clearcut/1/2/  --output-layers 2 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 1 --out-dir represent/result/clearcut/1/3/  --output-layers 3 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 1 --out-dir represent/result/clearcut/1/4/  --output-layers 4 &

singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 2 --out-dir represent/result/clearcut/2/1/  --output-layers 1 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 2 --out-dir represent/result/clearcut/2/2/  --output-layers 2 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 2 --out-dir represent/result/clearcut/2/3/  --output-layers 3 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 2 --out-dir represent/result/clearcut/2/4/  --output-layers 4 &

singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 3 --out-dir represent/result/clearcut/3/1/  --output-layers 1 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 3 --out-dir represent/result/clearcut/3/2/  --output-layers 2 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 3 --out-dir represent/result/clearcut/3/3/  --output-layers 3 &
singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 3 --out-dir represent/result/clearcut/3/4/  --output-layers 4