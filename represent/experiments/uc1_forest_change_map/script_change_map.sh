#!/bin/bash -x

module load Stages/2022

salloc --nodes=1 --ntasks-per-node=12 --gres=mem512 --time=12:09:00 --account=hai_eo_tree
srun --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=48 --time=23:09:00 --account=hai_eo_tree bash represent/experiments/uc1_forest_change_map/script_change_map_windstorm.sh
srun --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=48 --time=23:09:00 --account=hai_eo_tree bash represent/experiments/uc1_forest_change_map/script_change_map_windstorm.sh
srun --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=48 --time=23:09:00 --account=hai_eo_tree bash represent/experiments/uc1_forest_change_map/script_change_map_windstorm.sh

srun --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=8 --time=23:09:00 --account=hai_eo_tree bash represent/experiments/uc1_forest_change_map/script_change_map_snowdamage.sh
srun --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=24 --time=23:09:00 --account=hai_eo_tree bash represent/experiments/uc1_forest_change_map/script_change_map_clearcut.sh

srun --nodes=1 --ntasks-per-node=1 --gres=gpu:4 --cpus-per-task=24 --time=23:09:00 --account=hai_eo_tree bash represent/experiments/uc1_contrastive_learning/script.sh
