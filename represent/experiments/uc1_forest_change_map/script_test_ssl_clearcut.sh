#!/bin/bash

module load Stages/2022



base_folder="/p/project/hai_eo_tree/kuzu/represent/representlib/represent/results/contrastive_learning"
model_s2_clearcut="singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut --input-type 2 --output-layers 1"

# Get a list of folder numbers
folder_numbers=($(ls $base_folder | grep -E '^[0-9]+$'))

counter=0
for folder_num in "${folder_numbers[@]}"; do
    folder_path="$base_folder/$folder_num"
    pth_files=($(ls $folder_path | grep "\.pth$"))

    # Check if the folder is empty or has no .pth files
    if [ ${#pth_files[@]} -eq 0 ]; then
        echo "Folder $folder_num is empty or has no .pth files"
        continue
    fi

    # Extract parent directory name from folder path
    parent_dir=$(basename $folder_path)

    # Loop through S2 files and run Clearcut command
    for pth_file in "${pth_files[@]}"; do
        if [[ $pth_file == *"S2"* ]]; then
            # Extract base filename without extension
            filename=$(basename "$pth_file")
            epoch=$(echo "$filename" | sed -n 's/.*_N0_\([0-9]*\)_model_best\.pth$/\1/p')

            clearcut_out_dir="represent/result/clearcut/contrastive/$parent_dir/$epoch/"
            echo "Starting Clearcut command: $model_s2_clearcut --out-dir $clearcut_out_dir --model-s2-dir $folder_path/$pth_file"
            $model_s2_clearcut --out-dir $clearcut_out_dir --model-s2-dir $folder_path/$pth_file &

            # Increment counter and start a new background process group every 5 iterations
            ((counter++))
            if (( counter % 10 == 0 )); then
                wait
            fi
        fi
    done
done

# Wait for all background processes to finish
wait




