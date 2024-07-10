# Use Case 3: Building Anomaly Detection



## Running the Experiments


If you want to run the project on docker containers, pull the customly built images for this project:
```bash
$ docker pull ridvansalih/represent:latest
```

If you want to run the project on singularity containers, convert the docker images to singularity ones as described below:
```bash
$ export SINGULARITY_CACHEDIR=$(mktemp -d -p ${PWD})
$ export SINGULARITY_TMPDIR=$(mktemp -d -p ${PWD})
$ singularity pull represent_lateset.sif docker://represent:latest

```
Running LSTM-autoencoder experiments (either on docker or on singularity):
```bash
$ srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:59:00  --cpus-per-task=2 --account={ACCOUNT_NAME} singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc3_building_anomaly_detection.main_AE.py --n-trials 5 --loss-type 'dtw'  --gamma 0.1  -n -p --method 'lae'  --train-dir represent/data/UC3/database_train.csv --test-dir represent/data/UC3/database_test.csv --eval-dir UC3/data/database_synthetic_test.csv --log-dir represent/results/lae/
$ docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/represent:latest python -m represent.experiments.uc3_building_anomaly_detection.main_AE --n-trials 5 --loss-type 'dtw'  --gamma 0.1  -n -p --method 'lae'  --train-dir represent/data/UC3/database_train.csv --test-dir represent/data/UC3/database_test.csv --eval-dir represent/data/UC3/valdataset_3_3_3.csv --log-dir represent/result/lae/ --learning-rate 5e-4 --smooth-window 5
```

Running Autoencoder experiments (either on docker or on singularity):
```bash
$ srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:59:00  --cpus-per-task=2 --account={ACCOUNT_NAME} singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc3_building_anomaly_detection.main_AE --n-trials 5 --loss-type 'dtw'  --gamma 0.1  -n -p --method 'ae'  --train-dir represent/data/UC3/database_train.csv --test-dir represent/data/UC3/database_test.csv --eval-dir represent/data/UC3/database_synthetic_test.csv --log-dir represent/results/ae/
$ docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/represent:latest python -m represent.experiments.uc3_building_anomaly_detection.main_AE --n-trials 5 --loss-type 'l1'  --gamma 0.1  -n -p --method 'ae'  --train-dir represent/data/UC3//database_train.csv --test-dir represent/data/UC3/database_test.csv --eval-dir represent/data/UC3/database_synthetic_test.csv --log-dir represent/results/ae/
```


Running GANF experiments (either on docker or on singularity):
```bash
$ srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:59:00  --cpus-per-task=2 --account={ACCOUNT_NAME} singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc3_building_anomaly_detection.main_GANF --n-trials 1 -n --learning-rate 0.0001 --penalty-rate 0.1 --smooth_window 7 --embedding-dim 32 --batch-size 1024 --total-epoch 90 --method 'ganf' --train-dir represent/data/UC3/database_train.csv --test-dir represent/data/UC3/database_test.csv --eval-dir represent/data/UC3/valdataset_3_3_3.csv --log-dir represent/results/ganf/
$ docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/represent:latest python -m represent.experiments.uc3_building_anomaly_detection.main_GANF --n-trials 1 -n --learning-rate 0.0001 --penalty-rate 0.1 --smooth_window 7 --embedding-dim 32 --batch-size 1024 --total-epoch 90 --method 'ganf' --train-dir represent/data/UC3/database_train.csv --test-dir represent/data/UC3/database_test.csv --eval-dir represent/data/UC3/valdataset_3_3_3.csv --log-dir represent/results/ganf/
```

Running RRCF experiments (either on docker or on singularity):
```bash
$ srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:59:00  --cpus-per-task=2 --account={ACCOUNT_NAME} singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc3_building_anomaly_detection.main_ML --n-trials 1 -n --smooth_window 5 --batch-size 1024 --method 'rrcf' --train-dir represent/data/UC3/database_train.csv --test-dir represent/data/UC3/database_test.csv --eval-dir represent/data/UC3/valdataset_3_3_3.csv --log-dir represent/results/rrcf/
$ docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/represent:latest python -m represent.experiments.uc3_building_anomaly_detection.main_ML --n-trials 1 -n --smooth_window 5 --batch-size 1024 --method 'rrcf' --train-dir represent/data/UC3/database_train.csv --test-dir represent/data/UC3/database_test.csv --eval-dir represent/data/UC3/valdataset_3_3_3.csv --log-dir represent/results/rrcf/
```

Running MAXDIV experiments (either on docker or on singularity):
```bash
$ srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:59:00  --cpus-per-task=2 --account={ACCOUNT_NAME} singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc3_building_anomaly_detection.main_ML.py --n-trials 1 -n --smooth_window 7 --batch-size 48 --train-dir UC3/data/database_train.csv --test-dir UC3/data/database_test.csv --eval-dir UC3/data/valdataset_3_3_3.csv --log-dir UC3/results/maxdiv/01/ --method "maxdiv"
$ docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/represent:latest python -m represent.experiments.uc3_building_anomaly_detection.main_ML.py --n-trials 1 -n --smooth_window 7 --batch-size 48 --train-dir UC3/data/database_train.csv --test-dir UC3/data/database_test.csv --eval-dir UC3/data/valdataset_3_3_3.csv --log-dir UC3/results/maxdiv/01/ --method "maxdiv"
```

