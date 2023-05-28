# Use Case 1: Forest Change Detection



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
Running Windstorm detection experiments (either on docker or on singularity):
```bash
$ srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:59:00  --cpus-per-task=2 --account={ACCOUNT_NAME} singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_windstorm
$ docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/represent:latest python -m docker run  --init --rm --shm-size=16gb  -v "${PWD}:/app" ridvansalih/represent:latest python -m represent.experiments.uc1_forest_change_map.main_windstorm

```

Running Snow damage detection experiments (either on docker or on singularity):
```bash
$ srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:59:00  --cpus-per-task=2 --account={ACCOUNT_NAME} singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_snowdamage
$ docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/represent:latest python -m docker run  --init --rm --shm-size=16gb  -v "${PWD}:/app" ridvansalih/represent:latest python -m represent.experiments.uc1_forest_change_map.main_snowdamage

```

Running Clear cut detection experiments (either on docker or on singularity):
```bash
$ srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=00:59:00  --cpus-per-task=2 --account={ACCOUNT_NAME} singularity exec --bind "${PWD}:/mnt" --nv  ../represent.sif python -m represent.experiments.uc1_forest_change_map.main_clearcut
$ docker run  --init --rm --shm-size=16gb --gpus device=0  -v "${PWD}:/app" ridvansalih/represent:latest python -m docker run  --init --rm --shm-size=16gb  -v "${PWD}:/app" ridvansalih/represent:latest python -m represent.experiments.uc1_forest_change_map.main_clearcut

```