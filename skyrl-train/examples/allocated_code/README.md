# DiscoveryBench Training

## Data Preparation

```
cd /data/fan/clone/SkyRL/skyrl-gym/skyrl_gym/envs/allocated_code/discovery_bench
python create_discovery_dataset.py
python discovery_bench/create_test_dataset_with_gt.py
```

## Docker Preparation

```
python generate_compose_100.py -n 120 --types  "executor-prebuilt:120"  -m /data/fan/data_folder:/data:ro

sudo docker compose -f docker-compose.yml up -d --build
```

## Run DiscoveryBench Training
```
cd /data/fan/clone/SkyRL/skyrl-train
bash examples/allocated_code/run_allocated_code.sh
```