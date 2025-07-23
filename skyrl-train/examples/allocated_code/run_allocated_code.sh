#!/bin/bash
set -x

# Allocated Code training with GRPO
# Make sure your container manager is running at localhost:5000
# Usage: bash examples/allocated_code/run_allocated_code.sh

DATA_DIR="/data/fede/SkyRL/skyrl-train/skyrl-gym/skyrl_gym/envs/allocated_code/data/allocated_code"

# Container allocation math:
# train_batch_size * n_samples_per_prompt = total concurrent containers needed
# 2 * 1 = 2 containers (well within your 10 available)

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=2 \
  trainer.placement.ref_num_gpus_per_node=2 \
  generator.num_inference_engines=2 \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=2 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=512 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=allocated_code \
  +environment.skyrl_gym.allocated_code.manager_url="http://localhost:5000" \
  generator.n_samples_per_prompt=2 \
  generator.max_turns=3 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  generator.gpu_memory_utilization=0.7 \
  trainer.logger="console" \
  trainer.project_name="allocated_code" \
  trainer.run_name="allocated_code_test" \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  $@ 