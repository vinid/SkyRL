#!/bin/bash
set -x

# Allocated Code training with GRPO
# Make sure your container manager is running at localhost:5000
# export WANDB_API_KEY=<your_key_here>
# Usage: bash examples/allocated_code/run_allocated_code.sh

DATA_DIR="/data/fede/SkyRL/skyrl-train/skyrl-gym/skyrl_gym/envs/allocated_code/data/allocated_code"

# Container allocation math:
# train_batch_size * n_samples_per_prompt = total concurrent containers needed  
# Under 50: 12 * 4 = 48 containers

uv run --isolated --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=12 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="Qwen/Qwen2.5-3B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=2 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.gpu_memory_utilization=0.5 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=12 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=1024 \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=4 \
  generator.max_turns=4 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  environment.env_class=allocated_code \
  environment.skyrl_gym.max_env_workers=8 \
  +environment.skyrl_gym.allocated_code.manager_url="http://localhost:5000" \
  trainer.logger="wandb" \
  trainer.project_name="skyrl-allocated-code" \
  trainer.run_name="allocated_code_simple" \
  trainer.ckpt_interval=10 \
  trainer.hf_save_interval=50 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/skyrl-allocated-code_simple" \
  trainer.eval_batch_size=12 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  trainer.export_path="$HOME/skyrl-allocated-code_simple/exports" \
  trainer.eval_interval=10 \
  $@ 