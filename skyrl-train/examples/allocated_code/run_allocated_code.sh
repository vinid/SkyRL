#!/bin/bash
set -x

# # Limit to only use first 4 GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ -z "${TOGETHER_API_KEY}" ] || [ -z "${WANDB_API_KEY}" ]; then
    echo "Error: TOGETHER_API_KEY and WANDB_API_KEY must be set"
    exit 1
fi
# Allocated Code training with GRPO
# Make sure your container manager is running at localhost:5000
# export WANDB_API_KEY=
# export TOGETHER_API_KEY=
# Usage: bash examples/allocated_code/run_allocated_code.sh

DATA_DIR="/data/fede/SkyRL/skyrl-gym/skyrl_gym/envs/allocated_code/data/allocated_code"

# Container allocation math:
# train_batch_size * n_samples_per_prompt = total concurrent containers needed  
# Under 50: 12 * 4 = 48 containers


# ENV_DIR="/data/fan/skyrl_uv_env"

# if [ ! -d "$ENV_DIR" ]; then
#     echo ">>> Creating UV virtual environment at $ENV_DIR ..."
#     uv venv "$ENV_DIR"

#     echo ">>> Installing dependencies into UV environment..."
#     uv pip install --python "$ENV_DIR/bin/python" -e "/data/fan/SkyRL/skyrl-train[vllm]"
#     uv pip install --python "$ENV_DIR/bin/python" -e /data/fan/SkyRL/skyrl-gym
# fi

export RAY_worker_register_timeout_seconds=300

# "$ENV_DIR/bin/python" -m skyrl_train.entrypoints.main_base \
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
  trainer.train_batch_size=40 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
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
  environment.skyrl_gym.max_env_workers=4 \
  +environment.skyrl_gym.allocated_code.manager_url="http://localhost:5000" \
  trainer.logger="wandb" \
  trainer.project_name="skyrl-discovery" \
  trainer.run_name="discovery-4b" \
  trainer.ckpt_interval=25 \
  trainer.hf_save_interval=25 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="/data/fede/SkyRL/checkpoints/skyrl-discovery-4b" \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=true \
  generator.eval_sampling_params.temperature=0 \
  trainer.export_path="/data/fede/SkyRL/checkpoints/skyrl-discovery-4b/exports" \
  trainer.eval_interval=10 \
  $@ 
