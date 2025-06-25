set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-7B-Instruct on SkyRL-SQL-653 data.
# Uses 1 node with 8 GPUs.
# export WANDB_API_KEY=<your_key_here>
# bash examples/text_to_sql/run_sql_fsdp.sh

DATA_DIR="data/sql/"
train_data="['${DATA_DIR}/train.parquet']"
val_data="['${DATA_DIR}/validation.parquet']"
NUM_GPUS=8
NUM_INFERENCE_ENGINES=2
TP_SIZE=4
MAX_INPUT_LENGTH=29000
MAX_GENERATE_LENGTH=3000
TRAIN_BATCH_SIZE=256

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data=$train_data \
  data.val_data=$val_data \
  trainer.policy.model.path="Qwen/Qwen2.5-Coder-7B-Instruct" \
  trainer.epochs=50 \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.sequence_parallel_size=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=6000 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy_mini_batch_size=256 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.ppo_loss_type="dual_clip" \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.export_path=/mnt/cluster_storage/export/ \
  trainer.dump_data_batch=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=text2sql \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.7 \
  generator.max_turns=6 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  trainer.seed=1234 \
  trainer.logger="wandb" \
  trainer.project_name="skyrlsql" \
  trainer.run_name="skyrlsql_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/sql_32B_ckpt"