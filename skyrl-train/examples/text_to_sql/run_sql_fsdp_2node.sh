set -x

# NOTE (erictang000): WIP on this - this is still OOMing on 2 8xH100 on 32B model init
# Colocated GRPO training+generation for Qwen2.5-Coder-32B-Instruct on SkyRL-SQL-653 data.
# Uses 2 nodes with 8 GPUs each.
# TODO (sumanthrh): Remove the `resume_mode` and `ckpt_path` arguments before release.
# export WANDB_API_KEY=<your_key_here>
# bash examples/text_to_sql/run_sql_fsdp_2node.sh

DATA_DIR="$HOME/data/sql"


uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-Coder-32B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.sequence_parallel_size=2 \
  trainer.placement.policy_num_nodes=2 \
  trainer.placement.critic_num_nodes=2 \
  trainer.placement.ref_num_nodes=2 \
  trainer.placement.reward_num_nodes=2 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=4 \
  trainer.train_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=6000 \
  generator.max_input_length=29000 \
  generator.sampling_params.max_generate_length=3000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy_mini_batch_size=64 \
  trainer.algorithm.use_kl_loss=true \
  trainer.ckpt_interval=10 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=text2sql \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.7 \
  generator.max_turns=5 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  trainer.logger="wandb" \
  trainer.project_name="skyrlsql" \
  trainer.run_name="skyrlsql_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/sql_32B_ckpt"
