#!/bin/bash
set -e

# Fine-tuned Model Test Evaluation
# This script evaluates the fine-tuned model after training
# Usage: bash examples/allocated_code/run_finetuned_model_evaluation.sh [checkpoint_step]

echo "=== Fine-tuned Model Test Evaluation ==="

# Parse checkpoint step argument (default to 50)
CHECKPOINT_STEP=${1:-75}
echo "Using checkpoint: global_step_$CHECKPOINT_STEP"

if [ -z "${TOGETHER_API_KEY}" ] || [ -z "${WANDB_API_KEY}" ]; then
    echo "Error: TOGETHER_API_KEY and WANDB_API_KEY must be set"
    exit 1
fi

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "Error: OPENAI_API_KEY must be set for DiscoveryBench native evaluation"
    echo "The DiscoveryBench evaluation script uses GPT-5 for hypothesis comparison"
    exit 1
fi

# Configuration
DATA_DIR="/data/fan/SkyRL/skyrl-gym/skyrl_gym/envs/allocated_code/discovery_bench/data"
CHECKPOINT_PATH="/data/fan/SkyRL/checkpoints/skyrl-allocated-code_simple"
FINETUNED_MODEL_PATH="$CHECKPOINT_PATH/exports/global_step_$CHECKPOINT_STEP/policy"
DISCOVERY_EVAL_SCRIPT="/data/fan/discoverybench/discovery_eval.py"
OUTPUT_DIR="/data/fan/SkyRL/finetuned_model_evaluation_results_step_$CHECKPOINT_STEP"
PREDICTIONS_DIR="/data/fan/skyrl_finetuned_model_predictions_step_$CHECKPOINT_STEP"

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "$DATA_DIR/discovery_test_with_gt.parquet" ]; then
    echo "Error: Test dataset with ground truth not found"
    echo "Creating test dataset with ground truth..."
    cd /data/fan/SkyRL/skyrl-gym/skyrl_gym/envs/allocated_code/discovery_bench
    python create_test_dataset_with_gt.py
    cd -
fi

if [ ! -d "$FINETUNED_MODEL_PATH" ]; then
    echo "Error: Fine-tuned model not found at $FINETUNED_MODEL_PATH"
    echo "Available checkpoints:"
    ls -la "$CHECKPOINT_PATH/exports/" 2>/dev/null || echo "No exports directory found"
    exit 1
fi

if [ ! -f "$DISCOVERY_EVAL_SCRIPT" ]; then
    echo "Error: DiscoveryBench evaluation script not found at $DISCOVERY_EVAL_SCRIPT"
    echo "Please ensure discovery_eval.py is available"
    exit 1
fi


# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PREDICTIONS_DIR"

# Clean previous predictions
echo "Cleaning previous predictions..."
rm -f "$PREDICTIONS_DIR"/prediction_*.json

echo "=== Step 1: Running Fine-tuned Model Evaluation ==="
echo "Model: $FINETUNED_MODEL_PATH"
echo "Test Dataset: $DATA_DIR/discovery_test_with_gt.parquet"
echo "Output: $OUTPUT_DIR/"

# Run evaluation with FINE-TUNED MODEL
export RAY_worker_register_timeout_seconds=300
export SKYRL_TEST_OUTPUT_DIR="$PREDICTIONS_DIR"

uv run --isolated --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/discovery_test_with_gt.parquet']" \
  data.val_data="['$DATA_DIR/discovery_test_with_gt.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=12 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path="$FINETUNED_MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.gpu_memory_utilization=0.8 \
  trainer.epochs=0 \
  trainer.update_epochs_per_batch=0 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=5096 \
  generator.sampling_params.max_generate_length=1524 \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=1 \
  generator.max_turns=10 \
  generator.sampling_params.temperature=0 \
  generator.sampling_params.top_p=1.0 \
  environment.env_class=allocated_code_test \
  +environment.skyrl_gym.test_output_dir="$PREDICTIONS_DIR" \
  environment.skyrl_gym.max_env_workers=2 \
  +environment.skyrl_gym.allocated_code.manager_url="http://localhost:5000" \
  trainer.logger="wandb" \
  trainer.project_name="skyrl-discovery" \
  trainer.run_name="finetuned_model_evaluation_step_$CHECKPOINT_STEP" \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.max_ckpts_to_keep=1 \
  trainer.resume_mode=none \
  trainer.eval_batch_size=4 \
  trainer.eval_before_train=true \
  trainer.eval_interval=1 \
  generator.eval_sampling_params.temperature=0 \
  trainer.export_path="$OUTPUT_DIR/skyrl_exports" \
  trainer.dump_eval_results=true

echo "=== Step 2: Running DiscoveryBench Native Evaluation ==="

# Check if predictions were saved
if [ ! -d "$PREDICTIONS_DIR" ] || [ -z "$(ls -A "$PREDICTIONS_DIR"/prediction_*.json 2>/dev/null)" ]; then
    echo "Error: No prediction files found in $PREDICTIONS_DIR"
    echo "Fine-tuned model evaluation may have failed"
    exit 1
fi

echo "Found prediction files in: $PREDICTIONS_DIR"
echo "Running DiscoveryBench native evaluation..."

# Run DiscoveryBench native evaluation
python /data/fan/SkyRL/skyrl-train/examples/allocated_code/run_discoverybench_native_eval.py \
  --predictions_dir "$PREDICTIONS_DIR" \
  --discovery_eval_script "$DISCOVERY_EVAL_SCRIPT" \
  --base_dir "/data/fan" \
  --output_dir "$OUTPUT_DIR"

echo "=== Fine-tuned Model Evaluation Complete ==="
echo "Results: $OUTPUT_DIR/"
echo ""
echo "Check the following files for detailed results:"
echo "- $OUTPUT_DIR/discoverybench_native_evaluation.json (DiscoveryBench native evaluation)"
echo "- $OUTPUT_DIR/evaluation_comparison.csv (Comparison of SkyRL vs DiscoveryBench scores)"
echo "- $OUTPUT_DIR/skyrl_exports/dumped_evals/ (SkyRL evaluation results)"
echo "- $PREDICTIONS_DIR/ (Individual prediction files)"
