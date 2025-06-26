# Launches vllm server for Qwen2.5-1.5B-Instruct on 4 GPUs.
# bash examples/remote_inference_engine/run_vllm_server.sh
set -x

uv run --isolated --extra vllm -m skyrl_train.inference_engines.vllm.vllm_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --tensor-parallel-size 4 \
    --host 127.0.0.1 \
    --port 8001 \
    --seed 42 \
    --max-model-len 4096 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-sleep-mode \
    --max-num_batched_tokens 8192 \
    --max-num-seqs 1024 \
    --trust-remote-code \
    --distributed-executor-backend ray \
    --worker-extension-cls skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap