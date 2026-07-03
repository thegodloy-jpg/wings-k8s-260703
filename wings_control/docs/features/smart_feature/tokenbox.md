接收参数
[AI Service Log] Starting wings application (sidecar launcher) with args: --model-name Deepseek-v4-Flash --model-path /usr/local/serving/models/ --save-path /opt/wings/outputs --engine vllm --trust-remote-code --dtype auto --kv-cache-dtype fp8 --gpu-memory-utilization 0.8 --enable-chunked-prefill --block-size 16 --max-num-seqs 256 --seed 42 --max-num-batched-tokens 4096 --enable-prefix-caching --port 18000 --input-length 2048 --output-length 2048 --gpu-usage-mode full --device-count 4

生成参数
[AI Service Log] [wings-env] export PYTHONUNBUFFERED=1
[AI Service Log] [wings-cmd] >>> exec vllm serve /usr/local/serving/models/ --attention-backend FLASHMLA_SPARSE_DSV4 --kv-cache-dtype fp8 --block-size 256 --max-model-len 4096 --max-num-batched-tokens 4096 --enable-prefix-caching --no-enable-flashinfer-autotune --gpu-memory-utilization 0.8 --tokenizer-mode deepseek_v4 --compilation-config '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":768}' --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --async-scheduling --host 10.254.83.197 --port 17000 --served-model-name Deepseek-v4-Flash --trust-remote-code --dtype auto --enable-chunked-prefill --max-num-seqs 256 --seed 42 --default-chat-template-kwargs '{"thinking":false}' --tensor-parallel-size 4 --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' wings-control生成启动参数：
[wings-env] export PYTHONUNBUFFERED=1
[wings-env] export VLLM_EARS_TOLERANCE=0.5
[wings-env] export VLLM_DEEP_GEMM_WARMUP=skip
[wings-env] export VLLM_USE_DEEP_GEMM=0
[wings-env] export VLLM_FLASHINFER_MOE_BACKEND=latency
[wings-env] export VLLM_USE_FLASHINFER_MOE_FP4=1
[wings-cmd] >>> exec vllm serve /var/aispace/Qwen3.5-397B-A17B-NVFP4 --trust-remote-code --tool-call-parser qwen3_coder --mm-encoder-tp-mode data --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --kv-cache-dtype fp8 --enable-auto-tool-choice --enable-prefix-caching --kv-offloading-backend native --kv-offloading-size 200 --reasoning-parser qwen3 --host 0.0.0.0 --port 17000 --served-model-name Qwen3.5-397B-A17B-NVFP4 --dtype auto --gpu-memory-utilization 0.9 --max-num-batched-tokens 4096 --block-size 16 --max-num-seqs 32 --seed 0 --default-chat-template-kwargs '{"enable_thinking":true}' --tensor-parallel-size 8

推理引擎启动参数：
non-default args: {'model_tag': '/var/aispace/Qwen3.5-397B-A17B-NVFP4', 'default_chat_template_kwargs': {'enable_thinking': True}, 'enable_auto_tool_choice': True, 'tool_call_parser': 'qwen3_coder', 'host': '0.0.0.0', 'port': 17000, 'model': '/var/aispace/Qwen3.5-397B-A17B-NVFP4', 'trust_remote_code': True, 'served_model_name': ['Qwen3.5-397B-A17B-NVFP4'], 'reasoning_parser': 'qwen3', 'tensor_parallel_size': 8, 'block_size': 16, 'gpu_memory_utilization': 0.9, 'kv_cache_dtype': 'fp8', 'enable_prefix_caching': True, 'kv_offloading_size': 200.0, 'mm_encoder_tp_mode': 'data', 'max_num_batched_tokens': 4096, 'max_num_seqs': 32, 'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 3}}



export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

export PYTHONHASHSEED=0
export LMCACHE_TRACK_USAGE=false
export LMCACHE_MAX_LOCAL_CPU_SIZE="40"
export LMCACHE_LOCAL_CPU="True"
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_USE_LAYERWISE="False"
export LMCACHE_NUMA_MODE="auto"
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_EXTRA_CONFIG='{"save_only_first_rank": false}'
export LMCACHE_LOOKUP_SERVER_WORKER_IDS="0,1,2,3"

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
    --max_model_len 1048576 \
    --max-num-batched-tokens 10240 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --api-server-count 1 \
    --max-num-seqs 64 \
    --data-parallel-size 4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --safetensors-load-strategy 'prefetch' \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --quantization ascend \
    --port 8900 \
    --block-size 128 \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
    --async-scheduling \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex":true,
        "enable_static_kernel":false
        },
    "enable_cpu_binding": true,
    "multistream_overlap_shared_expert":true}' \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}'
    export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

export PYTHONHASHSEED=0
export LMCACHE_TRACK_USAGE=false
export LMCACHE_MAX_LOCAL_CPU_SIZE="40"
export LMCACHE_LOCAL_CPU="True"
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_USE_LAYERWISE="False"
export LMCACHE_NUMA_MODE="auto"
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_EXTRA_CONFIG='{"save_only_first_rank": false}'
export LMCACHE_LOOKUP_SERVER_WORKER_IDS="0,1,2,3"

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V4-Flash-w8a8-mtp \
    --max_model_len 133120 \
    --max-num-batched-tokens 8192 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 32 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --safetensors-load-strategy 'prefetch' \
    --no-enable-prefix-caching \
    --model-loader-extra-config='{"enable_multithread_load": "true", "num_threads": 128}' \
    --quantization ascend \
    --port 8900 \
    --block-size 128 \
    --speculative-config '{"num_speculative_tokens": 1,"method": "mtp","enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
    --async-scheduling \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex":true,
        "enable_static_kernel":false
        },
    "enable_cpu_binding": true,
    "enable_dsa_cp": true,
    "multistream_overlap_shared_expert":true}' \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnectorV1Dynamic","kv_role":"kv_both", "kv_connector_module_path":"lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"}'
