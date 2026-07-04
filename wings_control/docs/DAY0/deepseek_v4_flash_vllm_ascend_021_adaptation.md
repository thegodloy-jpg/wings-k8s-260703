# DeepSeek-V4-Flash vLLM-Ascend 0.21 Adaptation

## Scope

This adaptation replaces the old vLLM-Ascend 0.18 DeepSeek-V4-Flash startup defaults with the 0.21 A2/A3 recipes while keeping existing non-Flash scenarios unchanged.

Target model:

- `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp`
- architecture: `DeepseekV4ForCausalLM`
- engine: `vllm_ascend` / `vllm_ascend_distributed`

## Default Config

`wings_control/config/defaults/ascend_default.json` should not keep the generic `DeepSeek-V4-Flash` block.

Use two independent keys instead:

- `DeepSeek-V4-Flash-A2`
- `DeepSeek-V4-Flash-A3`

Both keys must contain complete `vllm_ascend` and `vllm_ascend_distributed` sections. Do not put `tensor_parallel_size` or `data_parallel_size` in JSON. Runtime topology is derived in `vllm_adapter.py`:

- A2: default TP = 8, DP = total cards / 8
- A3: default TP = 4, DP = total cards / 4

## A2 Recipe

A2 keeps the 0.21 single-node shape:

- `max_model_len=133120`
- `max_num_batched_tokens=8192`
- `max_num_seqs=32`
- `no_enable_prefix_caching=true`
- `additional_config.enable_dsa_cp=true`
- `additional_config.ascend_compilation_config.enable_npugraph_ex=true`
- `additional_config.ascend_compilation_config.enable_static_kernel=false`
- `additional_config.multistream_overlap_shared_expert=true`

## A3 Recipe

A3 keeps the long-context 0.21 shape:

- `max_model_len=1048576`
- `max_num_batched_tokens=10240`
- `max_num_seqs=64`
- `api_server_count=1`
- no explicit prefix-cache field
- no `enable_dsa_cp`
- `additional_config.ascend_compilation_config.enable_npugraph_ex=true`
- `additional_config.ascend_compilation_config.enable_static_kernel=false`
- `additional_config.multistream_overlap_shared_expert=true`

## Config Selection

Because the runtime model name is still `DeepSeek-V4-Flash-w8a8-mtp`, `config_loader.py` must map it to the platform-specific JSON key.

Platform selection priority:

- `WINGS_ASCEND_PLATFORM=a2|a3`
- `ASCEND_PLATFORM=a2|a3`
- `ENGINE_IMAGE_FLAVOR=a2|a3`
- `ENGINE_VERSION` suffix `-a3`
- `ASCEND_A3_ENABLE=true|1`
- fallback: A2

## Adapter Rules

`vllm_adapter.py` should respect JSON values and only add runtime-only values.

Required behavior:

- do not force `enable_prefix_caching=true`
- do not overwrite `no_enable_prefix_caching=true`
- do not force `additional_config.multistream_overlap_shared_expert=false`
- keep JSON-provided A2 `enable_dsa_cp=true`
- keep JSON-provided `ascend_compilation_config`

## Speculative Decode

Ascend DeepSeek-V4-Flash 0.21 uses:

```json
{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}
```

Therefore the smart-feature whitelist must include DeepSeek-V4-Flash under both:

- `spec`
- `offload`

## LMCache Dynamic Offload

0.21 Flash offload uses LMCache Ascend dynamic connector, not `CPUOffloadingConnector`.

`kv_transfer_config`:

```json
{
  "kv_connector": "LMCacheAscendConnectorV1Dynamic",
  "kv_role": "kv_both",
  "kv_connector_module_path": "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
}
```

Offload env mapping:

- upper switch `ENABLE_KV_MEM_OFFLOAD=true` maps to `LMCACHE_LOCAL_CPU=True`
- upper size `KV_MEM_OFFLOAD_SIZE=40` maps to per-card `LMCACHE_MAX_LOCAL_CPU_SIZE=5` on 8 cards

Default LMCache env:

```bash
export PYTHONHASHSEED=0
export LMCACHE_TRACK_USAGE=false
export LMCACHE_MAX_LOCAL_CPU_SIZE=5
export LMCACHE_LOCAL_CPU=True
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_USE_LAYERWISE=False
export LMCACHE_NUMA_MODE=auto
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_EXTRA_CONFIG='{"save_only_first_rank": false}'
export LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3
```
