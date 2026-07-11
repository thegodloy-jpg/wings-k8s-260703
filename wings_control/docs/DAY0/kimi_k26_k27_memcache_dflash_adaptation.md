# Kimi K2.6 / K2.7-Code MemCache Offload and DFlash Adaptation

## Scope

This document describes the required adaptation for Kimi models on Ascend vLLM with MemCache KV offload and optional DFlash speculative decoding.

Target model tokens:

- `Kimi-K2.6`
- `Kimi-K2.6-W4A8`
- `Kimi-2.6`
- `Kimi-2.6-W4A8`
- `Kimi-K2.7-Code`

The whitelist and model matching must use lower-case substring tokens and should include both `kimi-k2.6` and `kimi-2.6` spelling forms. The official sample uses `Eco-Tech/Kimi-K2.6-W4A8`, while the product requirement may refer to it as `kimi-2.6`.

Kimi K2.7 Code adaptation scope is limited to `Kimi-K2.7-Code`.

Feature matrix:

| Model | KV offload | Speculative decode | Spec method |
| --- | --- | --- | --- |
| Kimi K2.6 family | supported | supported only when DFlash draft model is provided | `dflash` |
| `Kimi-K2.7-Code` | supported | not supported | none |

Do not route this adaptation through `pd_config.json`. The official Kimi recipe uses `kv_role=kv_both` with `AscendStoreConnector`, not the existing P/D disaggregation producer-consumer path.

## Official Runtime Shape

MemCache is provided by the official image. Do not install or patch LMCache for this Kimi MemCache path.

MemCache setup details are documented in `wings_control/docs/features/memcache/memcache_hybrid_kimi.md`. The adaptation should render visible shell fragments and config files, then join them at the final startup-script assembly layer instead of hiding `ock.mmc.*` config construction inside `vllm_adapter.py`.

Required local MemCache config:

```ini
ock.mmc.meta_service_url = tcp://127.0.0.1:5000
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:6000
ock.mmc.log_level = error
ock.mmc.local_service.world_size = 256
ock.mmc.local_service.protocol = device_rdma
ock.mmc.local_service.dram.size = <page offload memory>GB
```

Required meta service config:

```ini
ock.mmc.meta_service_url = tcp://127.0.0.1:5000
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:6000
ock.mmc.log_level = error
```

The engine process must export:

```bash
export MMC_LOCAL_CONFIG_PATH=<path-to-mmc_local.conf>
```

The meta service process must export:

```bash
export MMC_META_CONFIG_PATH=<path-to-mmc_meta.conf>
python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

The meta service should be treated as a separate service or sidecar. Avoid starting it repeatedly inside the engine retry loop.

The document does not require `pip install memfabric-hybrid` or `pip install memcache-hybrid` at engine startup. The Kimi MemCache path assumes the official image already contains these packages. Startup code should only validate or consume the package when the feature is effective.

Recommended script boundary:

- engine startup script renders `mmc_local.conf` and exports `MMC_LOCAL_CONFIG_PATH`
- master startup script renders `mmc_meta.conf`, exports `MMC_META_CONFIG_PATH`, and runs `MetaService.main()`
- a dedicated MemCache helper should generate these shell fragments
- `wings_entry.py` or the final startup-script assembly layer should join the MemCache fragments with the generated engine script
- `vllm_adapter.py` should not concatenate MemCache fragments and should not own the raw MemCache config text

## KV Offload

Kimi MemCache offload must generate this `kv_transfer_config`:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "lookup_rpc_port": "0",
    "backend": "memcache"
  }
}
```

This is not the LMCache dynamic connector path and not `CPUOffloadingConnector`.

Required behavior:

- when Kimi offload is enabled and memory capacity is valid, generate `mmc_local.conf`
- export `MMC_LOCAL_CONFIG_PATH`
- inject the `AscendStoreConnector` `kv_transfer_config`
- mark `advanced_features.json.features.kv_offload=true`
- skip LMCache package patch and LMCache env generation for this path

When Kimi offload is not valid, remove the whole offload feature:

- do not generate `mmc_local.conf`
- do not export `MMC_LOCAL_CONFIG_PATH`
- do not inject `kv_transfer_config`
- mark `advanced_features.json.features.kv_offload=false`

## Offload Memory Capacity

`ock.mmc.local_service.dram.size` must be linked to the memory offload value configured on the page. The `<page offload memory>` placeholder is not a MemCache-specific default and not a new independent field.

Capacity source:

1. authoritative page/env source: `KV_MEM_OFFLOAD_SIZE`
2. compatibility fallback: `LMCACHE_MAX_LOCAL_CPU_SIZE`, only if the page still passes the value through that legacy field
3. auto mode: `KV_MEM_OFFLOAD_SIZE=auto` with `AVAILABLE_POD_MEM_SIZE`, using the existing auto-capacity calculation

After validation, the resolved page value must render directly into `mmc_local.conf`:

```ini
ock.mmc.local_service.dram.size = <resolved page offload memory>GB
```

Do not keep the official script's `20GB` as a default.

If no valid memory value exists, or `auto` cannot resolve to a valid value, disable the Kimi MemCache offload feature. The current project already has similar behavior for floor-disabled LMCache offload. Kimi MemCache should follow the same effective-feature model instead of silently falling back to `20GB`.

Capacity unit must be explicit:

- the page value is the user-visible total memory offload budget
- LMCache may convert that value to per-card env values in existing code
- MemCache `ock.mmc.local_service.dram.size` should use the page-resolved MemCache local-service budget directly, not accidentally reuse a per-card LMCache value

If implementation reuses `resolve_offload_cpu_capacity_gb()`, the wrapper must document whether the returned value is total budget or per-card budget before writing `mmc_local.conf`.

## Speculative Decode

Kimi K2.6 supports speculative decoding only through DFlash.

Expected config when the user provides a DFlash draft model path:

```json
{
  "method": "dflash",
  "model": "<user-provided-dflash-model-path>",
  "num_speculative_tokens": 15
}
```

Typical draft model path:

```text
z-lab/Kimi-K2.6-DFlash
```

Required behavior:

- add a DFlash detection branch before the generic `draft_model` speculative branch
- identify DFlash draft paths by model path or model id, for example containing `dflash`
- for Kimi K2.6, emit `method=dflash` only when a DFlash path is present
- if Kimi K2.6 has no DFlash path, disable speculative decoding and keep offload only
- do not fall back to suffix speculative decoding for Kimi
- do not enable speculative decoding for `Kimi-K2.7-Code`

The effective advanced feature state must be based on the resolved speculative strategy, not just the raw `enable_speculative_decode` switch. Otherwise the Kimi K2.6 family without a DFlash draft path would be incorrectly reported as speculative enabled.

Current project caveat:

- `config_loader.apply_effective_feature_enablement()` intentionally keeps `enable_speculative_decode=true` for whitelist misses so the adapter can fall back to suffix
- `vllm_adapter.resolve_speculative_strategy()` also falls through to suffix when no draft path exists
- Kimi must add a model-specific exception in both the effective-feature calculation and the adapter strategy path

For Kimi K2.6, `enable_speculative_decode=true` is not enough. The effective spec feature is true only when a DFlash draft model path is present and accepted.

## TP / DP Strategy

Do not hard-code topology only in `ascend_default.json`. Runtime card count still needs to be handled in `vllm_adapter.py` or the existing distributed config path.

Recommended defaults:

| Model | Official example | Runtime rule |
| --- | --- | --- |
| Kimi K2.6 family | `TP=4`, `DP=4` on 16 cards | default `TP=4`, `DP=device_count/4` when divisible |
| `Kimi-K2.7-Code` | `TP=16` on 16 cards | default `TP=device_count`, leave DP unset or `1` |

For the Kimi K2.6 family, if the detected device count cannot be divided by 4, do not inject an invalid DP topology. Either keep explicit user values or disable the automatic recipe with a clear diagnostic.

For `Kimi-K2.7-Code`, preserve the official single-DP shape. Do not reuse the existing Kimi distributed DP default if it would split 16 cards into smaller TP groups.

Current project caveat:

- both Kimi K2.6 and Kimi K2.7 Code are expected to map to `KimiK25ForConditionalGeneration`
- the existing architecture-level default returns `TP=device_count` for Kimi on 8 or 16 cards
- therefore topology cannot be decided by architecture alone
- adapter logic must branch on `model_name` / `model_path` tokens to separate K2.6 from K2.7 Code

## Function Call and Reason Parser

Function calling and reasoning parser fields must be complete for the new Kimi model keys, but both are still controlled by independent switches.

Function call defaults:

```json
{
  "tool_call_parser": "kimi_k2"
}
```

Function call behavior:

- `ascend_default.json` should provide `tool_call_parser=kimi_k2` as model capability metadata
- `enable_auto_tool_choice` is controlled by the page/CLI/env switch and must not be written into the default template
- `config_loader._set_function_call()` keeps these fields only when the function-call trigger is explicitly enabled by user input
- if `enable_auto_tool_choice` is not effective, `tool_call_parser` is removed from the final engine config

Reasoning parser:

- keep reasoning parser ownership in `docs/features/reasoning_parser/reason_parser.yaml`
- add explicit entries for Kimi K2.6 spelling forms
- keep `Kimi-K2.7-Code` mapped to `kimi_k2`
- do not move `reasoning_parser` into `ascend_default.json`

Reasoning behavior:

- `reason_parser.yaml` already provides `KimiK25ForConditionalGeneration.default = kimi_k2`
- explicit Kimi K2.6 rows improve readability and test coverage, but the parser still only reaches the final command when `enable_auto_think_choice` is enabled
- if `enable_auto_think_choice` is false, `config_loader._set_reasoning_parser()` removes `reasoning_parser`

`enable_auto_tool_choice` and `enable_auto_think_choice` are separate controls. Function call parser injection depends on tool-call settings. Reasoning parser injection depends on the thinking/reasoning setting.

## Files to Adapt

### `wings_control/config/smart_feature_whitelist.json`

Add model whitelist rows:

- Kimi K2.6 spelling forms: `spec`, `offload`
- `Kimi-K2.7-Code`: `offload`

Do not add `spec` for `Kimi-K2.7-Code`.

Rows must include `engine=vllm_ascend` and both `910b` / `910c` card tokens if both Ascend platforms are supported. Current whitelist matching is engine + name token + card token, so missing hardware card detection still suppresses the feature.

### `wings_control/utils/model_utils.py`

Add Kimi K2.6 spelling forms to the Kimi architecture mapping.

The current Kimi 2.7 Code entries already map to `KimiK25ForConditionalGeneration`; 2.6 should join the same Kimi family unless a separate architecture is introduced by runtime metadata.

### `wings_control/config/defaults/ascend_default.json`

Add model-specific defaults for:

- Kimi K2.6 spelling forms
- `Kimi-K2.7-Code`

Defaults should preserve:

- `quantization=ascend`
- `gpu_memory_utilization=0.9`
- `trust_remote_code=true` when represented in engine config
- `tool_call_parser=kimi_k2`
- `async_scheduling=true` where supported
- `no_enable_prefix_caching=true` for the official Kimi offload/spec recipe

Do not put fixed `tensor_parallel_size` / `data_parallel_size` here unless the deployment is explicitly single-shape. Topology should be runtime-derived.

Kimi K2.7 Code model-specific defaults should include the official recipe values:

```json
{
  "max_num_seqs": 48,
  "max_model_len": 81920,
  "max_num_batched_tokens": 4096,
  "gpu_memory_utilization": 0.9,
  "async_scheduling": true,
  "additional_config": {
    "enable_npugraph_ex": true,
    "fuse_muls_add": true,
    "multistream_overlap_shared_expert": true
  },
  "compilation_config": {
    "cudagraph_mode": "FULL_DECODE_ONLY"
  }
}
```

Current generic `KimiK25ForConditionalGeneration.default` is not enough for Kimi K2.7 Code because it uses `max_model_len=4096`, `max_num_seqs=16`, and `max_num_batched_tokens=16384`.

### `wings_control/core/config_loader.py`

Required changes:

- special-case Kimi MemCache before the LMCache connector branch
- build `AscendStoreConnector` `kv_transfer_config` for Kimi offload
- resolve Kimi offload memory capacity from page inputs
- disable offload when capacity is missing or invalid
- prevent Kimi MemCache from falling into LMCache patch/env behavior
- keep advanced feature state aligned with the effective offload/spec result
- add Kimi-specific speculative effective-state logic so K2.6 without DFlash does not keep suffix fallback alive

### `wings_control/engines/vllm_adapter.py`

Required changes:

- add Kimi DFlash detection before generic draft-model speculative logic
- prevent Kimi from falling back to suffix speculative decoding
- add Kimi-specific TP/DP defaults:
  - Kimi K2.6 family: `TP=4`, `DP=device_count/4`
  - `Kimi-K2.7-Code`: `TP=device_count`, DP unset or `1`
- expose the offload variant as `memcache`
- keep generating only the vLLM command surface; `AscendStoreConnector` is injected into `engine_config` by `config_loader.py`
- do not concatenate MemCache helper script fragments here
- skip the existing LMCache env/YAML rendering path when the effective offload variant is MemCache
- keep official environment knobs that belong to engine startup, for example `TASK_QUEUE_ENABLE`, `VLLM_ASCEND_ENABLE_FLASHCOMM1`, `HCCL_OP_EXPANSION_MODE`, `PYTORCH_NPU_ALLOC_CONF`, and `VLLM_ASCEND_ENABLE_MLAPO`, if they are not already supplied by the base image or deployment wrapper

Do not bury the `ock.mmc.*` config contents directly in adapter branches. Keep MemCache config rendering in a dedicated helper or deployment layer, and let the final startup-script assembly layer join the returned script text with the engine script. The adapter still needs a guard to avoid emitting LMCache env/YAML for the MemCache variant, because current offload env rendering is adapter-owned.

Do not place host-level tuning blindly into the engine command. Classify official script lines before adapting:

- engine env: safe to export from startup script
- image env: already supplied by image or Ascend setup scripts
- host tuning: `sysctl`, CPU governor, and similar commands require privileged runtime and should be handled by deployment policy, not unconditional engine launch
- optional preload: `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2` should be emitted only when the library exists or the deployment explicitly requests it

### `wings_control/core/wings_entry.py`

Required changes:

- treat the new MemCache offload variant as an active backend
- skip LMCache patch installation for Kimi MemCache
- write `advanced_features.json` from effective feature resolution
- avoid reporting speculative decoding as active when the Kimi K2.6 family has no DFlash draft model
- avoid adding Kimi MemCache to the generic LMCache fallback-removal path unless the fallback path knows how to remove `MMC_LOCAL_CONFIG_PATH` and `AscendStoreConnector`
- call the dedicated MemCache helper when MemCache offload is effective
- join the helper's engine-side prelude with the `script_body` produced from `_build_pid_tracked_script(start_engine_service(merged), ...)`
- prepend the helper's `fallback_cleanup` to advanced-feature fallback commands when MemCache is removed for fallback

### Dedicated MemCache helper

Required behavior:

- render `mmc_local.conf` and the `MMC_LOCAL_CONFIG_PATH` export as an engine-side prelude
- render `mmc_meta.conf` and `start_memcache_master.sh` as master-side artifacts
- read shell templates that own all `ock.mmc.*` parameter text
- accept resolved memory, service URLs, `world_size`, `protocol`, and log level as inputs
- return plain shell text/artifacts to the final assembly layer

The current helper lives under `wings_control/features/kv_offload/memcache/`: `hybrid.py` owns enablement, capacity, and fragment rendering, while `memcache_engine_prelude.sh` and `memcache_master.sh` own the `ock.mmc.*` shell text. `vllm_adapter.py` owns vLLM command generation and only consumes the MemCache helper to skip LMCache env/YAML rendering.

### Deployment or manifest layer

Required changes if the platform should manage MemCache automatically:

- provide `mmc_meta.conf`
- provide `mmc_local.conf` or allow the engine container to generate it in a writable shared path
- start the MemCache MetaService once as a sidecar or separate service
- expose or share the local config path with the engine container through `MMC_LOCAL_CONFIG_PATH`

If this layer is not implemented, Wings can only generate the vLLM command and MemCache prelude artifacts; it cannot guarantee that the MetaService exists.

See `wings_control/docs/features/memcache/memcache_hybrid_kimi.md` for the exact script fragments and enablement preconditions.

### `wings_control/docs/features/reasoning_parser/reason_parser.yaml`

Required changes:

- add explicit `Kimi-2.6` and `Kimi-2.6-W4A8` mappings to `kimi_k2`
- add explicit `Kimi-K2.6` and `Kimi-K2.6-W4A8` mappings to `kimi_k2`
- keep existing `Kimi-K2.7-Code` mapping

### Tests

Add or update tests around:

- Kimi K2.6 with DFlash path: offload plus `method=dflash`
- Kimi K2.6 without DFlash path: offload only, no speculative config and `advanced_features.json.features.speculative_decode=false`
- Kimi K2.7 Code: offload only, no speculative config
- missing offload memory: offload removed, no `MMC_LOCAL_CONFIG_PATH`, no `kv_transfer_config`
- valid offload memory: `ock.mmc.local_service.dram.size` equals the resolved MemCache local-service memory value
- function call parser remains `kimi_k2` only when function calling is effectively enabled
- reasoning parser mapping resolves to `kimi_k2` only when thinking/reasoning parsing is effectively enabled
- Kimi K2.7 Code topology does not inherit a K2.6 `TP=4, DP=device_count/4` recipe
- Kimi K2.6 topology does not inherit the current architecture-level `TP=device_count` Kimi default
- final assembled `start_command.sh` places the MemCache prelude before the engine command
- feature-disabled fallback unsets `MMC_LOCAL_CONFIG_PATH`
- MemCache variant does not emit LMCache env/YAML config

## Expected Command Fragments

### Kimi K2.6 with offload and DFlash

```bash
vllm serve <Kimi-K2.6-model-path> \
  --quantization ascend \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --speculative-config '{"method":"dflash","model":"<dflash-model-path>","num_speculative_tokens":15}' \
  --kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"memcache"}}'
```

### Kimi K2.6 without DFlash draft model

```bash
vllm serve <Kimi-K2.6-model-path> \
  --quantization ascend \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"memcache"}}'
```

No `--speculative-config` should be emitted.

### Kimi-K2.7-Code

```bash
vllm serve <Kimi-K2.7-Code-model-path> \
  --served-model-name kimi_k27 \
  --quantization ascend \
  --tensor-parallel-size <device_count> \
  --max-num-seqs 48 \
  --max-model-len 81920 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 \
  --async-scheduling \
  --additional-config '{"enable_npugraph_ex":true,"fuse_muls_add":true,"multistream_overlap_shared_expert":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lookup_rpc_port":"0","backend":"memcache"}}'
```

No speculative config should be emitted for `Kimi-K2.7-Code`.

## Acceptance Criteria

- Kimi K2.6 appears in the smart feature whitelist with `spec` and `offload`
- Kimi 2.7 Code appears in the smart feature whitelist with `offload` only
- Kimi MemCache offload uses `AscendStoreConnector`
- Kimi MemCache offload memory comes from page configuration
- no `20GB` fallback remains
- missing memory capacity disables offload instead of silently enabling it
- Kimi K2.6 speculative decoding is DFlash only
- Kimi K2.6 without DFlash draft model degrades to offload only
- Kimi 2.7 Code never emits speculative config
- function call parser is `kimi_k2` when function calling is enabled
- reasoning parser resolves to `kimi_k2` when thinking/reasoning parsing is enabled
- PD disaggregation config is not used for this feature
