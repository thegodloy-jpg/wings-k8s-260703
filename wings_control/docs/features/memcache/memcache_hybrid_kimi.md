# MemCache Hybrid Kimi Offload Script Contract

## Scope

This document defines the MemCache Hybrid startup contract for Kimi models.

MemCache setup must be represented as script fragments and environment exports that can be appended by the final `start_command.sh` assembly layer. It must not be embedded as hidden `vllm_adapter.py` logic.

Target usage:

- Kimi K2.6 family: MemCache KV offload, optionally combined with DFlash speculative decoding
- `Kimi-K2.7-Code`: MemCache KV offload only

The vLLM connector is still an engine config / CLI concern:

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

The MemCache runtime config is a separate script/env concern. Runtime shell
templates live under `wings_control/features/memcache/`, not under this docs
directory:

- `mmc_local.conf`
- `mmc_meta.conf`
- `MMC_LOCAL_CONFIG_PATH`
- `MMC_META_CONFIG_PATH`
- `start_memcache_master.sh`
- `wings_control/features/memcache/memcache_engine_prelude.sh`
- `wings_control/features/memcache/memcache_master.sh`

## Current Project Assembly Point

Current startup flow:

```text
wings_entry.py
  -> start_engine_service(merged)
     -> engine_manager.py
        -> vllm_adapter.build_start_script()
  -> _build_pid_tracked_script(...)
  -> _assemble_startup_command(...)
     -> final /shared-volume/start_command.sh
```

Therefore MemCache fragments should be appended in `wings_entry.py`, at the final assembly layer, before the engine body is executed.

Recommended assembly order:

```text
standard preamble
env_overrides
accel preamble
memcache engine prelude
engine script body
monitor / fallback script
```

Do not append MemCache fragments inside `vllm_adapter.py`. The adapter should keep generating only the vLLM command and vLLM-related environment.

There is one negative responsibility in `vllm_adapter.py`: when the effective offload variant is MemCache, the existing LMCache env/YAML rendering path must be skipped. This prevents `ENABLE_KV_OFFLOAD=true` from also emitting LMCache-specific environment for a MemCache deployment.

## Fragment API

Use `wings_control/features/memcache/hybrid.py` to read the shell templates and
return plain strings. A concrete implementation can use a dataclass, dict, or
tuple, but it should expose these logical fields:

```python
{
    "enabled": bool,
    "engine_prelude": str,
    "fallback_cleanup": str,
    "master_script": str,
    "env": dict,
}
```

Field meanings:

- `enabled`: whether MemCache offload is actually effective
- `engine_prelude`: shell text inserted before the engine command in `start_command.sh`
- `fallback_cleanup`: shell text prepended to advanced-feature fallback commands when offload is disabled for fallback
- `master_script`: standalone script text for starting the MemCache MetaService
- `env`: resolved environment values, useful for logs/tests

If `enabled=false`, all script strings should be empty and `env` should be empty.

## Enablement Preconditions

MemCache is effective only when all conditions are true:

- engine is `vllm_ascend`
- model matches the Kimi K2.6 token set or exact `Kimi-K2.7-Code` token
- offload is requested by page/env
- model hits `offload` in `wings_control/config/smart_feature_whitelist.json`
- memory offload resolves to a valid positive value
- official image contains `memfabric-hybrid` and `memcache-hybrid`
- MetaService is provided by sidecar, service, or generated master script

If any condition fails, remove the MemCache offload path:

- no `mmc_local.conf`
- no `MMC_LOCAL_CONFIG_PATH`
- no `AscendStoreConnector`
- `advanced_features.json.features.kv_offload=false`

Do not use the official sample `20GB` as a fallback.

## Environment Contract

The helper should resolve these values and render them into script text from
`memcache_engine_prelude.sh`:

```bash
export WINGS_MEMCACHE_DIR="${WINGS_MEMCACHE_DIR:-/shared-volume/memcache}"
export WINGS_MEMCACHE_META_SERVICE_URL="${WINGS_MEMCACHE_META_SERVICE_URL:-tcp://127.0.0.1:5000}"
export WINGS_MEMCACHE_CONFIG_STORE_URL="${WINGS_MEMCACHE_CONFIG_STORE_URL:-tcp://127.0.0.1:6000}"
export WINGS_MEMCACHE_LOG_LEVEL="${WINGS_MEMCACHE_LOG_LEVEL:-error}"
export WINGS_MEMCACHE_WORLD_SIZE="${WINGS_MEMCACHE_WORLD_SIZE:-256}"
export WINGS_MEMCACHE_PROTOCOL="${WINGS_MEMCACHE_PROTOCOL:-device_rdma}"
export WINGS_MEMCACHE_DRAM_GB="<resolved_memcache_memory>"
```

`WINGS_MEMCACHE_DRAM_GB` has no default. It must come from the page-owned memory offload value after validation.

In the current env plumbing, the page-owned source is `KV_MEM_OFFLOAD_SIZE`. `LMCACHE_MAX_LOCAL_CPU_SIZE` is only a compatibility fallback if the page still passes the value through that legacy field. The resolved page value is rendered 1:1 into `ock.mmc.local_service.dram.size = ${WINGS_MEMCACHE_DRAM_GB}GB`; do not introduce a separate MemCache capacity input and do not default to `20GB`.

The generated engine prelude must export:

```bash
export MMC_LOCAL_CONFIG_PATH="${WINGS_MEMCACHE_DIR}/mmc_local.conf"
```

The generated master script must export:

```bash
export MMC_META_CONFIG_PATH="${WINGS_MEMCACHE_DIR}/mmc_meta.conf"
```

The `127.0.0.1` defaults are valid only when engine and MetaService share a network namespace. For a separate pod/service, use service DNS or IP values through `WINGS_MEMCACHE_META_SERVICE_URL` and `WINGS_MEMCACHE_CONFIG_STORE_URL`.

## Engine Prelude

This is the script fragment that should be inserted into final `start_command.sh` before the engine command.
The runtime source of this text is:

```text
wings_control/features/memcache/memcache_engine_prelude.sh
```

```bash
# --- wings-memcache: engine prelude ---
export WINGS_MEMCACHE_DIR="${WINGS_MEMCACHE_DIR:-/shared-volume/memcache}"
export WINGS_MEMCACHE_META_SERVICE_URL="${WINGS_MEMCACHE_META_SERVICE_URL:-tcp://127.0.0.1:5000}"
export WINGS_MEMCACHE_CONFIG_STORE_URL="${WINGS_MEMCACHE_CONFIG_STORE_URL:-tcp://127.0.0.1:6000}"
export WINGS_MEMCACHE_LOG_LEVEL="${WINGS_MEMCACHE_LOG_LEVEL:-error}"
export WINGS_MEMCACHE_WORLD_SIZE="${WINGS_MEMCACHE_WORLD_SIZE:-256}"
export WINGS_MEMCACHE_PROTOCOL="${WINGS_MEMCACHE_PROTOCOL:-device_rdma}"
export WINGS_MEMCACHE_DRAM_GB="<resolved_memcache_memory>"

mkdir -p "${WINGS_MEMCACHE_DIR}"
cat > "${WINGS_MEMCACHE_DIR}/mmc_local.conf" <<EOF
ock.mmc.meta_service_url = ${WINGS_MEMCACHE_META_SERVICE_URL}
ock.mmc.local_service.config_store_url = ${WINGS_MEMCACHE_CONFIG_STORE_URL}
ock.mmc.log_level = ${WINGS_MEMCACHE_LOG_LEVEL}
ock.mmc.local_service.world_size = ${WINGS_MEMCACHE_WORLD_SIZE}
ock.mmc.local_service.protocol = ${WINGS_MEMCACHE_PROTOCOL}
ock.mmc.local_service.dram.size = ${WINGS_MEMCACHE_DRAM_GB}GB
EOF
export MMC_LOCAL_CONFIG_PATH="${WINGS_MEMCACHE_DIR}/mmc_local.conf"
cat > "${WINGS_MEMCACHE_DIR}/start_memcache_master.sh" <<'WINGS_MEMCACHE_MASTER'
<contents of wings_control/features/memcache/memcache_master.sh>
WINGS_MEMCACHE_MASTER
chmod +x "${WINGS_MEMCACHE_DIR}/start_memcache_master.sh"
# --- end wings-memcache: engine prelude ---
```

This fragment is directly concatenable with `script_body` in `wings_entry.py`.

## Master Script

This is a standalone script artifact. It should not be inserted into the vLLM retry loop.
The runtime source of this text is:

```text
wings_control/features/memcache/memcache_master.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail

export WINGS_MEMCACHE_DIR="${WINGS_MEMCACHE_DIR:-/shared-volume/memcache}"
export WINGS_MEMCACHE_META_SERVICE_URL="${WINGS_MEMCACHE_META_SERVICE_URL:-tcp://127.0.0.1:5000}"
export WINGS_MEMCACHE_CONFIG_STORE_URL="${WINGS_MEMCACHE_CONFIG_STORE_URL:-tcp://127.0.0.1:6000}"
export WINGS_MEMCACHE_LOG_LEVEL="${WINGS_MEMCACHE_LOG_LEVEL:-error}"

mkdir -p "${WINGS_MEMCACHE_DIR}"
cat > "${WINGS_MEMCACHE_DIR}/mmc_meta.conf" <<EOF
ock.mmc.meta_service_url = ${WINGS_MEMCACHE_META_SERVICE_URL}
ock.mmc.meta_service.config_store_url = ${WINGS_MEMCACHE_CONFIG_STORE_URL}
ock.mmc.log_level = ${WINGS_MEMCACHE_LOG_LEVEL}
EOF
export MMC_META_CONFIG_PATH="${WINGS_MEMCACHE_DIR}/mmc_meta.conf"

exec python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

Recommended artifact path:

```text
/shared-volume/memcache/start_memcache_master.sh
```

The launcher may write this script for operators or a sidecar to run. It must not execute it inside the main vLLM engine start/retry path.

## Fallback Cleanup

Current Wings fallback logic can start a second command in the same shell after advanced-feature failure. If MemCache offload is disabled for fallback, the shell may still contain `MMC_LOCAL_CONFIG_PATH` from the original prelude.

The MemCache helper should therefore provide cleanup text for fallback commands:

```bash
# --- wings-memcache: fallback cleanup ---
unset MMC_LOCAL_CONFIG_PATH
# --- end wings-memcache: fallback cleanup ---
```

When fallback removes `kv_transfer_config`, it should also prepend this cleanup. This prevents a feature-disabled fallback command from inheriting MemCache-specific environment.

## Capacity Rule

Capacity resolution must follow the page-owned offload input. This is the same value the user configures on the page for memory offload; MemCache should not have a separate hidden capacity.

1. authoritative page/env source: `KV_MEM_OFFLOAD_SIZE`
2. compatibility fallback: `LMCACHE_MAX_LOCAL_CPU_SIZE`, only if page plumbing still uses that field
3. auto mode: `KV_MEM_OFFLOAD_SIZE=auto` plus `AVAILABLE_POD_MEM_SIZE`

Validation:

- empty value: disable MemCache offload
- invalid value: disable MemCache offload
- zero or negative resolved value: disable MemCache offload
- positive value: render `ock.mmc.local_service.dram.size = <value>GB`

Do not write `20GB` when the page did not provide a valid value.

`ock.mmc.local_service.dram.size` should use the resolved MemCache local-service budget. Do not accidentally reuse a per-card LMCache value unless implementation confirms MemCache expects that unit.

## Integration Responsibilities

### `config_loader.py`

Responsibilities:

- decide whether Kimi MemCache offload is effective
- validate memory capacity
- inject `AscendStoreConnector` into `engine_config.kv_transfer_config`
- suppress the feature when capacity is missing

### Dedicated MemCache helper

Responsibilities:

- render `engine_prelude`
- render `fallback_cleanup`
- render `master_script`
- read shell templates that own all `ock.mmc.*` text
- expose resolved env values for tests/logging

Current module and templates:

```text
wings_control/features/memcache/hybrid.py
wings_control/features/memcache/memcache_engine_prelude.sh
wings_control/features/memcache/memcache_master.sh
```

### `wings_entry.py`

Responsibilities:

- call the helper after merged config is known
- concatenate `engine_prelude` before `script_body`
- prepend `fallback_cleanup` to feature-disabled fallback commands
- write or expose `master_script` as a separate artifact if this deployment manages MetaService
- skip LMCache patch installation for the MemCache variant
- write `advanced_features.json` using the effective MemCache state

### `vllm_adapter.py`

Responsibilities:

- keep generating vLLM command and CLI arguments
- skip the existing LMCache env/YAML renderer when the effective variant is MemCache
- do not render `mmc_local.conf`
- do not render `mmc_meta.conf`
- do not export `MMC_LOCAL_CONFIG_PATH`
- do not start `MetaService`

## Observability

When MemCache offload is effective:

- final `start_command.sh` contains `# --- wings-memcache: engine prelude ---`
- final `start_command.sh` exports `MMC_LOCAL_CONFIG_PATH`
- final `start_command.sh` writes `mmc_local.conf`
- vLLM command contains `AscendStoreConnector`
- `advanced_features.json.features.kv_offload=true`
- `advanced_features.json.variants.kv_offload=memcache`

When MemCache offload is disabled:

- no MemCache engine prelude is concatenated
- no `MMC_LOCAL_CONFIG_PATH`
- no `mmc_local.conf`
- no `AscendStoreConnector`
- `advanced_features.json.features.kv_offload=false`

## Test Coverage

Required tests:

- valid memory returns a non-empty `engine_prelude`
- missing memory returns empty script fragments
- invalid memory returns empty script fragments
- `engine_prelude` exports `MMC_LOCAL_CONFIG_PATH`
- `engine_prelude` renders all `ock.mmc.local_service.*` parameters
- `master_script` exports `MMC_META_CONFIG_PATH`
- `fallback_cleanup` unsets `MMC_LOCAL_CONFIG_PATH`
- final assembled script places MemCache prelude before the engine command
- MemCache variant does not emit LMCache env/YAML config
- `vllm_adapter.py` does not own MemCache connector builders or `ock.mmc.*` text
- `wings_control/features/memcache/hybrid.py` renders fragments from `.sh` templates
