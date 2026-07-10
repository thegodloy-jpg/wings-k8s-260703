#!/bin/bash
# --- wings-memcache: engine prelude ---
export WINGS_MEMCACHE_DIR="${WINGS_MEMCACHE_DIR:-/shared-volume/memcache}"
export WINGS_MEMCACHE_META_SERVICE_URL="${WINGS_MEMCACHE_META_SERVICE_URL:-{meta_service_url}}"
export WINGS_MEMCACHE_CONFIG_STORE_URL="${WINGS_MEMCACHE_CONFIG_STORE_URL:-{config_store_url}}"
export WINGS_MEMCACHE_LOG_LEVEL="${WINGS_MEMCACHE_LOG_LEVEL:-error}"
export WINGS_MEMCACHE_WORLD_SIZE="${WINGS_MEMCACHE_WORLD_SIZE:-256}"
export WINGS_MEMCACHE_PROTOCOL="${WINGS_MEMCACHE_PROTOCOL:-device_rdma}"
export WINGS_MEMCACHE_DRAM_GB="{dram_gb}"

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
{master_script}
WINGS_MEMCACHE_MASTER
chmod +x "${WINGS_MEMCACHE_DIR}/start_memcache_master.sh"

_wings_memcache_config_addr="${WINGS_MEMCACHE_CONFIG_STORE_URL#tcp://}"
_wings_memcache_config_host="${_wings_memcache_config_addr%:*}"
_wings_memcache_config_port="${_wings_memcache_config_addr##*:}"
_wings_memcache_master_log="${WINGS_MEMCACHE_DIR}/memcache-master.log"
_wings_memcache_master_pid_file="${WINGS_MEMCACHE_DIR}/memcache-master.pid"
_wings_memcache_master_pid=""

_wings_memcache_config_store_ready() {
    (: > "/dev/tcp/${_wings_memcache_config_host}/${_wings_memcache_config_port}") \
        >/dev/null 2>&1
}

if ! _wings_memcache_config_store_ready; then
    if [ -s "${_wings_memcache_master_pid_file}" ]; then
        _wings_memcache_master_pid="$(cat "${_wings_memcache_master_pid_file}" 2>/dev/null || true)"
        if ! kill -0 "${_wings_memcache_master_pid}" 2>/dev/null; then
            _wings_memcache_master_pid=""
            rm -f "${_wings_memcache_master_pid_file}"
        fi
    fi

    case "${_wings_memcache_config_host}" in
        127.0.0.1|localhost)
            if [ -z "${_wings_memcache_master_pid}" ]; then
                echo "[wings-memcache] Starting local MetaService for ConfigStore ${WINGS_MEMCACHE_CONFIG_STORE_URL}..."
                "${WINGS_MEMCACHE_DIR}/start_memcache_master.sh" \
                    >>"${_wings_memcache_master_log}" 2>&1 &
                _wings_memcache_master_pid=$!
                echo "${_wings_memcache_master_pid}" > "${_wings_memcache_master_pid_file}"
            else
                echo "[wings-memcache] Reusing MetaService process PID ${_wings_memcache_master_pid}."
            fi
            ;;
        *)
            echo "[wings-memcache] Waiting for external ConfigStore ${WINGS_MEMCACHE_CONFIG_STORE_URL}..."
            ;;
    esac

    _wings_memcache_ready=false
    for ((_wings_memcache_attempt=1; _wings_memcache_attempt<=60; _wings_memcache_attempt++)); do
        if _wings_memcache_config_store_ready; then
            _wings_memcache_ready=true
            break
        fi
        if [ -n "${_wings_memcache_master_pid}" ] \
                && ! kill -0 "${_wings_memcache_master_pid}" 2>/dev/null; then
            echo "[wings-memcache] ERROR: MetaService exited before ConfigStore became ready."
            break
        fi
        sleep 1
    done

    if [ "${_wings_memcache_ready}" != true ]; then
        echo "[wings-memcache] ERROR: ConfigStore ${WINGS_MEMCACHE_CONFIG_STORE_URL} is not ready."
        tail -n 50 "${_wings_memcache_master_log}" 2>/dev/null || true
        exit 1
    fi
fi
echo "[wings-memcache] ConfigStore ${WINGS_MEMCACHE_CONFIG_STORE_URL} is ready."
unset -f _wings_memcache_config_store_ready
# --- end wings-memcache: engine prelude ---
