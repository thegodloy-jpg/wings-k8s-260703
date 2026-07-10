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
# --- end wings-memcache: engine prelude ---
