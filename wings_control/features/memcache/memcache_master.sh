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
