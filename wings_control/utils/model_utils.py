"""模型元数据解析和架构识别辅助函数。

在命令生成路径中被复用。

功能概述:
    本模块提供模型元信息提取，用于引擎自动选择和参数默认值决策:
    - ModelIdentifier 类: 读取 config.json 并解析模型架构/类型/量化方式
    - 模型架构映射表: 以架构名为 key，映射到已验证模型列表
    - 模型类型分类: llm/embedding/rerank

支持的模型架构:
    - LLM:       DeepseekV3ForCausalLM, DeepseekV32ForCausalLM,
                 GlmMoeDsaForCausalLM,
                 Glm4ForCausalLM, Glm4MoeForCausalLM,
                 Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM,
                 Qwen3NextForCausalLM, Qwen3_5ForConditionalGeneration,
                 Qwen3_5MoeForConditionalGeneration, MiniMaxM2ForCausalLM,
                 LlamaForCausalLM
    - Embedding: XLMRobertaModel, BertModel, Qwen3ForCausalLM(Embedding)
    - Rerank:    XLMRobertaForSequenceClassification

Sidecar 架构契约:
    - 模型识别必须保持确定性（同参数同结果）
    - 解析器行为向后兼容
"""
# Copyright (c) xFusion Digital Technologies Co., Ltd. 2025-2025. All rights reserved.
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Any, Optional

from utils.file_utils import load_json_config
from core.version_util import engine_version_platform
from utils.device_utils import resolve_card_token
from utils.env_utils import get_config_force_env

logger = logging.getLogger(__name__)

# IndexCache 支持的模型架构列表
# 当 KV 稀疏开启时，这些架构使用 IndexCache 策略；其他架构使用 FP8 KV CACHE
INDEXCACHE_ARCHS: frozenset[str] = frozenset({
    "GlmMoeDsaForCausalLM",
    "DeepseekV32ForCausalLM",
})

# ── Smart 三特性白名单（投机 spec / 稀疏 sparse / 卸载 offload）────────────────
# 数据外置于 config/smart_feature_whitelist.json，按特性拆为**三独立表**（最终白名单
# + 全枚举矩阵详解见 xuqiu/smart/需求一-白名单详解.md）。本模块加载为每特性一张表：
#   每条 = (engine 精确, name_tokens 小写子串, card_tokens 子串|"*")。
#   无 forced：有效=开关 on AND 命中对应表；miss → 该特性不产（§0 裁定1/2）。
#   表内顺序即优先级（首命中即命中）；更具体型号须排在更泛型号之前。
SMART_FEATURES = ("spec", "sparse", "offload")
_SMART_WHITELIST_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "smart_feature_whitelist.json"
)


def _normalize_match_rows(rows) -> tuple[dict, ...]:
    """Normalize model/card matching fields while retaining row metadata."""
    normalized = []
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        normalized.append({
            **row,
            "engine": str(row.get("engine", "")),
            "name_tokens": tuple(str(tok).lower() for tok in row.get("name_tokens", ())),
            "exclude_name_tokens": tuple(
                str(tok).lower() for tok in row.get("exclude_name_tokens", ())
            ),
            "card_tokens": tuple(str(tok).lower() for tok in row.get("card_tokens", ())),
        })
    return tuple(normalized)


def _load_smart_feature_whitelists(path: Path = _SMART_WHITELIST_PATH) -> dict:
    """从 JSON 加载三独立白名单表，返回 {feature: tuple[row, ...]}。

    每条记录保留整行元数据，并规整 engine/name_tokens/card_tokens 供匹配使用。
    文件缺失/解析失败或某表为空 → 该表空 tuple（miss 即不产，安全降级）。
    """
    data = load_json_config(str(path))
    tables = {}
    for feat in SMART_FEATURES:
        rows = data.get(feat, []) if isinstance(data, dict) else []
        tables[feat] = _normalize_match_rows(rows)
    if not any(tables.values()):
        logger.error(
            "[SmartFeature] all whitelists empty after load from %s "
            "-> spec/sparse/offload will all miss.", path,
        )
    return tables


_SMART_WHITELISTS: dict = _load_smart_feature_whitelists()


def _whitelist_table_match(table, engine: str, hay: str, ct: str) -> Optional[dict]:
    """三维（engine 精确 → 名子串 → 卡型 "*"/子串）与匹配，首命中返回行。"""
    for row in table:
        if row["engine"] != engine:
            continue
        if not any(tok in hay for tok in row["name_tokens"]):
            continue
        if any(tok in hay for tok in row.get("exclude_name_tokens", ())):
            continue
        card_tokens = row["card_tokens"]
        if not (("*" in card_tokens) or any(c in ct for c in card_tokens)):
            continue
        return row
    return None


def _whitelist_table_hit(table, engine: str, hay: str, ct: str) -> bool:
    return _whitelist_table_match(table, engine, hay, ct) is not None


def feature_allowed(engine, model_name, model_path, card_token, feature) -> bool:
    """白名单门控：该 (engine, model, card) 是否命中某特性独立表（spec/sparse/offload）。

    无 forced（需求一 §0 裁定1：只开关不强制，白名单只收窄、永不强开）。
    """
    table = _SMART_WHITELISTS.get(feature)
    if not table:
        return False
    hay = " ".join(str(x).lower() for x in (model_name, model_path) if x)
    ct = (card_token or "").lower()
    return _whitelist_table_hit(table, engine, hay, ct)


def resolve_feature_whitelist_row(engine, model_name, model_path, card_token, feature) -> Optional[dict]:
    """返回首条匹配 engine/model/card 的 smart feature 白名单行。"""
    table = _SMART_WHITELISTS.get(feature)
    if not table:
        return None
    hay = " ".join(str(x).lower() for x in (model_name, model_path) if x)
    ct = (card_token or "").lower()
    return _whitelist_table_match(table, engine, hay, ct)


def resolve_feature_whitelist_row_from_params(
    params: Optional[dict],
    engine: str,
    feature: str,
    *,
    require_enabled: bool = False,
) -> Optional[dict]:
    """Return the matched smart-feature row for launcher/adapter params.

    The smart whitelist module owns model/card row lookup.  Launcher, config
    loader, feature helpers and engine adapters should call this instead of
    rebuilding ``model_name``/``model_path``/``_smart_card_token`` lookup rules.
    ``CONFIG_FORCE=true`` bypasses smart whitelist ownership so explicit config
    can fully take over.
    """
    if not params or get_config_force_env():
        return None
    smart_feats = params.get("_smart_feats")
    if require_enabled and smart_feats is not None and feature not in smart_feats:
        return None
    return resolve_feature_whitelist_row(
        engine,
        params.get("model_name"),
        params.get("model_path"),
        params.get("_smart_card_token") or resolve_card_token(),
        feature,
    )


def resolve_offload_whitelist_backend(
    params: Optional[dict],
    engine: str,
) -> str:
    """Return the explicit offload backend declared by the matched whitelist row."""
    row = resolve_feature_whitelist_row_from_params(
        params,
        engine,
        "offload",
        require_enabled=True,
    )
    return str(row.get("backend") or "") if row else ""


def resolve_feature_whitelist(engine, model_name, model_path, card_token):
    """返回 (engine, model, card) 命中的允许特性 frozenset（聚合三独立表，保持原返回契约）。"""
    hay = " ".join(str(x).lower() for x in (model_name, model_path) if x)
    ct = (card_token or "").lower()
    return frozenset(
        feat for feat in SMART_FEATURES
        if _whitelist_table_hit(_SMART_WHITELISTS[feat], engine, hay, ct)
    )


def resolve_forced_feature_whitelist(engine, model_name, model_path, card_token):
    """Return whitelisted smart features whose matching row explicitly forces enablement."""
    hay = " ".join(str(x).lower() for x in (model_name, model_path) if x)
    ct = (card_token or "").lower()
    forced = []
    for feat in SMART_FEATURES:
        row = _whitelist_table_match(_SMART_WHITELISTS[feat], engine, hay, ct)
        if row and row.get("forced") is True:
            forced.append(feat)
    return frozenset(forced)


def resolve_sparse_topk(engine, model_name, model_path, card_token, sparse_level, default: int = 4) -> int:
    """返回 sparse 表当前匹配行在指定档位下的 index_topk_freq。

    未命中 sparse 表、未声明 topk、或该档位缺失时回退 accuracy_first，再回退 default。
    """
    hay = " ".join(str(x).lower() for x in (model_name, model_path) if x)
    ct = (card_token or "").lower()
    row = _whitelist_table_match(_SMART_WHITELISTS.get("sparse", ()), engine, hay, ct)
    topk = row.get("topk", {}) if row else {}
    if not isinstance(topk, dict):
        return int(default)
    value = topk.get(sparse_level, topk.get("accuracy_first", default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)

_GLM51_NAME_MARKERS = (
    "glm-5.1",
    "glm5.1",
    "glm_5.1",
    "glm 5.1",
    "glm-51",
    "glm51",
)

_MODEL_NAME_CONFIG_KEYS = (
    "_name_or_path",
    "name_or_path",
    "model_name",
    "model_id",
    "hub_model_id",
)


def _contains_glm51_marker(value: Any) -> bool:
    """Return True when a free-form metadata value clearly names GLM-5.1."""
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _GLM51_NAME_MARKERS)


def is_glm51_model(model_name: Any = None, model_path: Any = None,
                   config: Optional[dict] = None) -> bool:
    """Best-effort GLM-5.1 variant detection from stable metadata.

    ``GLM-5`` and ``GLM-5.1`` both use ``GlmMoeDsaForCausalLM`` in
    ``config.json``.  Architecture alone cannot distinguish them, so this
    helper checks explicit model-name sources first: user ``model_name``, model
    path, and common HuggingFace name fields in ``config.json``.
    """
    if _contains_glm51_marker(model_name) or _contains_glm51_marker(model_path):
        return True
    if isinstance(config, dict):
        for key in _MODEL_NAME_CONFIG_KEYS:
            if _contains_glm51_marker(config.get(key)):
                return True
    return False


def is_glm_moe_dsa_glm51(model_info: Any, model_name: Any = None,
                         model_path: Any = None) -> bool:
    """Return True for the GLM-5.1 variant of ``GlmMoeDsaForCausalLM``."""
    if getattr(model_info, "model_architecture", None) != "GlmMoeDsaForCausalLM":
        return False
    return is_glm51_model(
        model_name if model_name is not None else getattr(model_info, "model_name", None),
        model_path if model_path is not None else getattr(model_info, "model_path", None),
        getattr(model_info, "config", None),
    )


# ── [GLM5.1-Ascend-Tmp] TEMPORARY: GLM-5.1 + vllm_ascend KV-Sparse Whitelist ──
# Scope: engine == "vllm_ascend" AND is_glm_moe_dsa_glm51(...)
# Behavior gated by this predicate:
#   - vllm_adapter._build_kv_sparse_cmd: 走 IndexCache --hf-overrides 分支
#     （不写 engine_config、不触发 indexcache 补丁安装）
# 注：原 _force_kv_sparse_for_glm51_ascend「强制开（即便 enable_sparse=False 也产）」
#    已按需求一 §0 裁定1 删除；GLM-5.1·Ascend(sparse) 现降为普通开关门控
#    （开关 on 且命中白名单才产，由 config_loader.apply_effective_feature_enablement 收口）。
# 范围说明：仅 GLM-5.1（架构 GlmMoeDsaForCausalLM + 名称/路径标记 5.1），
# GLM-5 (非 5.1)、DeepseekV32 在 ascend 上不进入 IndexCache。
# 移除时机：vllm-ascend 支持 indexcache 补丁安装时一次性拆除。
# 移除方法：grep "[GLM5.1-Ascend-Tmp]" 一次性定位所有触点。
def is_glm51_ascend_kvsparse_tmp_scope(model_info: Any, engine: Any,
                                       model_name: Any = None,
                                       model_path: Any = None) -> bool:
    """[GLM5.1-Ascend-Tmp] Return True for vllm_ascend + GLM-5.1（单机/双机均适用）."""
    if engine != "vllm_ascend":
        return False
    name = model_name if model_name is not None else getattr(model_info, "model_name", None)
    path = model_path if model_path is not None else getattr(model_info, "model_path", None)
    if not is_glm51_model(name, path, getattr(model_info, "config", None)):
        return False
    arch = getattr(model_info, "model_architecture", None)
    return arch in (None, "", "unknown_architecture", "GlmMoeDsaForCausalLM")


# ── GLM-5.2 识别（架构同 GlmMoeDsaForCausalLM，与 GLM-5/5.1 靠名称/路径区分）──
# GLM-5 / GLM-5.1 / GLM-5.2 的 config.json architectures[0] 同为 GlmMoeDsaForCausalLM，
# 架构维度无分辨力。GLM-5.2 在 wings 上需与 GLM-5/5.1 反向处理（MTP num=3、A3 双机保留
# additional_config），故用与 is_glm51_model 同范式、互斥的名称标识切出：
#   5.2 标识 {glm-5.2, glm5.2, glm_5.2, glm 5.2, glm-52, glm52} 与
#   5.1 标识 {glm-5.1, ..., glm-51, glm51} 不相交，亦不命中 GLM-5.0 基座（不含 5.2/52）。
_GLM52_NAME_MARKERS = (
    "glm-5.2",
    "glm5.2",
    "glm_5.2",
    "glm 5.2",
    "glm-52",
    "glm52",
)


def _contains_glm52_marker(value: Any) -> bool:
    """Return True when a free-form metadata value clearly names GLM-5.2."""
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _GLM52_NAME_MARKERS)


def is_glm52_model(model_name: Any = None, model_path: Any = None,
                   config: Optional[dict] = None) -> bool:
    """Best-effort GLM-5.2 variant detection from stable metadata.

    与 :func:`is_glm51_model` 同范式：先看 ``model_name`` / ``model_path``，再看
    ``config.json`` 常见名字段（``_name_or_path`` 等）。标识集与 GLM-5.1 严格互斥，
    且不会误命中 GLM-5.0 基座。
    """
    if _contains_glm52_marker(model_name) or _contains_glm52_marker(model_path):
        return True
    if isinstance(config, dict):
        for key in _MODEL_NAME_CONFIG_KEYS:
            if _contains_glm52_marker(config.get(key)):
                return True
    return False


def is_glm52_single_node_even(source: Optional[dict]) -> bool:
    """GLM-5.2 + vllm_ascend + 单机(nnodes==1) + 偶数卡 + **a3(910C)** 的统一判定（共享真相源）。

    config_loader._set_parallelism_params(用 ctx)与
    vllm_adapter._apply_generic_deepseek_ascend_dp_defaults(用 params)共用此判定，
    确保「config_loader 让位 ⇔ adapter 接管 TP=device_count//DP + DP2」的条件逐字一致，
    不会因两处各写一份布尔链而漂移。``source`` 须含 engine / nnodes / device_count /
    model_name / model_path 键（ctx 与 params 同构）。

    平台门控：单机 TP 减半 + DP2 是 **910C(a3) 官方 recipe**，仅 a3 触发；910B(a2) 上触发
    GLM-5.2 不进此分支，回落通用 GlmMoeDsa 单机整卡 TP（DP1）。平台**只按 engine-version
    后缀**判定（``-a3``），复用 :func:`core.version_util.engine_version_platform`——它读全局
    ``ENGINE_VERSION``，ctx / params 两侧取值一致，不依赖各自字典里是否带平台键。
    """
    src = source or {}
    if src.get("engine") != "vllm_ascend":
        return False
    try:
        nnodes = int(src.get("nnodes") or 1)
        device_count = int(src.get("device_count") or 0)
    except (TypeError, ValueError):
        return False
    if nnodes != 1 or device_count <= 0 or device_count % 2 != 0:
        return False
    if engine_version_platform() != "a3":
        return False
    return is_glm52_model(src.get("model_name"), src.get("model_path"))


def is_qwen3_5_397b_nvfp4_vllm(source: Optional[dict], engine: Optional[str] = None) -> bool:
    """Qwen3.5-397B-A17B-NVFP4 + vllm 的统一判定。

    **不耦合硬件**：
      - NVFP4 FP4 运行时 env（DEEP_GEMM / FLASHINFER_MOE_FP4）是模型量化自带的配方，
        只要 engine=="vllm" 且模型为 Qwen3.5-397B-A17B-NVFP4 即注入。NVFP4 量化本身
        只在 Blackwell 上可加载，故不再叠加硬件判定——避免硬件探测失败时漏注入。

    命中条件（二者同时满足）：
      1. engine == "vllm"
      2. model_name/model_path 同时含 qwen3.5（或 qwen3_5）与 nvfp4
         （Qwen3.5 系列仅 397B-A17B 有 NVFP4 变体，故 qwen3.5+nvfp4 已足够精确）

    ``source`` 须含 engine / model_name / model_path 键（ctx 与 params 同构）；
    ``engine`` 参数可选，source 缺 engine 时用它兜底。
    """
    src = source or {}
    eff_engine = engine or src.get("engine")
    if eff_engine != "vllm":
        return False
    for key in ("model_name", "model_path"):
        val = str(src.get(key) or "").lower()
        if ("qwen3.5" in val or "qwen3_5" in val) and "nvfp4" in val:
            return True
    return False


def is_deepseek_v4_flash_rtx_pro_5000(source: Optional[dict], engine: Optional[str] = None) -> bool:
    """DeepSeek-V4-Flash + rtx_pro_5000_72G + vllm 的统一判定。

    config_loader._set_parallelism_params(用 ctx，提前短路通用 TP 兜底)与
    vllm_adapter._apply_deepseek_v4_flash_nv_engine_defaults(用 params，接管
    TP=min(4,device_count) + DP=device_count/TP 动态策略)共用此判定，确保
    「config_loader 让位 ⇔ adapter 接管 TP/DP」的条件逐字一致。

    三者同时满足才命中（与 nvidia_default.json 的 rtx_pro_5000_72G 配置块选择同口径）：
      1. engine == "vllm"
      2. model_name/model_path 含 deepseek-v4-flash 标识（覆盖 -w8a8-mtp 等后缀）
      3. 硬件为 rtx_pro_5000_72G（source.hardware_family/device_details/details，
         或 loader 已解析出的 _smart_card_token=rtxpro5000-72）

    ``source`` 须含 engine / model_name / model_path 键（ctx 与 params 同构）；
    ``engine`` 参数可选，source 缺 engine 时用它兜底。
    """
    src = source or {}
    eff_engine = engine or src.get("engine")
    if eff_engine != "vllm":
        return False
    model_hit = False
    for key in ("model_name", "model_path"):
        val = str(src.get(key) or "").lower()
        if ("deepseek-v4-flash" in val
                or "deepseek_v4_flash" in val
                or "deepseekv4flash" in val):
            model_hit = True
            break
    if not model_hit:
        return False
    card_token = str(src.get("_smart_card_token") or src.get("card_token") or "").lower()
    if "rtxpro5000-72" in card_token:
        return True
    hardware_texts = [str(src.get("hardware_family") or "")]
    for detail in src.get("device_details") or src.get("details") or []:
        if isinstance(detail, dict):
            hardware_texts.append(str(detail.get("name") or ""))

    for text in hardware_texts:
        normalized = text.lower().replace("-", "_").replace(" ", "_")
        if "rtx_pro_5000" in normalized and "72" in normalized:
            return True
    return False


def is_minimax_m27_rtx_pro_5000_vllm(source: Optional[dict], engine: Optional[str] = None) -> bool:
    """MiniMax-M2.7 + rtx_pro_5000_72G + vllm 的统一判定。

    vllm_adapter.build_start_script 据此切换到「MiniMax-M2.7 RTX-PRO-5000 专属启动配方」
    （固定 LMCache 环境变量 + 固定 vLLM CLI，仅 model_path/port 用实际值）。

    三者同时满足才命中（与 smart_feature_whitelist.json 的 minimax-m2.7/rtxpro5000-72 行同口径）：
      1. engine == "vllm"
      2. model_name/model_path 含 minimax-m2.7 标识（覆盖 -NVFP4 等后缀）
      3. 硬件为 rtx_pro_5000_72G（source.hardware_family/device_details/details，
         或 loader 已解析出的 _smart_card_token=rtxpro5000-72）

    ``source`` 须含 engine / model_name / model_path 键（ctx 与 params 同构）；
    ``engine`` 参数可选，source 缺 engine 时用它兜底。
    """
    src = source or {}
    eff_engine = engine or src.get("engine")
    if eff_engine != "vllm":
        return False
    model_hit = False
    for key in ("model_name", "model_path"):
        val = str(src.get(key) or "").lower()
        if ("minimax-m2.7" in val
                or "minimax_m2.7" in val
                or "minimaxm2.7" in val):
            model_hit = True
            break
    if not model_hit:
        return False
    card_token = str(src.get("_smart_card_token") or src.get("card_token") or "").lower()
    if "rtxpro5000-72" in card_token:
        return True
    hardware_texts = [str(src.get("hardware_family") or "")]
    for detail in src.get("device_details") or src.get("details") or []:
        if isinstance(detail, dict):
            hardware_texts.append(str(detail.get("name") or ""))

    for text in hardware_texts:
        normalized = text.lower().replace("-", "_").replace(" ", "_")
        if "rtx_pro_5000" in normalized and "72" in normalized:
            return True
    return False


#
_LLM_MODELS = {
    "DeepseekV3ForCausalLM": [
        "DeepSeek-R1",
        "DeepSeek-R1-0528",
        "DeepSeek-V3",
        "DeepSeek-V3-0324",
        "DeepSeek-V3.1",
        "DeepSeek-Coder-V2-Instruct",
        "DeepSeek-R1-w8a8",
        "DeepSeek-R1-0528-w8a8",
        "DeepSeek-V3-w8a8",
        "DeepSeek-V3-0324-w8a8",
        "DeepSeek-V3.1-w8a8",
        "DeepSeek-Coder-V2-Instruct-w8a8"
        ],
    # DeepSeek-V4 系列权重的 ``config.json -> architectures[0]`` 为
    # ``DeepseekV4ForCausalLM``。独立架构键保证 ascend_default.json 下的
    # V4-Flash / V4-Pro 默认值（``method: mtp``、``enable_cpu_binding: bool true``、
    # ``multistream_overlap_shared_expert: false`` 等）能够命中。
    "DeepseekV4ForCausalLM": [
        "DeepSeek-V4",
        "DeepSeek-V4-w8a8",
        "DeepSeek-V4-Flash",
        "DeepSeek-V4-Flash-w8a8-mtp",
        "DeepSeek-V4-Pro",
        "DeepSeek-V4-Pro-w4a8-mtp",
        ],
    "KimiK25ForConditionalGeneration": [
        "Kimi-K2.5",
        "Kimi-K2.5-w4a8",
        "Kimi-K2.7",
        "Kimi-K2.7-w4a8",
        "Kimi-K2.7-Code",
        "Kimi-K2.7-Code-w4a8",
        ],
    "DeepseekV32ForCausalLM": [
        "DeepSeek-V3.2",
        "DeepSeek-V3.2-w8a8",
        "DeepSeek-V3.2-Exp",
        ],
    "Glm4ForCausalLM": [
        "GLM-4-9B-0414"
        ],
    "GlmMoeDsaForCausalLM": [
        "GLM-5",
        "GLM-5-FP8",
        "GLM-5-w4a8",
        "GLM-5.1",
        "GLM5.1",
        "GLM-5.1-w8a8",
        "GLM-5.1-FP8",
        "GLM-5.2",
        "GLM-5.2-w8a8",
        ],
    "Glm4MoeForCausalLM": [
        "GLM-4.7",
        "GLM4.7",
        "GLM-4.7-w8a8"
        ],
    "Qwen2ForCausalLM": [
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Qwen-14B",
        "DeepSeek-R1-Distill-Qwen-32B",
        "Qwen2.5-32B-Instruct",
        "QwQ-32B"
        ],
    "Qwen3ForCausalLM": [
        "Qwen3-32B"
        ],
    "Qwen3MoeForCausalLM": [
        "Qwen3-30B-A3B",
        "Qwen3-235B-A22B"
        ],
    "Qwen3NextForCausalLM": [
        "Qwen3-Next-80B-A3B-Instruct"
        ],
    "Qwen3_5ForConditionalGeneration": [
        "Qwen3.5-27B",
        "Qwen3.5-27B-w8a8",
        "Qwen3.6-27B",
        "Qwen3.6-27B-w8a8"
        ],
    "Qwen3_5MoeForConditionalGeneration": [
        "Qwen3.5-397B-A17B-NVFP4",
        "Qwen3.5-397B-A17B-w8a8",
        "Qwen3.5-35B-A3B",
        "Qwen3.5-122B-A10B",
        "Qwen3.5-397B-A17B",
        "Qwen3.6-35B-A3B",
        "Qwen3.6-35B-A3B-w8a8",
        "Qwen-AgentWorld-35B-A3B"
        ],
    "MiniMaxM2ForCausalLM": [
        "MiniMax-M2.5",
        "MiniMax-M2.5-w8a8",
        "MiniMax-M2.7",
        "MiniMax-M2.7-w8a8"
        ],
    "MiniMaxM3SparseForConditionalGeneration": [
        "MiniMax-M3",
        "MiniMax-M3-MXFP8",
        ],
    "LlamaForCausalLM": [
        "LLaMA3-8B",
        "LLaMA3.1-70B",
        "LLaMA3.1-70B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Llama-70B"
        ]
}

_EMBEDDING_MODELS = {
    "XLMRobertaModel": [
        "bge-m3"
        ],
    "BertModel": [
        "bge-large-zh-v1.5"
        ],
    "Qwen3ForCausalLM": [
        'Qwen3-Embedding-0.6B'
        ]
}

_RERANK_MODELS = {
    "XLMRobertaForSequenceClassification": [
        "bge-reranker-v2-m3",
        "bge-reranker-large"
        ]
}


# ---------------------------------------------------------------------------
# Thinking-mode 策略（供 --enable-auto-think-choice 在启动命令注入服务级默认思考状态）
# ---------------------------------------------------------------------------
# 背景：reasoning_parser 只控制服务端是否「解析」思维链，控制不了模型「是否思考」。
# 生成端（vllm/vllm_ascend）按开关在启动命令注入 --default-chat-template-kwargs 设服务级默认：
#   - 开关开 → {key: true}  强制默认【打开】思考；开关关 → {key: false} 强制默认【关闭】思考。
#   （见 config_loader._set_thinking_default；本函数只负责解析【键名】，布尔值由开关决定。）
# 注：这是服务级默认，请求级 chat_template_kwargs 优先级更高、可覆盖（客户端自负，不兜底）。
# 不同模型族的键名不同（已对齐各家官方/vLLM）：
#   - Qwen3 系列 / GLM-4.5+ MoE 系列 → enable_thinking
#   - DeepSeek-V3.1 / V3.2（混合推理）→ thinking
# 始终推理模型（R1 / QwQ / MiniMax-M2 等）无法切换思考，返回 THINKING_ALWAYS_ON 仅用于告警。

# resolve_thinking_off_policy 的 mode 取值（恒为 str，配合恒为 dict 的 off_kwargs，
# 使所有分支返回的类型与个数一致，符合 G.CTL.01）。
THINKING_HYBRID = "hybrid"          # 混合推理：off_kwargs 为关闭思考所需 chat_template_kwargs
THINKING_ALWAYS_ON = "always_on"    # 始终推理：无法关闭思考，off_kwargs 为空，仅告警
THINKING_NONE = "none"              # 非思考模型 / 无法识别：不介入，off_kwargs 为空
# 始终推理（无法关闭思考）模型名片段，优先于混合推理判断。
_ALWAYS_ON_THINKING_TOKENS = ("r1", "qwq", "minimax-m2")


def resolve_thinking_off_policy(model_name: str):
    """根据模型名解析思考策略，返回 (mode, off_kwargs)。

    Returns:
        tuple[str, dict]: 二元组，所有分支类型与个数一致——
            mode       : THINKING_HYBRID / THINKING_ALWAYS_ON / THINKING_NONE 之一；
            off_kwargs : 仅 THINKING_HYBRID 时非空，为强制关闭思考需注入/覆盖的
                         chat_template_kwargs（如 {"enable_thinking": False}）；
                         THINKING_ALWAYS_ON / THINKING_NONE 时恒为空 dict。

    注：取值按模型名子串匹配（忽略大小写、下划线归一为连字符），与各家官方
    chat 模板键名对齐；R1/蒸馏 R1/QwQ/MiniMax-M2 等优先判定为 THINKING_ALWAYS_ON。
    """
    if not model_name:
        return THINKING_NONE, {}
    name = model_name.lower().replace("_", "-")

    # 1) 始终推理模型优先（R1 / R1-Distill / QwQ / MiniMax-M2）
    if any(tok in name for tok in _ALWAYS_ON_THINKING_TOKENS):
        return THINKING_ALWAYS_ON, {}

    # 2) Qwen3 系列（混合推理）：enable_thinking
    #    排除 Qwen3-Coder-*：官方为非思考模型，reason_parser.yaml 显式置 null，
    #    不应注入思考默认（否则开关开时会把非思考的 Coder 误强制思考）。
    if "qwen3" in name and "coder" not in name:
        return THINKING_HYBRID, {"enable_thinking": False}

    # 3) GLM MoE 混合推理（GLM-4.5/4.6/4.7/5/5.1）：enable_thinking
    #    排除非思考的 glm-4-9b 等（不含 4.5+ 版本号则不匹配）。
    if any(tag in name for tag in ("glm-4.5", "glm-4.6", "glm-4.7", "glm-5")):
        return THINKING_HYBRID, {"enable_thinking": False}

    # 4) DeepSeek V3.1/V3.2/V4 + Kimi-K2.x（混合推理）：thinking 键（注意不是 enable_thinking）
    #    - DeepSeek V4-Flash/-Pro：官方 vLLM Recipes 确认混合推理、键名 thinking；另有
    #      reasoning_effort=high/max 控制思考【深度】，属请求级、与本开/关正交，不在此注入。
    #    - Kimi-K2.x：moonshotai/Kimi-K2.5 chat_template.jinja 字面以
    #      `{% if thinking is defined and thinking is false %}` 关闭思考（键名 thinking，
    #      非 enable_thinking），混合推理（思考/即时双模）；与 reasoning_parser=kimi_k2 对应。
    if "deepseek-v3" in name or "deepseek-v4" in name or "kimi-k2" in name:
        return THINKING_HYBRID, {"thinking": False}

    # 其余（Qwen2.5 / Llama / GLM-4-9B / embedding / rerank 等非思考模型）→ 无需处理
    return THINKING_NONE, {}


class ModelIdentifier:
    """模型元信息识别器，从模型目录的 config.json 提取架构、类型、量化信息。

    Attributes:
        model_name:         模型名称（用户传入）
        model_path:         模型权重目录路径
        model_type:         模型类型（'auto' 时自动推断）
        config:             从 config.json 加载的配置字典
        model_architecture: 模型架构名（如 'DeepseekV3ForCausalLM'）
        model_quantize:     量化方式（如 'fp8'、'bfloat16'）
        num_hidden_layers:  隐藏层数量（用于 CUDA Graph 计算）
    """
    def __init__(self, model_name: str, model_path: str, model_type: str):
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.config = load_json_config(self.model_path / "config.json")
        self.model_architecture = self.identify_model_architecture()
        self.model_quantize = self.identify_model_quantize()
        self.num_hidden_layers = self.config.get("num_hidden_layers")
        self.model_dict = {
                "llm": _LLM_MODELS,
                "embedding": _EMBEDDING_MODELS,
                "rerank": _RERANK_MODELS
            }

    def identify_model_architecture(self) -> Optional[str]:
        """从 config.json 中提取模型架构名称。

        读取 architectures 字段的第一个元素，如 ["DeepseekV3ForCausalLM"].

        Returns:
            str: 模型架构名称，未找到时返回 'unknown_architecture'
        """
        # Read the 'architectures' list from model config
        architectures = self.config.get("architectures", [])
        if architectures:
            return architectures[0]
        else:
            return "unknown_architecture"

    def identify_model_type(self) -> Optional[str]:
        """推断模型类型（llm/embedding/rerank）。

        当 model_type 为 'auto'、空字符串或 None 时，根据 model_name 与内置映射表匹配;
        否则直接返回用户指定值。

        Returns:
            str | None: 模型类型，无法推断时返回 None
        """
        if self.model_type in ('auto', '', None):
            model_name = self.model_name.lower()
            for model_type, models in self.model_dict.items():
                support_model_name = []
                for lst in models.values():
                    support_model_name += [name.lower() for name in lst]
                if model_name in support_model_name:
                    return model_type
            # llm
            return "llm"
        return self.model_type


    def identify_model_quantize(self) -> Optional[str]:
        model_quantize = ""
        if "quantize" in self.config:
            model_quantize = self.config["quantize"]
        elif "quantization_config" in self.config:
            model_quantize = self.config["quantization_config"].get("quant_method", "")
        if model_quantize:
            return model_quantize
        else:
            return self.config.get("torch_dtype", "")


    def is_wings_supported(self):
        support_model_architecture = []
        for models in self.model_dict.values():
            support_model_architecture += list(models.keys())
        if self.model_architecture in support_model_architecture:
            return True
        else:
            return False


class ModelIdentifierDraft:
    """草稿模型识别机制"""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.config = load_json_config(self.model_path / "config.json")
        self.draft_model_architecture = self.identify_model_architecture()
        self.model_draft_vocab_size = self.identify_draft_vocab_size()

    def identify_model_architecture(self) -> Optional[str]:
        """识别模型类型"""
        architectures = self.config.get("architectures", [])
        if architectures:
            return architectures[0]
        else:
            return "unknown_architecture"

    def identify_draft_vocab_size(self) -> Optional[bool]:
        """识别eagle3模型特有特征"""
        draft_vocab_size = self.config.get("draft_vocab_size", 0)
        if draft_vocab_size:
            return True
        else:
            return False
