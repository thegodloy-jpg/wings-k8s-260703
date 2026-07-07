#!/usr/bin/env python3
"""Wings-infer Dry-Run 脚本：通过官方入口生成 start_command.sh。

真实性原则（与生产链路对齐）：
  生产时用户只敲 wings_control/wings_start.sh 支持的那组「简单 CLI」，
  拓扑/平台/芯片由编排层(K8s/Master)注入 env，架构/量化由模型 config.json 发现。
  因此每个场景拆成三段硬边界，杜绝把「非用户输入」误当成用户入参：

    user_cli          —— 用户真敲的 CLI，key 必须 ⊆ wings_start.sh 支持集（否则报错）
    orchestration_env —— 编排层/K8s 注入的 env（NNODES/平台/engine-version/拓扑…）
    model_config      —— 模型自带（architecture + quantization_config，写进 mock config.json）

  入参流水线复刻真实链路：
    reset_managed_env()            每个场景 = 全新 pod，清掉上一轮 env
    create_mock_model_dir()        模型权重目录（含 config.json）
    apply_orchestration_env()      编排层在 launcher 进程启动前注入 env
    simulate_wings_start()         复刻 wings_start.sh：校验 + 端口/代理特例 + 双路下发(env+APP_ARGS)
    parse_launch_args(APP_ARGS)    与真实 `exec python -m wings_control "${APP_ARGS[@]}"` 等价
    + Master 分发动态注入 --node-rank（argparse 不读 NODE_RANK env）

使用方法：
  python dry_run.py --scenario glm52-910b-dual   # 跑单个场景
  python dry_run.py --list                        # 列出所有场景
  python dry_run.py                               # 跑全部场景

输出目录: build/output/
"""
import argparse
import json
import logging
import os
import sys
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("dry_run")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINGS_CONTROL = os.path.join(SCRIPT_DIR, "wings_control")
sys.path.insert(0, WINGS_CONTROL)

# ════════════════════════════════════════════════════════════════════════════
#  wings_start.sh CLI 契约（真相源：wings_control/wings_start.sh 的 case 解析器）
#  user_cli 的 key 必须落在这个集合里，否则一定是把「编排 env / 模型 config」
#  误当成了用户输入。
# ════════════════════════════════════════════════════════════════════════════
WINGS_START_VALUE_FLAGS = {
    "host", "port", "model-name", "dtype", "kv-cache-dtype", "quantization",
    "quantization-param-path", "gpu-memory-utilization", "block-size", "max-num-seqs",
    "seed", "max-num-batched-tokens", "model-path", "save-path", "engine",
    "input-length", "output-length", "config-file", "gpu-usage-mode", "device-count",
    "model-type", "speculative-decode-model-path",
}
WINGS_START_BOOL_FLAGS = {
    "trust-remote-code", "enable-chunked-prefill", "enable-expert-parallel",
    "enable-prefix-caching", "distributed", "enable-speculative-decode",
    "enable-sparse", "enable-rag-acc", "enable-auto-tool-choice", "enable-auto-think-choice",
}
WINGS_START_CLI_FLAGS = WINGS_START_VALUE_FLAGS | WINGS_START_BOOL_FLAGS


def _cli_flag_to_env(flag: str) -> str:
    """wings_start.sh 的 CLI→ENV 映射规则：'-' → '_'，转大写（model-name → MODEL_NAME）。"""
    return flag.replace("-", "_").upper()


# ── 编排层 / 基础设施 env 的 key（非 wings_start.sh CLI）；每个场景开始时一并清掉 ──
_PORT_ENV_KEYS = {"PORT", "PROXY_PORT", "ENABLE_REASON_PROXY"}
_INFRA_ENV_KEYS = {
    "NNODES", "NODE_RANK", "HEAD_NODE_ADDR", "MASTER_IP", "NODE_IPS", "NODES",
    "DISTRIBUTED_EXECUTOR_BACKEND", "RAY_HEAD_IP", "POD_IP", "RANK_IP",
    "NETWORK_INTERFACE", "ENABLE_ACCEL", "ENABLE_KV_OFFLOAD", "KV_MEM_OFFLOAD_SIZE",
    "ENABLE_KV_MEM_OFFLOAD", "SHARED_VOLUME_PATH", "WINGS_DEVICE", "WINGS_ASCEND_PLATFORM", "ENGINE_VERSION",
}


# ════════════════════════════════════════════════════════════════════════════
#  预置场景（三段式硬边界）
# ════════════════════════════════════════════════════════════════════════════
SCENARIOS = {
    "req1-glm47-fp8-nv-all-features": {
        "description": "Requirement 1: GLM-4.7-FP8 on NVIDIA with spec+sparse+offload requested",
        "user_cli": {
            "model-name": "GLM-4.7-FP8",
            "engine": "vllm",
            "device-count": 8,
            "enable-speculative-decode": True,
            "enable-sparse": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "mp",
            "ENABLE_KV_OFFLOAD": "true",
            "ENABLE_KV_MEM_OFFLOAD": "true",
            "KV_MEM_OFFLOAD_SIZE": "25",
        },
        "model_config": {
            "architecture": "Glm4MoeForCausalLM",
            "quantization_config": {"quant_method": "fp8"},
        },
    },
    "glm51-910b-dual": {
        "description": "GLM-5.1 + 910B(A2) 双机 dp_deployment",
        "user_cli": {
            "model-name": "glm-5.1-32b-chat",
            "engine": "vllm_ascend",
            "device-count": 8,
            "distributed": True,
            "enable-speculative-decode": True,
            "enable-sparse": True,
        },
        "orchestration_env": {
            "NNODES": "2",
            "NODE_IPS": "192.168.1.100,192.168.1.101",
            "HEAD_NODE_ADDR": "192.168.1.100",
            "DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment",
            "WINGS_ASCEND_PLATFORM": "a2",
        },
        "model_config": {
            "architecture": "GlmMoeDsaForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "glm52-910c-single": {
        "description": "GLM-5.2-w8a8 + 910C(A3) 单机16卡 (用户报障复现：期望 TP8/DP2)",
        "user_cli": {
            "model-name": "GLM-5.2-w8a8",
            "engine": "vllm_ascend",
            "device-count": 16,
            "enable-speculative-decode": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "ray",
            "ENGINE_VERSION": "0.21.0-a3",   # 芯片(A3/910C)由 engine-version 后缀确定
        },
        "model_config": {
            "architecture": "GlmMoeDsaForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "v4flash-a3-16": {
        "description": "DeepSeek-V4-Flash + 910C(A3) 单机16卡",
        "user_cli": {
            "model-name": "DeepSeek-V4-Flash",
            "engine": "vllm_ascend",
            "device-count": 16,
            # A3 会自动开启投机，用户不传 --enable-speculative-decode
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment",
            "WINGS_ASCEND_PLATFORM": "a3",
        },
        "model_config": {
            "architecture": "DeepseekV4ForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "glm51-910b-single": {
        "description": "GLM-5.1 + 910B(A2) 单机8卡",
        "user_cli": {
            "model-name": "glm-5.1-32b-chat",
            "engine": "vllm_ascend",
            "device-count": 8,
            "enable-speculative-decode": True,
            "enable-sparse": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "ray",
            "WINGS_ASCEND_PLATFORM": "a2",
        },
        "model_config": {
            "architecture": "GlmMoeDsaForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "v4flash-a2-8": {
        "description": "DeepSeek-V4-Flash + 910B(A2) 单机8卡",
        "user_cli": {
            "model-name": "DeepSeek-V4-Flash",
            "engine": "vllm_ascend",
            "device-count": 8,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment",
            "WINGS_ASCEND_PLATFORM": "a2",
        },
        "model_config": {
            "architecture": "DeepseekV4ForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "v4flash-nv-h20-8": {
        "description": "DeepSeek-V4-Flash + NVIDIA H20 单机8卡 (投机推理 + IndexCache默认强制开 + native KV 卸载)",
        "user_cli": {
            "model-name": "DeepSeek-V4-Flash",
            "engine": "vllm",
            "device-count": 8,
            "enable-speculative-decode": True,
            # [V4-Flash-NV-Day0] 不传 --enable-sparse：验证 IndexCache 由强制闸默认开（方案 A）
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "mp",
            "ENABLE_KV_OFFLOAD": "true",
            "KV_MEM_OFFLOAD_SIZE": "25",   # 每卡 GB，V4-Flash 乘本节点卡数
        },
        "model_config": {
            "architecture": "DeepseekV4ForCausalLM",
        },
    },
    "glm51-a3-dual": {
        "description": "GLM-5.1 + 910C(A3) 双机32卡 dp_deployment",
        "user_cli": {
            "model-name": "glm-5.1-32b-chat",
            "engine": "vllm_ascend",
            "device-count": 16,
            "distributed": True,
            "enable-speculative-decode": True,
            "enable-sparse": True,
        },
        "orchestration_env": {
            "NNODES": "2",
            "NODE_IPS": "192.168.1.100,192.168.1.101",
            "HEAD_NODE_ADDR": "192.168.1.100",
            "DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment",
            "WINGS_ASCEND_PLATFORM": "a3",
        },
        "model_config": {
            "architecture": "GlmMoeDsaForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "glm51-a3-16": {
        "description": "GLM-5.1 + 910C(A3) 单机16卡",
        "user_cli": {
            "model-name": "glm-5.1-32b-chat",
            "engine": "vllm_ascend",
            "device-count": 16,
            "enable-speculative-decode": True,
            "enable-sparse": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "ray",
            "WINGS_ASCEND_PLATFORM": "a3",
        },
        "model_config": {
            "architecture": "GlmMoeDsaForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "glm52-a3-16": {
        "description": "GLM-5.2(复杂名) + 910C(A3 via engine-version) 单机16卡 (验证 is_glm52 子串检测/num=3/保留 additional_config/MLAPO)",
        "user_cli": {
            "model-name": "GLM-5.2-355B-A3B-W8A8-Chat",
            "engine": "vllm_ascend",
            "device-count": 16,
            "enable-speculative-decode": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment",
            "ENGINE_VERSION": "0.21.0-a3",   # 芯片(A3/910C)由 engine-version 后缀确定
        },
        "model_config": {
            "architecture": "GlmMoeDsaForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "glm52-a3-dual": {
        "description": "GLM-5.2(复杂名) + 910C(A3 via engine-version) 双机32卡 dp_deployment (验证 num=3/A3双机保留 additional_config/EP/prefix/拓扑)",
        "user_cli": {
            "model-name": "GLM-5.2-355B-A3B-W8A8-Chat",
            "engine": "vllm_ascend",
            "device-count": 16,
            "distributed": True,
            "enable-speculative-decode": True,
        },
        "orchestration_env": {
            "NNODES": "2",
            "NODE_IPS": "192.168.1.100,192.168.1.101",
            "HEAD_NODE_ADDR": "192.168.1.100",
            "DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment",
            "ENGINE_VERSION": "0.21.0-a3",   # 芯片(A3/910C)由 engine-version 后缀确定
        },
        "model_config": {
            "architecture": "GlmMoeDsaForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "glm52-910b-single": {
        "description": "GLM-5.2-w8a8 + 910B(A2 via engine-version) 单机16卡 (平台门控:a2 回落整卡 TP=16/无DP2/保留 TASK_QUEUE)",
        "user_cli": {
            "model-name": "GLM-5.2-w8a8",
            "engine": "vllm_ascend",
            "device-count": 16,
            "enable-speculative-decode": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "ray",
            "ENGINE_VERSION": "0.21.0-a2",   # A2/910B；单机减半门控只认 -a3，不触发
        },
        "model_config": {
            "architecture": "GlmMoeDsaForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "glm52-910b-dual": {
        "description": "GLM-5.2-w8a8 + 910B(A2 via engine-version) 双机32卡 dp_deployment (双机不受单机门控影响:TP16/DP2/local1/保留 TASK_QUEUE)",
        "user_cli": {
            "model-name": "GLM-5.2-w8a8",
            "engine": "vllm_ascend",
            "device-count": 16,
            "distributed": True,
            "enable-speculative-decode": True,
        },
        "orchestration_env": {
            "NNODES": "2",
            "NODE_IPS": "192.168.1.100,192.168.1.101",
            "HEAD_NODE_ADDR": "192.168.1.100",
            "DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment",
            "ENGINE_VERSION": "0.21.0-a2",   # 芯片(A2/910B)由 engine-version 后缀确定
        },
        "model_config": {
            "architecture": "GlmMoeDsaForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "v4pro-a3-dual": {
        "description": "DeepSeek-V4-Pro + 910C(A3) 双机32卡 dp_deployment (DP=2)",
        "user_cli": {
            "model-name": "DeepSeek-V4-Pro",
            "engine": "vllm_ascend",
            "device-count": 16,
            "distributed": True,
            # A3 会自动开启投机，用户不传 --enable-speculative-decode
        },
        "orchestration_env": {
            "NNODES": "2",
            "NODE_IPS": "192.168.1.100,192.168.1.101",
            "HEAD_NODE_ADDR": "192.168.1.100",
            "DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment",
            "WINGS_ASCEND_PLATFORM": "a3",
        },
        "model_config": {
            "architecture": "DeepseekV4ForCausalLM",
            "quantization_config": {"quant_method": "ascend"},
            "quantize": "w4a8_dynamic",
        },
    },
    "qwen36-35b-a3b": {
        "description": "Qwen3.6-35B-A3B(MoE) + 910C(A3) 单机2卡 (FC+think+spec, 验证 qwen3_coder/enforce_eager/MoE recipe)",
        "user_cli": {
            "model-name": "Qwen3.6-35B-A3B",
            "engine": "vllm_ascend",
            "device-count": 2,
            "enable-speculative-decode": True,
            "enable-auto-tool-choice": True,
            "enable-auto-think-choice": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "mp",
            "WINGS_ASCEND_PLATFORM": "a3",
        },
        "model_config": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "qwen35-397b-a17b": {
        "description": "Qwen3.5-397B-A17B(MoE) + 910C(A3) 单机16卡 (FC+think, spec默认关 → 验证无 speculative_config)",
        "user_cli": {
            "model-name": "Qwen3.5-397B-A17B",
            "engine": "vllm_ascend",
            "device-count": 16,
            "enable-auto-tool-choice": True,
            "enable-auto-think-choice": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "mp",
            "WINGS_ASCEND_PLATFORM": "a3",
        },
        "model_config": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "qwen36-27b": {
        "description": "Qwen3.6-27B(dense) + 910C(A3) 单机2卡 (FC+think+spec, 验证 mamba_cache_mode=align + enforce_eager)",
        "user_cli": {
            "model-name": "Qwen3.6-27B",
            "engine": "vllm_ascend",
            "device-count": 2,
            "enable-speculative-decode": True,
            "enable-auto-tool-choice": True,
            "enable-auto-think-choice": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "mp",
            "WINGS_ASCEND_PLATFORM": "a3",
        },
        "model_config": {
            "architecture": "Qwen3_5ForConditionalGeneration",
        },
    },
    "minimax-m3-nv-8": {
        "description": "MiniMax-M3-MXFP8 + NVIDIA 单机8卡 (仅NV; 单模型 enforce-eager via JSON; GQA+MSA/MXFP8, 无 trust-remote-code/parser 对齐手工命令)",
        "user_cli": {
            "model-name": "MiniMax-M3-MXFP8",
            "engine": "vllm",
            "device-count": 8,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "mp",
        },
        "model_config": {
            "architecture": "MiniMaxM3SparseForConditionalGeneration",
            "quantization_config": {"quant_method": "mxfp8"},
        },
    },
    "kimi-k27-ascend-16": {
        "description": "Kimi-K2.7-Code + 910C(A3) 单机16卡 (MLA + EP, 验证 DP/TP=纯TP16不走DP + KimiK25 env/parser/FULL_DECODE_ONLY)",
        "user_cli": {
            "model-name": "Kimi-K2.7-Code",
            "engine": "vllm_ascend",
            "device-count": 16,
            "enable-auto-tool-choice": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "dp_deployment",
            "WINGS_ASCEND_PLATFORM": "a3",
            "ENGINE_VERSION": "0.21.0-a3",
            "ENABLE_KV_OFFLOAD": "true",
            "ENABLE_KV_MEM_OFFLOAD": "true",
            "KV_MEM_OFFLOAD_SIZE": "20",
        },
        "model_config": {
            "architecture": "KimiK25ForConditionalGeneration",
            "quantization_config": {"quant_method": "ascend"},
        },
    },
    "sglang-think-on": {
        "description": "SGLang + Qwen3.6-27B + think/tool ON (验证 sglang 启动期不支持思考开关 → 仅告警、不注入 chat-template-kwargs)",
        "user_cli": {
            "model-name": "Qwen3.6-27B",
            "engine": "sglang",
            "device-count": 8,
            "enable-auto-tool-choice": True,
            "enable-auto-think-choice": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "mp",
        },
        "model_config": {
            "architecture": "Qwen3_5ForConditionalGeneration",
        },
    },
    "mindie-think-on": {
        "description": "MindIE + DeepSeek-V3.1 + think ON (验证 mindie 启动期不支持思考开关 → 仅告警)",
        "user_cli": {
            "model-name": "DeepSeek-V3.1",
            "engine": "mindie",
            "device-count": 8,
            "enable-auto-tool-choice": True,
            "enable-auto-think-choice": True,
        },
        "orchestration_env": {
            "DISTRIBUTED_EXECUTOR_BACKEND": "mp",
        },
        "model_config": {
            "architecture": "DeepseekV3ForCausalLM",
        },
    },
}


def create_mock_model_dir(model_config: dict) -> str:
    """创建模拟模型目录（含 config.json）。架构/量化等模型自带信息从这里被引擎发现。"""
    build_dir = os.path.join(SCRIPT_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)
    model_dir = tempfile.mkdtemp(prefix="model_", dir=build_dir).replace("\\", "/")
    architecture = model_config["architecture"]
    config = {
        "architectures": [architecture],
        "model_type": "deepseek_v4" if "Deepseek" in architecture else "glm4",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 64,
    }
    # model_config 里除 architecture 外的键（quantization_config / quantize…）合并进 config.json
    config.update({k: v for k, v in model_config.items() if k != "architecture"})
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return model_dir


def reset_managed_env() -> None:
    """清掉上一场景注入的所有 env，模拟「每个场景 = 一个全新的 K8s pod」。

    这一步是 dry_run 真实性的前提：wings_start.sh 对没传的 CLI「不 export」靠的是
    每个 pod 是全新进程；而 dry_run 在同一进程里串行跑多场景，必须显式清理，
    否则会出现 ENGINE_VERSION / WINGS_ASCEND_PLATFORM 等跨场景串味。
    """
    for key in _PORT_ENV_KEYS | _INFRA_ENV_KEYS:
        os.environ.pop(key, None)
    for flag in WINGS_START_CLI_FLAGS:
        os.environ.pop(_cli_flag_to_env(flag), None)
    os.environ.pop("MODEL_PATH", None)


def apply_orchestration_env(scenario: dict, model_dir: str) -> None:
    """模拟编排层/K8s 在 launcher 进程启动「之前」就注入 pod env（非用户 CLI）。"""
    user_cli = scenario["user_cli"]
    orch = dict(scenario.get("orchestration_env", {}))
    engine = str(user_cli.get("engine", "vllm"))
    # 根据引擎推断设备类型：vllm_ascend → ascend，否则 nvidia（决定加载哪个 *_default.json）
    device_type = "ascend" if "ascend" in engine else "nvidia"
    head = orch.get("HEAD_NODE_ADDR", "127.0.0.1")
    shared_vol = tempfile.mkdtemp(prefix="sv_", dir=os.path.join(SCRIPT_DIR, "build")).replace("\\", "/")

    base = {
        "NODE_RANK": "0",                 # 注：argparse 不读它(default=0)，靠 Master 分发 CLI 注入
        "POD_IP": "192.168.1.100",
        "RANK_IP": "192.168.1.100",
        "NODE_IPS": "192.168.1.100",      # 单机默认本机；双机由 orchestration_env 覆盖
        "HEAD_NODE_ADDR": head,
        "MASTER_IP": head,
        "NETWORK_INTERFACE": "eth0",
        "ENABLE_ACCEL": "false",
        "ENABLE_KV_OFFLOAD": "false",       # 默认不卸载；KV 卸载场景由 orchestration_env 覆盖
        "WINGS_DEVICE": device_type,
        "WINGS_ASCEND_PLATFORM": "",      # 默认空；用 platform 的场景覆盖（A2/A3 细分）
        "ENGINE_VERSION": "",             # 默认空；用 engine-version 后缀的场景覆盖
        "SHARED_VOLUME_PATH": shared_vol,
        "MODEL_PATH": model_dir,          # 模型权重目录由编排挂载/注入
    }
    base.update(orch)                     # 场景的编排 env 覆盖基础设施默认
    os.environ.update(base)


def _apply_port_proxy_rule(user_cli: dict) -> str:
    """复刻 wings_start.sh 的端口/代理逻辑（234-246 / 272-275 / 305-309），返回 proxy_port。

    代理开（默认）：proxy_port = 用户 --port | PROXY_PORT | 18000；backend 固定 17000。
    代理关：backend = 用户 --port | 18000。
    无论开关，脚本 309 行都无条件 `export PORT=$PROXY_PORT`，这里忠实复刻。
    """
    enable_proxy = os.environ.get("ENABLE_REASON_PROXY", "true").lower() != "false"
    default_port = "18000"
    user_port = str(user_cli.get("port") or os.environ.get("PORT") or "")
    if enable_proxy:
        proxy_port = os.environ.get("PROXY_PORT") or user_port or default_port
    else:
        # 代理关：backend 取用户口；PROXY_PORT 缺省 18000（脚本 307 行）
        proxy_port = os.environ.get("PROXY_PORT") or default_port
    os.environ["ENABLE_REASON_PROXY"] = "true" if enable_proxy else "false"
    os.environ["PROXY_PORT"] = proxy_port
    os.environ["PORT"] = proxy_port       # 复刻脚本 309 行：start_args_compat 的 --port 默认从 PORT 读
    return proxy_port


def simulate_wings_start(user_cli: dict) -> list[str]:
    """复刻 wings_control/wings_start.sh 的入参处理：校验 → 端口/代理特例 → 双路下发。

    双路下发（脚本 262-350）：每个用户 CLI 既 `export` 成 env，又拼进 APP_ARGS。
    其中 env 路径在 dry_run 下尤其关键 —— config_loader._detect_explicit_cli_keys()
    在本进程靠 env（不是 sys.argv）识别「用户显式覆盖」，少了 env 路径，
    --gpu-memory-utilization / --quantization 等覆盖会被当默认值丢掉。

    返回 APP_ARGS（交给 parse_launch_args，等价于真实 `exec python -m wings_control "$@"`）。
    """
    bad = set(user_cli) - WINGS_START_CLI_FLAGS
    if bad:
        raise ValueError(
            f"user_cli 含 wings_start.sh 不支持的 flag: {sorted(bad)}；"
            "拓扑/平台/芯片应放 orchestration_env，架构/量化放 model_config"
        )
    if not user_cli.get("model-name"):
        raise ValueError("user_cli 必须含 model-name（wings_start.sh 必填项）")

    app_args: list[str] = []
    for flag, val in user_cli.items():
        if flag == "port":
            continue  # 端口走 proxy 特例统一处理
        env_name = _cli_flag_to_env(flag)
        if flag in WINGS_START_BOOL_FLAGS:
            # 复刻脚本：布尔仅在 true 时 export + 进 APP_ARGS；false/缺省一律不下发
            if val:
                app_args.append(f"--{flag}")
                os.environ[env_name] = "true"
        else:
            app_args += [f"--{flag}", str(val)]
            os.environ[env_name] = str(val)

    proxy_port = _apply_port_proxy_rule(user_cli)
    app_args += ["--port", proxy_port]
    return app_args


def _write_command(output_dir: str, filename: str, command: str) -> None:
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(command)


def run_dry_run(scenario_name: str, scenario: dict) -> None:
    """执行 dry-run 并输出 start_command.sh。"""
    from core.start_args_compat import parse_launch_args
    from core.port_plan import derive_port_plan
    from core.wings_entry import build_launcher_plan
    from config.settings import settings

    # 每个场景 = 全新 pod
    reset_managed_env()
    # ① 模型自带（架构/量化）→ mock config.json
    model_dir = create_mock_model_dir(scenario["model_config"])
    # ② 编排层在进程启动前注入 env
    apply_orchestration_env(scenario, model_dir)
    # ③ wings_start.sh 解析用户 CLI → 双路下发 → APP_ARGS
    app_args = simulate_wings_start(scenario["user_cli"])

    orch = scenario.get("orchestration_env", {})
    logger.info("=" * 80)
    logger.info("场景: %s — %s", scenario_name, scenario["description"])
    logger.info("  user_cli          = %s", scenario["user_cli"])
    logger.info("  orchestration_env = %s", orch)
    logger.info("  APP_ARGS          = %s", " ".join(app_args))
    logger.info("=" * 80)

    output_dir = os.path.join(SCRIPT_DIR, "build", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Node 0 —— Master 分发动态注入 node-rank（argparse 不读 NODE_RANK env，default=0）
    la0 = parse_launch_args(app_args + ["--node-rank", "0"])
    port_plan = derive_port_plan(
        port=la0.port,
        enable_reason_proxy=settings.ENABLE_REASON_PROXY,
        health_port=settings.HEALTH_PORT,
    )
    plan = build_launcher_plan(la0, port_plan)
    _write_command(output_dir, f"start_command_{scenario_name}_node0.sh", plan.command)
    logger.info("Node 0 → start_command_%s_node0.sh (%d bytes)", scenario_name, len(plan.command))

    # 多节点：模拟 Master 给 node1 分发（改 NODE_RANK/RANK_IP/POD_IP env + --node-rank 1）
    nnodes = int(orch.get("NNODES", 1))
    if nnodes > 1:
        node_ips = orch.get("NODE_IPS", "")
        rank_ip = node_ips.split(",")[1] if "," in node_ips else "192.168.1.101"
        os.environ["NODE_RANK"] = "1"
        os.environ["RANK_IP"] = rank_ip
        os.environ["POD_IP"] = rank_ip
        la1 = parse_launch_args(app_args + ["--node-rank", "1"])
        plan1 = build_launcher_plan(la1, port_plan)
        _write_command(output_dir, f"start_command_{scenario_name}_node1.sh", plan1.command)
        logger.info("Node 1 → start_command_%s_node1.sh (%d bytes)", scenario_name, len(plan1.command))

    # 打印关键参数
    ec = plan.merged_params.get("engine_config", {})
    logger.info("  engine_config 关键字段:")
    for k in ["tensor_parallel_size", "data_parallel_size", "max_model_len",
              "enable_expert_parallel", "quantization", "enable_prefix_caching",
              "enable_chunked_prefill", "compilation_config", "additional_config",
              "speculative_config"]:
        if k in ec:
            logger.info("    %s = %s", k, ec[k])

    # 提取最终 vllm 命令
    for line in plan.command.splitlines():
        stripped = line.strip()
        if stripped.startswith("vllm serve") or stripped.startswith("exec python3 -m vllm") or stripped.startswith("exec vllm"):
            print(f"\n【{scenario_name} Node 0 最终命令】")
            print(stripped)
            break

    # 清理临时模型目录
    import shutil
    shutil.rmtree(model_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Wings-infer Dry-Run: 生成 start_command.sh")
    parser.add_argument("--scenario", "-s", choices=list(SCENARIOS.keys()),
                        help="预置场景名称")
    parser.add_argument("--list", "-l", action="store_true",
                        help="列出所有预置场景")
    args = parser.parse_args()

    if args.list:
        print("可用场景:")
        for name, cfg in SCENARIOS.items():
            print(f"  {name:20s} — {cfg['description']}")
        return

    if not args.scenario:
        # 默认跑所有场景
        for name, cfg in SCENARIOS.items():
            run_dry_run(name, cfg)
    else:
        run_dry_run(args.scenario, SCENARIOS[args.scenario])

    logger.info("=" * 80)
    logger.info("DRY RUN COMPLETE — 输出目录: build/output/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
