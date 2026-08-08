# Kimi-K3 Ascend 910B 四机适配记录

## 1. 文档状态

- 整理日期：2026-07-30
- 目标模型：Kimi-K3 / `KimiK3ForConditionalGeneration`
- 目标硬件：Ascend 910B2C，每节点 16 卡，共 4 节点 64 卡
- 推理框架：vLLM 0.23.0 + vLLM-Ascend
- 当前决策：暂不在项目代码中启用，保留本文作为后续重新适配依据
- 已验证结论：原最小适配实现能够生成符合标准形态的 rank0～rank3 命令；相关回归曾达到 `224 passed`

本文记录的是已经验证过的适配方法。重新启用时，应先核对实际镜像、模型权重和标准启动命令是否发生变化，不能直接把本文中的 910B 参数用于 910C。

## 2. 标准部署拓扑

标准部署采用 vLLM 原生 `dp_deployment`：

| 项目 | 标准值 |
| --- | --- |
| 节点数 | 4 |
| 每节点设备数 | 16 |
| 总设备数 | 64 |
| Tensor Parallel | 16 |
| Data Parallel | 4 |
| Data Parallel Local | 1 |
| Master / DP Address | `nodeIps` 第一个 IP |
| API 端口 | 18000 |
| DP RPC 端口 | 27777 |
| 通信网卡 | bond0，实际由 `NETWORK_INTERFACE` 提供 |

`nodeIps` 必须包含全部分布式节点 IP，并按 rank 顺序排列：

```text
70.189.109.97,70.189.109.102,70.189.109.103,70.189.109.104
```

含义如下：

| node_rank | 节点 IP | 角色 | data-parallel-start-rank |
| --- | --- | --- | --- |
| 0 | 70.189.109.97 | Master + API Server | 0 |
| 1 | 70.189.109.102 | headless Worker | 1 |
| 2 | 70.189.109.103 | headless Worker | 2 |
| 3 | 70.189.109.104 | headless Worker | 3 |

该场景不是 Ray 集群，也不是 NVIDIA Kimi-K3 使用的原生 MP。每台机器内部 16 卡组成一个 TP 组，四台机器分别提供一个 DP replica。

## 3. 标准环境变量

四台机器使用相同的通信环境结构，本机 IP 和网卡由运行环境动态解析：

```bash
export HCCL_IF_IP=$VLLM_HOST_IP
export GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE
export TP_SOCKET_IFNAME=$NETWORK_INTERFACE
export HCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=800
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

约束：

- 单节点设备数必须是 16，否则当前 910B recipe 应直接拒绝启动。
- `HCCL_IF_IP` 必须是本机 RoCE 通信 IP，不能固定写成 Master IP。
- `NETWORK_INTERFACE` 应由节点运行环境提供，例如 `bond0`。
- `VLLM_ENGINE_READY_TIMEOUT_S` 可由环境变量覆盖，未配置时使用 7200。
- `HCCL_BUFFSIZE=800` 是本次标准命令值，不应继承其他 DeepSeek/GLM DP recipe。

## 4. 标准 vLLM 参数

模型静态参数如下：

```text
--served-model-name kimi-k3
--allowed-local-media-path /
--trust-remote-code
--tokenizer-mode kimi_k3
--tensor-parallel-size 16
--enable-prefix-caching
--enable-expert-parallel
--max-num-seqs 16
--max-model-len 200000
--max-num-batched-tokens 8192
--gpu-memory-utilization 0.95
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
--mm-processor-cache-gb 0
--additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true}'
--mm-encoder-tp-mode data
--limit-mm-per-prompt '{"vision_chunk":40}'
```

工具调用和思维解析继续遵循 Wings-Control 现有开关契约：

- 模型能力配置提供 `tool_call_parser=kimi_k3`。
- `enable_auto_tool_choice=true` 时才输出 `--enable-auto-tool-choice --tool-call-parser kimi_k3`。
- `enable_auto_think_choice=true` 时才输出 `--reasoning-parser kimi_k3`。
- 不应为了对齐手工标准命令而修改全局 function-call / reasoning 开关语义。

## 5. 主从命令形态

### 5.1 rank0

```bash
vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 18000 \
  --served-model-name kimi-k3 \
  --allowed-local-media-path / \
  --trust-remote-code \
  --tokenizer-mode kimi_k3 \
  --tensor-parallel-size 16 \
  --data-parallel-size 4 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 0 \
  --data-parallel-address 70.189.109.97 \
  --data-parallel-rpc-port 27777 \
  --enable-prefix-caching \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --max-model-len 200000 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true}' \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"vision_chunk":40}' \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3
```

### 5.2 rank1～rank3

Worker 与 rank0 的模型参数保持一致，仅角色参数不同：

```bash
vllm serve "$MODEL_PATH" \
  --port 18000 \
  --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank "$NODE_RANK" \
  --data-parallel-address 70.189.109.97 \
  --data-parallel-rpc-port 27777 \
  ...其他模型参数与 rank0 相同
```

关键差异：

- Worker 删除 `--host`。
- Worker 保留 `--port 18000`，这是该标准命令与项目通用 DP Worker 逻辑的差异。
- Worker 增加 `--headless`。
- rank1、rank2、rank3 的 `--data-parallel-start-rank` 分别为 1、2、3。
- rank0 也必须显式输出 `--data-parallel-start-rank 0`。

## 6. 最小嵌入式适配方法

重新适配时应复用现有 `nodeIps`、Master 分发、`dp_deployment` 拓扑计算和启动脚本生成链路，只增加 Kimi-K3 Ascend 的模型选择条件和 910B recipe。

### 6.1 `wings_control/config/defaults/ascend_default.json`

在 `model_deploy_config.llm` 下新增：

```json
{
  "KimiK3ForConditionalGeneration": {
    "Kimi-K3-Ascend910B": {
      "vllm_ascend_distributed": {
        "use_vllm_serve": true,
        "trust_remote_code": true,
        "served_model_name": "kimi-k3",
        "allowed-local-media-path": "/",
        "tokenizer_mode": "kimi_k3",
        "tensor_parallel_size": 16,
        "enable_prefix_caching": true,
        "enable_expert_parallel": true,
        "max_num_seqs": 16,
        "max_model_len": 200000,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.95,
        "compilation_config": {
          "cudagraph_mode": "FULL_DECODE_ONLY"
        },
        "mm_processor_cache_gb": 0,
        "additional_config": {
          "enable_cpu_binding": true,
          "enable_flashcomm1": true
        },
        "mm_encoder_tp_mode": "data",
        "limit_mm_per_prompt": {
          "vision_chunk": 40
        },
        "tool_call_parser": "kimi_k3"
      }
    }
  }
}
```

必须使用带 `Ascend910B` 后缀的 profile，避免 910C 误命中。

### 6.2 `wings_control/core/config_loader.py`

需要完成三件事：

1. 在 `_select_ascend_engine()` 的 vLLM-Ascend 架构集合中加入 `KimiK3ForConditionalGeneration`。
2. 在 `_handle_vllm_distributed()` 中将 `KimiK3ForConditionalGeneration + is_ascend` 路由到现有 `dp_deployment` 分支。
3. 设置两个仅供内部启动脚本消费的标记：

```python
cmd_params["_force_data_parallel_start_rank_on_rank0"] = True
cmd_params["_preserve_dp_worker_port"] = True
```

DP RPC 端口采用：

```python
rpc_port = os.getenv("VLLM_DP_RPC_PORT") or distributed_config[
    "vllm_distributed"
].get("rpc_port", 27071)
```

部署该标准场景时设置：

```bash
export VLLM_DP_RPC_PORT=27777
```

不要把 27777 改成所有 DP 模型的全局默认值。

### 6.3 `wings_control/core/wings_entry.py`

项目通用 Worker 参数准备逻辑会删除 `host` 和 `port`。Kimi-K3 Ascend 标准 Worker 要求保留端口，因此在 `_prepare_merged_params()` 的非 rank0 分支中：

```python
if merged.get("_preserve_dp_worker_port"):
    merged["port"] = port_plan.backend_port
    engine_cfg["port"] = port_plan.backend_port
else:
    merged.pop("port", None)
    engine_cfg.pop("port", None)
```

`host` 仍然必须删除，不能让 headless Worker 启动 API Server。

### 6.4 `wings_control/engines/vllm_distributed.py`

增加独立的 `_build_kimi_k3_910b_dp_env_commands()`：

- 校验 `device_count == 16`。
- 输出第 3 节的标准环境变量。
- `ASCEND_RT_VISIBLE_DEVICES` 根据 16 卡生成。

在 `_build_ascend_dp_env_commands()` 中根据模型架构和卡型分发：

```python
if model_architecture == "KimiK3ForConditionalGeneration":
    platform = vllm_adapter.ascend_platform_from_runtime(params)
    if platform == "a2":
        return _build_kimi_k3_910b_dp_env_commands(params, net_if)
    raise ValueError(
        "Kimi-K3 Ascend dp_deployment has no validated 910C/A3 environment profile"
    )
```

同时扩展 `_build_dp_exec_command()`：

- rank0 根据内部标记输出 `--data-parallel-start-rank 0`。
- Worker 始终删除 `--host`。
- 只有 `_preserve_dp_worker_port` 未启用时才删除 `--port`。

### 6.5 测试文件

需要恢复以下覆盖：

- `tests/test_default_config.py`
  - Kimi-K3 Ascend 自动选择 `vllm_ascend`
  - 四机 Ascend 路由到 `dp_deployment`
  - 910B profile 完整性和 910C 不误命中
  - Ascend profile 计数同步增加 1
- `tests/test_vllm_distributed.py`
  - rank0 标准环境和 DP 参数
  - rank1～rank3 保留端口、删除 host、增加 headless
  - 910C recipe 未验证时拒绝套用 910B 配置
- `tests/test_distributed_backend_propagation.py`
  - Worker 参数合并层不会提前删除 Kimi-K3 标准端口

## 7. 910C 后续复用边界

910C 可以复用：

- `nodeIps` 与第一个 IP 作为 Master 的节点编排约定
- `dp_deployment` 路由
- rank0 / headless Worker 命令生成
- DP Address、DP RPC Port、DP rank 计算
- function-call / reasoning parser 的能力配置和开关
- 配置合并与分发链路

910C 不能直接复用：

- 910B 的 TP=16、DP-local=1 假设
- `HCCL_BUFFSIZE=800`
- 910B 环境变量全集
- `ASCEND_RT_VISIBLE_DEVICES=0..15`
- 910B 的 compilation/additional config
- 910B 的显存、并发和上下文参数

收到 910C 标准命令后，应新增：

1. `Kimi-K3-Ascend910C` 独立默认 profile。
2. `_build_kimi_k3_910c_dp_env_commands()` 独立环境 recipe。
3. 910C 卡型精确命中和错误卡型隔离测试。
4. 根据 910C 实际 TP/DP 重新验证四个节点命令。

在 910C 标准 recipe 未确认前，推荐 fail-fast，禁止静默回退到 910B。

## 8. 重新启用时的验证清单

### 8.1 静态检查

```powershell
python -m py_compile `
  wings_control/core/config_loader.py `
  wings_control/core/wings_entry.py `
  wings_control/engines/vllm_distributed.py

python -c "import json; json.load(open(r'wings_control/config/defaults/ascend_default.json', encoding='utf-8'))"
git diff --check
```

### 8.2 定向回归

```powershell
python -m pytest -q `
  tests/test_default_config.py `
  tests/test_vllm_distributed.py `
  tests/test_distributed_backend_propagation.py `
  tests/test_kv_offload_gating.py
```

历史适配结果为：

```text
224 passed
```

### 8.3 最终命令核对

必须展开并逐节点比较 rank0、rank1、rank2、rank3：

- 4 个节点均为 `--tensor-parallel-size 16`
- 4 个节点均为 `--data-parallel-size 4`
- 4 个节点均为 `--data-parallel-size-local 1`
- 4 个节点均指向相同 Master IP 和 RPC 端口 27777
- rank0 有 `--host 0.0.0.0 --port 18000`
- rank0 显式包含 `--data-parallel-start-rank 0`
- rank1～rank3 无 `--host`
- rank1～rank3 保留 `--port 18000`
- rank1～rank3 包含 `--headless`
- rank1～rank3 的 start rank 分别为 1、2、3
- 网卡必须是每个节点实际可用的 RoCE 网卡

## 9. 本次回退范围

暂不适配时，应仅删除以下改动并保留本文：

- `ascend_default.json` 的 Kimi-K3 910B profile
- `config_loader.py` 的 Kimi-K3 Ascend 引擎选择、DP 路由、RPC/内部标记
- `wings_entry.py` 的 Kimi-K3 Worker 端口保留分支
- `vllm_distributed.py` 的 Kimi-K3 910B 环境 recipe 和 Worker 端口扩展
- 三个测试文件中的 Kimi-K3 Ascend 用例及对应 profile 计数调整

不得回退已经存在的 NVIDIA H20 Kimi-K3 四机 MP 支持，也不得改动与本适配无关的工作区文件。
