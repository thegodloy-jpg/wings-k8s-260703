# Kimi-K3 四机 H20 dry-run 完备差异报告

- 总体部署契约：**PASS**
- 严格参考命令逐字段一致性：**FAIL**
- 应用 Wings 架构允许差异后：**PASS**
- 严重/未预期差异：**0**
- backend 传播：`ray`（CLI 默认）→ `mp`（rank0 合并）→ `mp`（三个 Worker payload）
- 拓扑：`7.6.25.57,7.6.25.58,7.6.25.59,7.6.25.60`，master=`7.6.25.57:29501`，TP=32，world size=32

## 结论

四个节点均通过部署契约校验：rank0 提供 frontend，rank1-3 使用 headless；所有节点均为 MP、TP32、相同 rendezvous；Worker 使用本机模型路径；未生成 Ray 或 data-parallel 参数。

严格比较为 FAIL，但只包含下列已确认的 Wings 架构差异，不属于四机 MP 适配缺陷。

## 节点结果

| 节点 | 角色 | IP | 严格比较 | 差异数 | 部署策略比较 | 契约检查 |
|---:|---|---|---|---:|---|---|
| 0 | master | `7.6.25.57` | FAIL | 3 | PASS | 27/27 PASS |
| 1 | worker | `7.6.25.58` | FAIL | 1 | PASS | 29/29 PASS |
| 2 | worker | `7.6.25.59` | FAIL | 1 | PASS | 29/29 PASS |
| 3 | worker | `7.6.25.60` | FAIL | 1 | PASS | 29/29 PASS |

## 四节点启动命令逐项对比

以下命令均来自本次 dry-run 产物，不是根据配置手工拼接。参数顺序不参与语义比较，环境变量、模型路径和 CLI 值参与比较。

### node0：Master（7.6.25.57）

角色要求：

- 对外提供 OpenAI frontend，不包含 `--headless`。
- 使用 master 本机模型路径 `/var/aispace/model/Kimi-K3`。
- 保留 `--enable-auto-tool-choice`、`--tool-call-parser`、`--reasoning-parser`。
- 使用统一 rendezvous：`7.6.25.57:29501`。

标准命令：

```bash
export NCCL_SOCKET_IFNAME=enp66s0f1
export GLOO_SOCKET_IFNAME=enp66s0f1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export VLLM_SSM_CONV_STATE_LAYOUT=DS

vllm serve /var/aispace/model/Kimi-K3 \
  --trust-remote-code \
  --gpu-memory-utilization 0.98 \
  --tensor-parallel-size 32 \
  --nnodes 4 \
  --node-rank 0 \
  --master-addr 7.6.25.57 \
  --master-port 29501 \
  --no-enable-flashinfer-autotune \
  -cc.pass_config.fuse_allreduce_rms=False \
  --moe-backend marlin \
  --disable-custom-all-reduce \
  --distributed-timeout-seconds 1200 \
  --max-num-seqs 32 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --served-model-name kimi_k3 \
  --host 0.0.0.0 \
  --max-num-batched-tokens 8192 \
  --max-model-len 40960 \
  --port 8000
```

项目实际生成命令：

```bash
export NCCL_SOCKET_IFNAME=enp66s0f1
export GLOO_SOCKET_IFNAME=enp66s0f1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export VLLM_SSM_CONV_STATE_LAYOUT=DS

vllm serve /var/aispace/model/Kimi-K3 \
  --trust-remote-code \
  --served-model-name kimi_k3 \
  --gpu-memory-utilization 0.98 \
  --no-enable-flashinfer-autotune \
  --moe-backend marlin \
  --disable-custom-all-reduce \
  --distributed-timeout-seconds 1200 \
  --max-num-seqs 32 \
  --tool-call-parser kimi_k3 \
  --max-num-batched-tokens 8192 \
  --max-model-len 40960 \
  --reasoning-parser kimi_k3 \
  --host 7.6.25.57 \
  --port 17000 \
  --enable-auto-tool-choice \
  --tensor-parallel-size 32 \
  -cc.pass_config.fuse_allreduce_rms=False \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 0 \
  --master-addr 7.6.25.57 \
  --master-port 29501
```

node0 语义差异：

```diff
- --host 0.0.0.0
+ --host 7.6.25.57

- --port 8000
+ --port 17000

+ --distributed-executor-backend mp
```

其余环境变量、模型路径、TP、节点数、rank、rendezvous、Kimi parser、Marlin、超时和批处理参数全部一致。

完整产物：

- `node0/standard_exec_command.txt`
- `node0/actual_exec_command.txt`
- `node0/start_command.sh`
- `node0/strict_comparison.md`

### node1：Worker（7.6.25.58）

角色要求：

- 使用 `--headless`，不启动 frontend。
- 使用 Worker 本机模型路径 `/var/ai-model/Kimi-K3/`。
- 不包含 host、port、tool parser、reasoning parser 和 auto-tool 参数。
- 使用 `node-rank=1`，连接 `7.6.25.57:29501`。

标准命令：

```bash
export NCCL_SOCKET_IFNAME=ens3f3
export GLOO_SOCKET_IFNAME=ens3f3
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export VLLM_SSM_CONV_STATE_LAYOUT=DS

vllm serve /var/ai-model/Kimi-K3/ \
  --trust-remote-code \
  --gpu-memory-utilization 0.98 \
  --tensor-parallel-size 32 \
  --nnodes 4 \
  --node-rank 1 \
  --master-addr 7.6.25.57 \
  --master-port 29501 \
  --no-enable-flashinfer-autotune \
  -cc.pass_config.fuse_allreduce_rms=False \
  --moe-backend marlin \
  --disable-custom-all-reduce \
  --distributed-timeout-seconds 1200 \
  --max-num-seqs 32 \
  --headless \
  --served-model-name kimi_k3 \
  --max-num-batched-tokens 8192 \
  --max-model-len 40960
```

项目实际生成命令：

```bash
export NCCL_SOCKET_IFNAME=ens3f3
export GLOO_SOCKET_IFNAME=ens3f3
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export VLLM_SSM_CONV_STATE_LAYOUT=DS

vllm serve /var/ai-model/Kimi-K3/ \
  --trust-remote-code \
  --served-model-name kimi_k3 \
  --gpu-memory-utilization 0.98 \
  --no-enable-flashinfer-autotune \
  --moe-backend marlin \
  --disable-custom-all-reduce \
  --distributed-timeout-seconds 1200 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --max-model-len 40960 \
  --tensor-parallel-size 32 \
  -cc.pass_config.fuse_allreduce_rms=False \
  --headless \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 1 \
  --master-addr 7.6.25.57 \
  --master-port 29501
```

node1 语义差异：

```diff
+ --distributed-executor-backend mp
```

除项目显式固定 MP 外，其余环境变量、模型路径、headless 角色、TP、rank、rendezvous 和推理参数全部一致。

完整产物：

- `node1/standard_exec_command.txt`
- `node1/actual_exec_command.txt`
- `node1/start_command.sh`
- `node1/strict_comparison.md`

### node2：Worker（7.6.25.59）

角色要求与 node1 相同，仅 `node-rank` 和节点 IP 不同。

标准命令：

```bash
export NCCL_SOCKET_IFNAME=ens3f3
export GLOO_SOCKET_IFNAME=ens3f3
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export VLLM_SSM_CONV_STATE_LAYOUT=DS

vllm serve /var/ai-model/Kimi-K3/ \
  --trust-remote-code \
  --gpu-memory-utilization 0.98 \
  --tensor-parallel-size 32 \
  --nnodes 4 \
  --node-rank 2 \
  --master-addr 7.6.25.57 \
  --master-port 29501 \
  --no-enable-flashinfer-autotune \
  -cc.pass_config.fuse_allreduce_rms=False \
  --moe-backend marlin \
  --disable-custom-all-reduce \
  --distributed-timeout-seconds 1200 \
  --max-num-seqs 32 \
  --headless \
  --served-model-name kimi_k3 \
  --max-num-batched-tokens 8192 \
  --max-model-len 40960
```

项目实际生成命令：

```bash
export NCCL_SOCKET_IFNAME=ens3f3
export GLOO_SOCKET_IFNAME=ens3f3
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export VLLM_SSM_CONV_STATE_LAYOUT=DS

vllm serve /var/ai-model/Kimi-K3/ \
  --trust-remote-code \
  --served-model-name kimi_k3 \
  --gpu-memory-utilization 0.98 \
  --no-enable-flashinfer-autotune \
  --moe-backend marlin \
  --disable-custom-all-reduce \
  --distributed-timeout-seconds 1200 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --max-model-len 40960 \
  --tensor-parallel-size 32 \
  -cc.pass_config.fuse_allreduce_rms=False \
  --headless \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 2 \
  --master-addr 7.6.25.57 \
  --master-port 29501
```

node2 语义差异：

```diff
+ --distributed-executor-backend mp
```

完整产物：

- `node2/standard_exec_command.txt`
- `node2/actual_exec_command.txt`
- `node2/start_command.sh`
- `node2/strict_comparison.md`

### node3：Worker（7.6.25.60）

角色要求与 node1 相同，仅 `node-rank` 和节点 IP 不同。

标准命令：

```bash
export NCCL_SOCKET_IFNAME=ens3f3
export GLOO_SOCKET_IFNAME=ens3f3
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export VLLM_SSM_CONV_STATE_LAYOUT=DS

vllm serve /var/ai-model/Kimi-K3/ \
  --trust-remote-code \
  --gpu-memory-utilization 0.98 \
  --tensor-parallel-size 32 \
  --nnodes 4 \
  --node-rank 3 \
  --master-addr 7.6.25.57 \
  --master-port 29501 \
  --no-enable-flashinfer-autotune \
  -cc.pass_config.fuse_allreduce_rms=False \
  --moe-backend marlin \
  --disable-custom-all-reduce \
  --distributed-timeout-seconds 1200 \
  --max-num-seqs 32 \
  --headless \
  --served-model-name kimi_k3 \
  --max-num-batched-tokens 8192 \
  --max-model-len 40960
```

项目实际生成命令：

```bash
export NCCL_SOCKET_IFNAME=ens3f3
export GLOO_SOCKET_IFNAME=ens3f3
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export VLLM_SSM_CONV_STATE_LAYOUT=DS

vllm serve /var/ai-model/Kimi-K3/ \
  --trust-remote-code \
  --served-model-name kimi_k3 \
  --gpu-memory-utilization 0.98 \
  --no-enable-flashinfer-autotune \
  --moe-backend marlin \
  --disable-custom-all-reduce \
  --distributed-timeout-seconds 1200 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --max-model-len 40960 \
  --tensor-parallel-size 32 \
  -cc.pass_config.fuse_allreduce_rms=False \
  --headless \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank 3 \
  --master-addr 7.6.25.57 \
  --master-port 29501
```

node3 语义差异：

```diff
+ --distributed-executor-backend mp
```

完整产物：

- `node3/standard_exec_command.txt`
- `node3/actual_exec_command.txt`
- `node3/start_command.sh`
- `node3/strict_comparison.md`

## 已确认差异

| 节点 | 字段 | 标准值 | 实际值 | 分类 | 说明 |
|---|---|---|---|---|---|
| node0 | `host` | `0.0.0.0` | `7.6.25.57` | 项目架构 | rank0 engine 绑定具体 Pod/RANK IP。 |
| node0 | `port` | `8000` | `17000` | 项目架构 | sidecar proxy 对外 `8000`，vLLM backend 使用 `17000`。 |
| node0-node3 | `distributed-executor-backend` | 未显式提供 | `mp` | 必要增强 | 显式锁定原生 MP，避免 Worker 回退默认 Ray。 |

## 核心链路校验

- CLI 合并前 backend：`ray`；rank0 合并后：`mp`。
- Worker payload：`3` 个，分发失败：`0`，backend 均为 `mp`。
- Worker 本机模型路径：`/var/ai-model/Kimi-K3/`。
- rank0 保留 frontend 参数；Worker 均为 headless，且移除 host、port、parser 和 auto-tool 参数。
- 四节点均未出现 `ray start`、`--distributed-executor-backend ray` 或 `--data-parallel-*`。
- 主启动与失败重试分支的 vLLM 命令一致。
- 四份完整 `start_command.sh` 已生成；独立 Linux Bash 语法检查因本机 WSL 未安装发行版而跳过。

## 标准化工具兼容处理

参考命令按用户给定顺序保留。项目标准化工具会把紧跟 `--no-enable-flashinfer-autotune` 的单横杠参数 `-cc.pass_config.fuse_allreduce_rms=False` 暂时解析成前一参数的值；报告在结构化比较对象中将其拆回独立透传参数，并由 `raw_cc_flag` 单独断言。此处理只作用于 dry-run 报告，不修改生产代码或参考命令。

## 回归验证

- 四机 dry-run 集群契约：PASS。
- 四个节点部署策略比较：4/4 PASS。
- 节点契约检查：node0 `27/27`，node1-node3 各 `29/29` PASS。
- 相关自动化测试：`106 passed in 2.59s`。
- 测试原始输出：`../regression_tests.txt`。

## 产物

- `node*/start_command.sh`：生产链路生成的完整脚本。
- `node*/actual_exec_command.txt`：按参数换行的实际命令。
- `node*/standard_exec_command.txt`：按参数换行的参考命令。
- `node*/strict_comparison.*`：不豁免字段的比较报告。
- `node*/deployment_policy_comparison.*`：只豁免上述项目架构差异。
- `node*/contract_checks.json`：节点角色、拓扑、参数、环境和禁用分支校验。
- `dispatch_payloads.json`：Master 实际构造的 Worker 分发参数。
- `comparison.json`：机器可读集群汇总。

## 验证边界

本次为命令生成与分发链路 dry-run，不启动 vLLM、不建立 NCCL 连接、不加载真实权重。H20 信息来自本目录 fixture；模型 `config.json` 元数据仅在 dry-run 进程内模拟，容器模型路径原样进入生产配置合并与命令生成。

为使提交后的完整脚本可直接按容器语义阅读，`node*/start_command.sh` 中仅 dry-run 本地共享目录被机械归一化为运行时路径 `/shared-volume`；四节点 vLLM 执行命令、环境变量、拓扑和比较结果未改动。
