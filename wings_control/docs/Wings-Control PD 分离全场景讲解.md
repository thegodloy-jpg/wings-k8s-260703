# Wings-Control PD 分离全场景讲解

> 面向页面、编排平台和 Wings-Control 维护人员。本文只解释当前 checkout 的有效路径，目标是：平台给出明确拓扑后，Wings-Control 为每个 pod 生成角色、并行度、rank、卡组、端口和 KV 配置都正确的 `start_command.sh`。

---

## 1. 方案目标和场景边界

### 1.1 最终交付物是逐节点启动命令

```text
页面固定输入 + 编排动态输入 + 模型 config.json
                    │
                    ▼
         Wings-Control 配置合并与 PD 规划
                    │
                    ▼
           当前 pod 的 start_command.sh
                    │
                    ▼
      正确的角色、TP/DP、rank、卡组、端口、KV 配置
```

验收必须检查最终脚本，不能只检查页面保存值、`engine_config` 中间态或 `pd_config.json` 某一行。

### 1.2 当前保留的场景

| 场景 | 触发条件 | 最终命令形态 |
| --- | --- | --- |
| 非 PD | 无有效 `PD_ROLE` | 普通单机或普通分布式命令 |
| Ascend 单 service PD | Ascend + `PD_ROLE=P/D` + 本角色 DP=1 | 单个 vLLM service，不带 external-lb DP flags |
| Ascend 多 PD / 大 EP | Ascend + `PD_ROLE=P/D` + 本角色 DP>1 | 每 pod fork `DP_SIZE_LOCAL` 个 vLLM service |
| NVIDIA 单节点 PD | NVIDIA + `PD_ROLE=P/D` | `NixlConnector` 单进程 |

### 1.3 触发真相源

PD 总开关只认大写角色：

```bash
PD_ROLE=P  # Prefill
PD_ROLE=D  # Decode
```

未设置、空字符串或其它值都表示未开启 PD。

设备类型从 `WINGS_HARDWARE_FILE` 指向的 `hardware_info.json` 读取；`WINGS_DEVICE=ascend` 不是生产真相源。Ascend 文件至少应包含：

```json
{"device":"ascend","count":8,"details":[{"name":"Ascend 910B"}]}
```

### 1.4 四层拓扑对象

| 层级 | 权威输入或派生值 | 负责回答的问题 |
| --- | --- | --- |
| P/D 全局层 | `PD_PREFILL_*`、`PD_DECODE_*` | P、D 各有多少 service，每个 service 用多少卡 |
| 角色域 | `PD_ROLE`、`Master_IP`、`NODE_IPS` | 当前 pod 属于哪个 DP 域，coordinator 在哪里 |
| PD pod 层 | `DP_SIZE_LOCAL`、`PD_DP_RANK_START`、`PD_INDEX` | 当前 pod 的 PD 身份、fork 数和起始 rank |
| 本地 service 层 | `i`、rank、卡组、API/bootstrap 端口 | pod 内每条最终命令具体使用什么 |

页面和编排提供 PD pod 层及以上输入；本地 service 层由 Wings-Control 确定性生成。

---

## 2. 外部环境变量契约

### 2.1 页面与编排的责任边界

- 页面固定字段：模型、引擎、角色、P/D 全局 TP/DP、当前 pod 本地 service 数、Ascend 平台和对外端口规划；
- 编排动态字段：硬件文件、pod IP、当前角色 coordinator、角色节点列表、通信网卡、当前 pod 的 rank/PD 起点；
- Wings-Control 生成字段：每个 service 的 rank、卡组、API 端口、bootstrap 端口、`engine_id` 和 KV port。

代码默认值主要用于兼容和测试。生产页面不能依赖默认值，否则不同 pod 可能生成重复命令。

### 2.2 所有场景公共必传

页面请求必须传：

| 环境变量 | 约束 |
| --- | --- |
| `MODEL_NAME` | 非空，和实际模型一致 |
| `MODEL_PATH` | 容器内真实权重路径 |
| `ENGINE` | Ascend=`vllm_ascend`；NVIDIA=`vllm` |
| `DEVICE_COUNT` | 与当前 pod 可见卡数一致 |
| `PORT` | Wings proxy 对外端口 |
| `ENGINE_PORT` | 当前 pod 的引擎基础端口 |
| `HEALTH_PORT` | 与 K8s 探针一致 |

编排必须注入：

| 环境变量 | 约束 |
| --- | --- |
| `WINGS_HARDWARE_FILE` | 指向已挂载的 `hardware_info.json` |
| `POD_IP` | 当前 pod 地址 |
| `RANK_IP` | 当前 pod 的 rank 身份地址，通常等于 `POD_IP` |
| `NETWORK_INTERFACE` | 实际 NPU/NCCL 通信网卡 |

### 2.3 分场景页面最小契约

#### 场景 A：非 PD

只需传 2.2 的公共变量，并保证不传 `PD_ROLE`。残留的其它 `PD_*` 不应被当作有效拓扑。

#### 场景 B：Ascend 单 service PD / 1P1D

页面除公共变量外必须传：

```bash
ENGINE=vllm_ascend
PD_ROLE=<P或D>
PD_PREFILL_TP_SIZE=<P单service TP>
PD_PREFILL_DP_SIZE=1
PD_DECODE_TP_SIZE=<D单service TP>
PD_DECODE_DP_SIZE=1
DP_SIZE_LOCAL=1
PD_INDEX=<当前pod的全局PD编号>
WINGS_ASCEND_PLATFORM=<a2或a3>
```

编排按 2.2 注入硬件、IP 和网卡即可。`Master_IP/NODE_IPS` 在 DP=1 最终命令中不会形成 DP flags，不是硬必传；若统一下发，必须使用当前角色自己的地址域。

推荐 1P1D 编号：P=`PD_INDEX=0`，D=`PD_INDEX=1`。

#### 场景 C：Ascend 多 PD / 大 EP

该场景必须按层下发，不能只传当前角色的 TP/DP。

**P/D 全局层：所有 P、D pod 完全一致**

```bash
ENGINE=vllm_ascend
PD_PREFILL_TP_SIZE=<P单service TP>
PD_PREFILL_DP_SIZE=<P全局service数>
PD_DECODE_TP_SIZE=<D单service TP>
PD_DECODE_DP_SIZE=<D全局service数>
WINGS_ASCEND_PLATFORM=<a2或a3>
```

**角色域：同角色 pod 一致，P/D 分开**

```bash
PD_ROLE=<P或D>
Master_IP=<当前角色rank0地址>
NODE_IPS=<当前角色全部pod地址，按rank顺序>
```

**PD pod 层：每个 pod 单独注入**

```bash
DP_SIZE_LOCAL=<当前pod fork数>
PD_INDEX=<上层为当前PD pod按P先D后顺序直接下发的编号>
POD_IP=<当前pod地址>
RANK_IP=<当前pod地址，必须存在于NODE_IPS>
NETWORK_INTERFACE=<NPU通信网卡>
```

非均匀 pod 还必须传：

```bash
PD_DP_RANK_START=<当前pod在本角色DP域的起始rank>
```

均匀 pod 可使用自动公式：

```text
dp_rank_start = NODE_IPS.index(RANK_IP) × DP_SIZE_LOCAL
```

`PD_INDEX` 由上层直接分配和下发。Wings-Control 不根据 `rank_start`、P/D DP 数或本地序号推导它。

页面/后端创建工作负载前必须校验：

```text
sum(同角色全部 pod 的 DP_SIZE_LOCAL) = 当前角色全局 DP_SIZE
每个 pod：TP_SIZE × DP_SIZE_LOCAL ≤ DEVICE_COUNT
P/D 四个全局拓扑变量在所有 pod 一致
P、D 分别使用自己的 Master_IP / NODE_IPS
RANK_IP = POD_IP，且 RANK_IP 存在于 NODE_IPS
同角色所有 pod 的 rank 区间不重叠
按 P pod 在前、D pod 在后的顺序检查 PD_INDEX 为 0,1,...,N-1
全局 PD_INDEX 无跳号、无重复，P/D 不复用编号
ENGINE_PORT+i、bootstrap+i 不冲突；不同 PD pod 的 KV port 不冲突
external LB 覆盖所有 ENGINE_PORT+i
```

#### 场景 D：NVIDIA 单节点 PD

页面除公共变量外必须传：

```bash
ENGINE=vllm
PD_ROLE=<P或D>
PD_PREFILL_TP_SIZE=<P TP>
PD_PREFILL_DP_SIZE=1
PD_DECODE_TP_SIZE=<D TP>
PD_DECODE_DP_SIZE=1
```

该路径使用 `NixlConnector`，不使用 `PD_INDEX`。

#### 场景 E：PD 与页面 Smart 三特性

任一有效 `PD_ROLE` 下，页面必须禁止或清零：

```bash
ENABLE_KV_OFFLOAD=false
LMCACHE_OFFLOAD=false
ENABLE_SPECULATIVE_DECODE=false
ENABLE_SPARSE=false
```

后端会再次执行 PD veto，但页面应提前阻止，避免展示状态与最终命令不一致。

### 2.4 规范字段与兼容优先级

页面只使用：

```text
PD_PREFILL_TP_SIZE
PD_PREFILL_DP_SIZE
PD_DECODE_TP_SIZE
PD_DECODE_DP_SIZE
DP_SIZE_LOCAL
PD_INDEX
PD_DP_RANK_START（条件必传）
```

不要同时下发 `TP_SIZE`、`DP_SIZE`、`PD_TP_SIZE`、`PD_DP_SIZE`、`PD_DP_SIZE_LOCAL`。当前代码兼容优先级是：

```text
TP_SIZE > PD_TP_SIZE > PD_{PREFILL|DECODE}_TP_SIZE > 默认 1
DP_SIZE > PD_DP_SIZE > PD_{PREFILL|DECODE}_DP_SIZE > 默认 1
DP_SIZE_LOCAL > PD_DP_SIZE_LOCAL > 默认 1
```

### 2.5 Ascend 平台和可选控制

| 环境变量 | 用途 |
| --- | --- |
| `WINGS_CONFIG_DIR` | 可选重定向 defaults 目录，会改变实际加载的 `pd_config.json` |
| `WINGS_ASCEND_PLATFORM=a2/a3` | 页面首选的平台信号 |
| `ASCEND_PLATFORM` / `ENGINE_IMAGE_FLAVOR` | 平台兼容信号 |
| `ENGINE_VERSION=...-a3` / `ASCEND_A3_ENABLE=true` | 没有显式平台信号时推导 A3 |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | bootstrap 基址；P 默认 23000，D 默认 23100 |
| `PD_DISABLE_ASCEND_DIRECT=true` | 从 KV extra 删除 `use_ascend_direct` 的部署级规避开关 |
| `ASCEND_ENFORCE_EAGER=true` | A+X/Triton 冲突场景强制 eager |

平台解析优先级是显式平台信号 → `ENGINE_VERSION/ASCEND_A3_ENABLE` → 注册表条目默认或 base。生产部署应显式传 `WINGS_ASCEND_PLATFORM`。

### 2.6 页面不应计算的值

| 值 | 当前生成规则 |
| --- | --- |
| service DP rank | `dp_rank_start+i` |
| service API port | `ENGINE_PORT+i` |
| service bootstrap port | `bootstrap_base+i` |
| service 卡组 | `[i×TP_SIZE, (i+1)×TP_SIZE)` |
| DP RPC port | P=`12890`；D=`12777` |
| `engine_id` / KV port | 只使用上层传入的 `PD_INDEX`，见第 3、6 章 |

`VLLM_LLMDD_RPC_PORT`、`PD_DP_RPC_PORT` 不控制 Ascend external-lb 的 DP RPC 端口。Ascend 多 PD 也不需要 `DISTRIBUTED=true`。

---

## 3. PD_INDEX：精简定义与使用

`PD_INDEX` 是上层直接传给当前 PD pod 的全局顺序号。对 Wings-Control 来说，它是 pod 级权威输入，只依赖这个值，不从 `rank_start`、P/D DP 数或本地序号 `i` 计算。

上层分配必须遵守与官方一致的顺序：

```text
从 0 开始
先编号全部 P pod
再编号全部 D pod
全局连续、无跳号、P/D 不重复
```

Wings-Control 直接使用：

```text
engine_id = PD_INDEX
kv_port = 30000 + PD_INDEX × 100
```

例如 4P1D（4 个 P pod、1 个 D pod）：

| pod | P0 | P1 | P2 | P3 | D0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `PD_INDEX` | 0 | 1 | 2 | 3 | 4 |

D0 必须是 4，不能从 0 重新编号；`23,24,36,37` 这类“唯一但不连续”的 ID 也不符合契约。

pod 内所有本地 service 复用该 pod 的同一个 `PD_INDEX`。Wings-Control 不关心该值和 DP rank、P/D DP 数或本地 `i` 的关系，也不使用它决定 API 端口、bootstrap 端口或 NPU 卡组。顺序和唯一性由上层保证。

---

## 4. 从外部输入到最终命令

### 4.1 主链路

```text
parse_launch_args()
  └─ 读取页面公共环境变量/CLI

load_and_merge_configs()
  ├─ detect_hardware()：读取 hardware_info.json
  ├─ _set_pd_parallelism_params()：PD TP/DP 防火墙
  ├─ _get_pd_config()：生成早期 standalone KV 中间态
  ├─ apply_effective_feature_enablement()：执行 PD veto
  ├─ _apply_pd_external_lb()：注册表、平台和角色 recipe
  └─ _apply_pd_final_guard()：重申最终 TP/DP

build_start_script()
  ├─ _prepare_engine_config()：重申注册表 engine 覆盖
  ├─ _build_pd_external_lb_env_cmds()：构建隔离环境
  └─ _build_vllm_pd_external_lb_script()：展开当前 pod 命令
```

### 4.2 Ascend PD 命中条件

`PD_ROLE` 有效且设备或引擎识别为 Ascend 后，DP=1 和 DP>1 都进入 `pd_config.json` 注册表路径：

```text
模型 config.json architecture
  → 专属 architecture 条目
  → 找不到时回退 default
  → platform_overrides[a2/a3]
  → common + prefill/decode
```

注册表负责 connector、模型 engine 参数、环境变量和 extra config；外部拓扑负责 TP/DP、rank、地址和编号，两者不能互相代替。

### 4.3 最终覆盖和隔离环境

注册表 engine 配置会保存到 `_pd_engine_overrides`，并在模型默认注入器之后重申；`None` 表示删除已有键。`_apply_pd_final_guard()` 最后再次按 PD 环境变量校正 TP/DP。

Ascend PD 脚本使用隔离环境，基础项是：

```bash
export HCCL_IF_IP=<POD_IP，缺失时RANK_IP>
export GLOO_SOCKET_IFNAME=<NETWORK_INTERFACE>
export TP_SOCKET_IFNAME=<NETWORK_INTERFACE>
export HCCL_SOCKET_IFNAME=<NETWORK_INTERFACE>
export PD_INDEX=<上层传入的当前PD pod编号>
```

然后叠加注册表 `common_env + 角色 env`，再按 `strip_env` 删除不适用变量。Mooncake 连接器的 `/usr/local/lib` 仅作为引擎进程前缀，不污染全局环境。

---

## 5. Ascend 单 service PD

### 5.1 1P1D 输入示例

P pod：

```bash
ENGINE=vllm_ascend
PD_ROLE=P
PD_INDEX=0
PD_PREFILL_TP_SIZE=8
PD_PREFILL_DP_SIZE=1
PD_DECODE_TP_SIZE=8
PD_DECODE_DP_SIZE=1
DP_SIZE_LOCAL=1
POD_IP=10.0.0.10
RANK_IP=10.0.0.10
NETWORK_INTERFACE=ens65f1np1
WINGS_ASCEND_PLATFORM=a2
```

D pod 仅改变角色、编号和本机地址：

```bash
PD_ROLE=D
PD_INDEX=1
POD_IP=10.0.0.11
RANK_IP=10.0.0.11
```

### 5.2 最终命令要求

DP=1 仍读取注册表，但生成一个前台命令：

```bash
LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-} \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_MOONCAKE_BOOTSTRAP_PORT=<P=23000或D=23100> \
python3 -m vllm.entrypoints.openai.api_server \
  ... \
  --kv-transfer-config '<注册表生成的JSON>' \
  --port <ENGINE_PORT> \
  --tensor-parallel-size 8
```

必须满足：

- P=`kv_producer`，D=`kv_consumer`；
- KV extra 中 P/D TP/DP 完整且两侧一致；
- `engine_id` 和 KV port 使用当前 pod 的 `PD_INDEX`；
- 不出现 `--data-parallel-external-lb`；
- `strip_env` 和连接器可以决定是否保留 bootstrap/linker 前缀。

---

## 6. Ascend 多 PD / 大 EP：逐 pod、逐 service 方案

### 6.1 目标和不变量

一个角色可以由多个 PD pod 组成，每个 pod 又可以 fork 若干独立 vLLM API service。必须分清三个计数：

```text
PD pod 数          = 编排创建的 P/D pod 数，用于上层分配 PD_INDEX
DP_SIZE            = 当前角色全局 service / DP rank 数
DP_SIZE_LOCAL      = 当前 pod fork 的本地 service 数
每 pod NPU 需求    = TP_SIZE × DP_SIZE_LOCAL
```

`PD_INDEX` 只标识 PD pod。一个 pod 内无论 fork 多少个本地 service，都复用该 pod 的同一个 `PD_INDEX`。

必须满足：

```text
sum(同角色 DP_SIZE_LOCAL) = 当前角色 DP_SIZE
同角色 rank 区间连续且不重叠
所有 PD pod 的 PD_INDEX 由上层按 P→D 从 0 连续分配
同一 pod 内所有本地 service 使用相同 PD_INDEX
同角色所有 pod 指向同一个 Master_IP 和固定 RPC port
P/D 两侧都携带完整的 P/D TP/DP 拓扑
```

当前代码会把 `DP_SIZE_LOCAL` 截断到不超过全局 DP，但不会完整校验 `TP_SIZE × DP_SIZE_LOCAL ≤ DEVICE_COUNT`；资源等式必须由平台先校验。

### 6.2 示例 A：2P8D，每个 pod 一个本地 service

这里的 2P8D 表示 2 个 P pod 和 8 个 D pod：

| 角色 | 全局 DP | 每 service TP | pod 数 | 每 pod 本地 DP |
| --- | ---: | ---: | ---: | ---: |
| P | 2 | 4 | 2 | 1 |
| D | 8 | 1 | 8 | 1 |

所有 pod 共用：

```bash
PD_PREFILL_DP_SIZE=2
PD_PREFILL_TP_SIZE=4
PD_DECODE_DP_SIZE=8
PD_DECODE_TP_SIZE=1
```

角色地址域：

| 角色 | `Master_IP` | `NODE_IPS` |
| --- | --- | --- |
| P | `10.0.0.10` | `10.0.0.10,10.0.0.11` |
| D | `10.0.1.20` | `10.0.1.20,10.0.1.21,10.0.1.22,10.0.1.23,10.0.1.24,10.0.1.25,10.0.1.26,10.0.1.27` |

每个 pod 的差异化输入：

| pod | `PD_ROLE` | `POD_IP/RANK_IP` | `DEVICE_COUNT` | `DP_SIZE_LOCAL` | `PD_DP_RANK_START` | `PD_INDEX` |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| P0 | P | `10.0.0.10` | 4 | 1 | 0 | 0 |
| P1 | P | `10.0.0.11` | 4 | 1 | 1 | 1 |
| D0 | D | `10.0.1.20` | 1 | 1 | 0 | 2 |
| D1 | D | `10.0.1.21` | 1 | 1 | 1 | 3 |
| D2 | D | `10.0.1.22` | 1 | 1 | 2 | 4 |
| D3 | D | `10.0.1.23` | 1 | 1 | 3 | 5 |
| D4 | D | `10.0.1.24` | 1 | 1 | 4 | 6 |
| D5 | D | `10.0.1.25` | 1 | 1 | 5 | 7 |
| D6 | D | `10.0.1.26` | 1 | 1 | 6 | 8 |
| D7 | D | `10.0.1.27` | 1 | 1 | 7 | 9 |

`PD_INDEX=0～9` 是上层按 pod 顺序直接分配的结果。Wings-Control 不根据表中的 rank 或 DP 数计算这些值。

若 `RANK_IP` 不在多节点 `NODE_IPS` 中，当前代码只记录 error 并回退 rank 0，不会 fail-fast。平台必须阻止该工作负载创建，不能依赖回退。

### 6.3 示例 B：2P1D，D pod 本地 fork 8 个 service

全局 DP/TP 仍然是 P=`DP2/TP4`、D=`DP8/TP1`，但编排只创建 P0、P1、D0 三个 PD pod：

| pod | `PD_ROLE` | `DP_SIZE_LOCAL` | `PD_DP_RANK_START` | 上层 `PD_INDEX` |
| --- | --- | ---: | ---: | ---: |
| P0 | P | 1 | 0 | 0 |
| P1 | P | 1 | 1 | 1 |
| D0 | D | 8 | 0 | 2 |

假设 D0：`ENGINE_PORT=18000`、bootstrap base=`23100`、`TP_SIZE=1`、`PD_INDEX=2`。pod 内八个本地 service 的目标命令展开为：

| `i` | DP rank | pod `PD_INDEX` | `engine_id` | API port | bootstrap | NPU 卡 | KV port |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 2 | 2 | 18000 | 23100 | 0 | 30200 |
| 1 | 1 | 2 | 2 | 18001 | 23101 | 1 | 30200 |
| 2 | 2 | 2 | 2 | 18002 | 23102 | 2 | 30200 |
| 3 | 3 | 2 | 2 | 18003 | 23103 | 3 | 30200 |
| 4 | 4 | 2 | 2 | 18004 | 23104 | 4 | 30200 |
| 5 | 5 | 2 | 2 | 18005 | 23105 | 5 | 30200 |
| 6 | 6 | 2 | 2 | 18006 | 23106 | 6 | 30200 |
| 7 | 7 | 2 | 2 | 18007 | 23107 | 7 | 30200 |

八条命令的 rank、API 端口、bootstrap 和卡组随 `i` 变化；`PD_INDEX`、`engine_id` 和 KV port 都保持为 D0 pod 的值，本地 service 不产生新编号。

### 6.4 配置链路的七个细粒度步骤

| 步骤 | 代码职责 | 关键结果 |
| --- | --- | --- |
| 1. 并行度防火墙 | `_set_pd_parallelism_params()` | 按 `PD_ROLE` 选择 P 或 D 的 TP/DP，阻止 `device_count` 回填成普通 TP |
| 2. pod 规划 | `_get_pd_external_lb_params()` | 形成 role、TP、全局 DP、本地 DP、rank 起点、地址、RPC port，并读取外部 PD_INDEX |
| 3. 模型注册表 | `_apply_pd_external_lb()` | architecture → default → platform overlay → common/角色 recipe |
| 4. KV 配置 | `_build_pd_external_lb_kv()` | 生成 connector、P/D 全局拓扑、`__PD_INDEX__`、`__PD_KVPORT__` |
| 5. engine 重申 | `_prepare_engine_config()` + final guard | 防止模型默认覆盖 PD recipe 和 TP/DP |
| 6. 隔离环境 | `_build_pd_external_lb_env_cmds()` | 只保留通信环境与注册表声明环境 |
| 7. service 展开 | `_build_vllm_pd_external_lb_script()` | 按 `i` 生成 rank、端口、卡组；所有本地 service 复用 pod 的 PD_INDEX |

内部规划对象的核心字段是：

```python
{
  "role": "P" or "D",
  "tp_size": 当前角色每service TP,
  "dp_size": 当前角色全局DP,
  "dp_size_local": 当前pod service数,
  "dp_rank_start": 当前pod起始rank,
  "dp_address": 当前角色Master_IP,
  "rpc_port": "12890" or "12777",
  "pd_index_base": 外部传入的pod级PD_INDEX（当前内部字段名）
}
```

KV extra 的 `prefill/decode` 必须使用全局 TP/DP，而不是当前 pod 的 `DP_SIZE_LOCAL`。

### 6.5 目标 fork 脚本

命令生成器展开 rank、端口和卡组时使用 `i`；`PD_INDEX` 和 KV port 是 pod 级值，在循环内保持不变：

```bash
(
  pids=()
  KVPORT=$((30000 + PD_INDEX * 100))

  for i in $(seq 0 $((DP_SIZE_LOCAL - 1))); do
    RANK=$((DP_RANK_START + i))
    PORT=$((ENGINE_PORT + i))
    BOOTSTRAP=$((BOOTSTRAP_BASE + i))
    LO=$((i * TP_SIZE))
    HI=$((LO + TP_SIZE - 1))
    CARDS=$(seq -s, $LO $HI)

    # 所有本地 service 直接复用当前 pod 的 PD_INDEX/KVPORT
    ASCEND_RT_VISIBLE_DEVICES=$CARDS \
      VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOTSTRAP \
      python3 -m vllm.entrypoints.openai.api_server \
      ... \
      --kv-transfer-config "$KV_CONFIG_USING_PD_INDEX" \
      --port $PORT \
      --tensor-parallel-size $TP_SIZE \
      --data-parallel-size $DP_SIZE \
      --data-parallel-rank $RANK \
      --data-parallel-size-local 1 \
      --data-parallel-address $Master_IP \
      --data-parallel-rpc-port <P=12890或D=12777> \
      --data-parallel-external-lb &
    pids+=($!)
  done

  wait -n || true
  kill "${pids[@]}" 2>/dev/null || true
  exit 1
)
```

每个 fork 进程的 `--data-parallel-size-local` 固定为 1；`DP_SIZE_LOCAL` 只控制本 pod fork 次数，不改变 `PD_INDEX`。上层只需给每个 PD pod 下发一个编号。

### 6.6 当前代码与目标对齐状态

| 能力 | 状态 |
| --- | --- |
| P/D TP/DP 读取和最终守卫 | 已对齐 |
| architecture、平台 overlay、P/D recipe | 已对齐 |
| rank、API 端口、bootstrap、卡组按 `i` 派生 | 已对齐 |
| 每个进程 `data-parallel-size-local=1` | 已对齐 |
| 任一 service 退出后整 pod 退出 | 已对齐 |
| `engine_id/KVPORT` 只使用外部 `PD_INDEX` | 已对齐：当前直接读取和使用该值 |
| 同一 pod 的本地 service 共享该 pod 的 `PD_INDEX` | 已对齐：当前 fork 循环保持该值不变 |
| 注册表 `kv_port` 控制最终端口 | **未对齐：当前读取为 `kv_port_base` 后未消费** |

当前循环实际是：

```bash
RANK=$((dp_rank_start + i))
PD_INDEX=$PD_INDEX
KVPORT=$((30000 + PD_INDEX * 100))
```

这段代码符合 pod 级契约：rank 随 `i` 递增，`PD_INDEX` 不变。大 EP 验收只检查每个 PD pod 是否拿到上层分配的正确编号。

### 6.7 每个 pod 的最终命令验收

| 检查维度 | 必须满足 |
| --- | --- |
| 角色 | P=`kv_producer`；D=`kv_consumer` |
| 全局拓扑 | KV extra 的 P/D TP/DP 在所有 pod 一致 |
| 本地数量 | fork 次数等于 `DP_SIZE_LOCAL` |
| DP rank | 从 `dp_rank_start` 连续递增，同角色不重复 |
| PD 编号 | 每个 pod 直接使用上层 `PD_INDEX`；同 pod 本地 service 共享；所有 pod 按 P→D 为 `0,1,...,N-1` |
| 卡组 | 每个 service 独占 `TP_SIZE` 张卡，不重叠、不越界 |
| 端口 | API/bootstrap 按 service 唯一；KV port 由 pod 的 `PD_INDEX` 决定 |
| DP 地址 | 同角色所有 pod 指向同一个 `Master_IP` |
| RPC 端口 | P=12890；D=12777 |
| 失败语义 | 任一 service 退出后整 pod 退出 |

### 6.8 external LB、Proxy、健康和失败语义

每个 fork service 都是独立 API server，监听 `ENGINE_PORT+i`。平台 external LB 必须覆盖全部端口。

Wings-Control 自带 proxy 和 health 当前只指向基础 `ENGINE_PORT`：

- 基础 health 成功不代表全部本地 service 可接流量；
- external LB 只暴露基础端口时，其它 service 不会承接请求；
- 任一 service 退出会触发 `wait -n`，终止其余 service，并让编排层重启整个 pod；
- 跨 pod DP rendezvous 使用 `--data-parallel-address` 和固定 RPC 端口，不使用 Ray master/worker。

---

## 7. 大 EP recipe 的模型复用

### 7.1 注册表结构

新增模型优先修改 `config/defaults/pd_config.json`，key 使用模型 `config.json` 的 architecture：

```json
{
  "NewArchitecture": {
    "connector": "MooncakeConnectorV1",
    "kv_port": {"P":"30000","D":"30100"},
    "extra_config": {},
    "common_env": {},
    "common": {},
    "prefill": {"engine":{},"env":{},"extra_config":{},"strip_env":[]},
    "decode": {"engine":{},"env":{},"extra_config":{},"strip_env":[]},
    "platform_overrides": {"a2":{},"a3":{}}
  }
}
```

合并顺序：base → platform overlay → common → P/D 角色项 → 外部 TP/DP 强制覆盖。

### 7.2 当前主要 architecture

| architecture | connector | 特点 |
| --- | --- | --- |
| `default` | `MooncakeConnectorV1` | 未注册模型兜底 |
| `Qwen3MoeForCausalLM` | `MooncakeLayerwiseConnector` | NPU buffer、P/D 独立参数 |
| `DeepseekV32ForCausalLM` | `MooncakeLayerwiseConnector` | DeepSeek V3.2 recipe |
| `GlmMoeDsaForCausalLM` | `MooncakeConnector` | module path、A2 overlay、`strip_env` |
| `Qwen3_5MoeForConditionalGeneration` | `MooncakeLayerwiseConnector` | D 角色 extra config |
| `DeepseekV4ForCausalLM` | `MooncakeHybridConnector` | 平台 overlay、HMA、`strip_env` |

### 7.3 新模型接入检查表

1. 从真实 `config.json` 确认 architecture；
2. 从目标 vLLM-Ascend 版本确认 connector 和 module path；
3. 分离 P/D 的 engine、env、extra config；
4. A2/A3 差异放 `platform_overrides`；
5. 删除基础键使用 `null`，删除环境变量使用 `strip_env`；
6. 检查两侧 KV extra 中完整的 P/D TP/DP；
7. 分别验证 DP=1 和 DP>1 的最终脚本；
8. 对 `DP_SIZE_LOCAL>1` 验证每个 service 的 rank、卡组、端口、`engine_id`；
9. 做单 rank 退出和多轮请求测试。

不要按模型名复制 Python 分支，不要用 `PD_CONNECTOR_TYPE` 绕过注册表，也不要只测 `engine_config`。

---

## 8. NVIDIA 单节点和 PD 特性边界

### 8.1 NVIDIA 单节点 PD

NVIDIA 不进入 Ascend `pd_config.json` 路径，基础 KV 配置是：

```json
{"kv_connector":"NixlConnector","kv_role":"kv_both"}
```

它使用单进程命令，不消费 Ascend 的 `PD_INDEX`、`Master_IP/NODE_IPS` 或 external-lb fork 逻辑。

### 8.2 Smart 三特性 veto

任一有效 `PD_ROLE` 会关闭页面请求的：

- speculative decode；
- sparse KV；
- KV offload / LMCache。

因此当前标准链路不会生成“PD + LMCache MultiConnector”。注册表 recipe 自带的 `speculative_config` 是模型参数，不等于页面 SmartFeature 投机请求。

### 8.3 其它保护

| 机制 | 作用 |
| --- | --- |
| `_guard_pd_hybrid_kv_cache` | 移除与普通 PD connector 冲突的 HMA；Hybrid recipe 可重新注入 |
| `_ensure_pd_head_dim` | 模型配置缺 `head_dim` 时补 `hf_overrides` |
| `_filter_pd_incompatible_env` | 删除 PD 下不兼容的 balance scheduling 环境变量 |
| `_apply_pd_final_guard` | 防止普通模型默认污染 PD TP/DP |

SmartQoS、RAG、reasoning parser、tool-call parser 有独立路径，不能因为 PD 开启就默认全部关闭。

---

## 9. 验证、排障和代码索引

### 9.1 聚焦测试

```bash
python -m pytest tests/test_pd_deepseek_v4_env.py -q
```

现有测试覆盖 Ascend P/D 全局拓扑、角色环境、隔离环境、Mooncake linker 前缀和 NVIDIA 不误入 Ascend 注册表。Ascend 多 PD 还应补充：

1. DP=1：单前台命令，无 external-lb flags；
2. DP>1：fork 次数、rank、端口、卡组和 `wait -n`；
3. 多 pod：`NODE_IPS/RANK_IP` 推导 rank 起点；
4. 4P1D：P=`0,1,2,3`、D=`4`，拒绝跳号和 P/D 重号；
5. 每个 PD pod 直接使用上层传入的 `PD_INDEX`，同 pod 本地 service 复用该值。

### 9.2 启动日志

```text
[PD external-lb trigger]
[vllm_adapter.env_path] selected=pd_external_lb_isolated
[PD external-lb env] isolated_builder=True
[PD external-lb env merge]
[wings-cmd] >>> python3 -m vllm...
```

只有 `[PD Config]` 而没有 `[PD external-lb trigger]` 时，检查硬件文件、最终 engine、`PD_ROLE`、容器代码版本和 `pd_config.json` 打包结果。

### 9.3 常见问题

| 现象 | 优先检查 |
| --- | --- |
| TP 意外等于全部卡数 | `PD_*_TP_SIZE` 是否缺失或被 `TP_SIZE` 覆盖 |
| 多 pod rank 撞车 | `RANK_IP`、`NODE_IPS`、`PD_DP_RANK_START` |
| rank0 bind 其它地址失败 | `Master_IP` 是否为当前角色 coordinator |
| 只有基础端口有流量 | external LB 是否覆盖 `ENGINE_PORT+i` |
| Mooncake 找不到 so | 最终进程是否带 `/usr/local/lib` linker 前缀 |
| 任一 service 退出后整 pod 重启 | 当前 EP fail-together 设计 |
| offload 请求不生效 | PD veto，属于当前预期 |
| PD_INDEX 唯一但出现跳号 | 平台分配错误；必须从 0 按 P→D 连续编号 |
| D 从 0 重新编号 | 平台分配错误；D 编号必须接在全部 P pod 之后 |
| 同 pod 多个 service 共用 engine_id/KV port | pod 级 `PD_INDEX` 的预期行为，不应按本地 `i` 改写 |
| 注册表改 kv_port 但命令未变化 | 当前脚本未消费 `kv_port_base` |

### 9.4 代码索引

| 环节 | 函数/文件 | 当前位置 |
| --- | --- | --- |
| 页面/环境参数解析 | `build_parser` | [start_args_compat.py:261](../core/start_args_compat.py#L261) |
| 页面公共必传校验 | `wings_start.sh` | [wings_start.sh:261](../wings_start.sh#L261) |
| 端口规划 | `derive_port_plan` | [port_plan.py:35](../core/port_plan.py#L35) |
| 硬件文件加载 | `detect_hardware` | [hardware_detect.py:144](../core/hardware_detect.py#L144) |
| PD 总开关 | `get_pd_role_env` | [env_utils.py:299](../utils/env_utils.py#L299) |
| PD TP/DP 防火墙 | `_set_pd_parallelism_params` | [config_loader.py:894](../core/config_loader.py#L894) |
| 早期 KV 中间态 | `_get_pd_config` | [config_loader.py:1124](../core/config_loader.py#L1124) |
| Ascend pod 规划 | `_get_pd_external_lb_params` | [config_loader.py:1213](../core/config_loader.py#L1213) |
| Ascend KV 构建 | `_build_pd_external_lb_kv` | [config_loader.py:1356](../core/config_loader.py#L1356) |
| 平台解析 | `_resolve_ascend_platform` | [config_loader.py:1435](../core/config_loader.py#L1435) |
| 注册表应用 | `_apply_pd_external_lb` | [config_loader.py:1522](../core/config_loader.py#L1522) |
| KV 初始决策 | `_set_kv_cache_config` | [config_loader.py:1743](../core/config_loader.py#L1743) |
| SmartFeature PD veto | `_apply_smart_feature_pd_veto` | [config_loader.py:2963](../core/config_loader.py#L2963) |
| PD 最终拓扑守卫 | `_apply_pd_final_guard` | [config_loader.py:4399](../core/config_loader.py#L4399) |
| 最终注册表调用点 | `load_and_merge_configs` | [config_loader.py:4663](../core/config_loader.py#L4663) |
| 最终 engine 重申 | `_prepare_engine_config` | [vllm_adapter.py:2677](../engines/vllm_adapter.py#L2677) |
| Ascend 隔离环境 | `_build_pd_external_lb_env_cmds` | [vllm_adapter.py:4719](../engines/vllm_adapter.py#L4719) |
| Ascend 脚本生成 | `_build_vllm_pd_external_lb_script` | [vllm_adapter.py:4779](../engines/vllm_adapter.py#L4779) |
| 脚本分派 | `build_start_script` | [vllm_adapter.py:4918](../engines/vllm_adapter.py#L4918) |
| 启动器角色 | `_determine_role` | [wings_control.py:603](../wings_control.py#L603) |
| 模型 recipe | `pd_config.json` | [pd_config.json](../config/defaults/pd_config.json) |
| 聚焦测试 | `test_pd_deepseek_v4_env.py` | [test_pd_deepseek_v4_env.py](../../tests/test_pd_deepseek_v4_env.py) |

---

## 10. 维护原则

1. 以每个 pod 的最终 `start_command.sh` 为验收对象；
2. 页面字段、pod 规划和最终命令必须表达同一个拓扑；
3. 平台负责固定输入和动态拓扑，注册表负责模型 recipe，脚本生成器负责 service 展开；
4. 新模型优先扩展注册表，不按模型名复制 Python 分支；
5. 每条目标公式必须有最终脚本断言；
6. `PD_INDEX` 只作为外部权威输入使用，不从 rank、P/D DP 数或本地 `i` 派生。
