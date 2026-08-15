# 加速特性回显接口

## 1. 接口定义

该接口用于查询模型实例最终生效的加速特性，页面应以接口返回值回显特性开关和智能推荐容量。

| 项目 | 内容 |
| --- | --- |
| 方法 | `GET` |
| 路径 | `/v1/startup/accel` |
| 默认端口 | `19000` |
| 请求参数 | 无 |
| 鉴权 | 当前未配置接口级鉴权 |

```bash
curl http://127.0.0.1:19000/v1/startup/accel
```

Kubernetes Pod 内调用：

```bash
kubectl exec -n <namespace> <pod-name> -- \
  curl -s http://127.0.0.1:19000/v1/startup/accel
```

## 2. 响应样例

```json
{
  "code": 200,
  "msg": "",
  "data": {
    "engine": "vllm",
    "features": {
      "speculative_decode": true,
      "sparse_kv": true,
      "kv_offload": true,
      "rag_acc": false
    },
    "variants": {
      "speculative_decode": "mtp",
      "sparse_kv": "fp8_indexcache_use_index_cache_topk4",
      "kv_offload": "lmcache_cpu+auto"
    },
    "others": {
      "kv_mem_offload_size": 40,
      "speculative_decode": {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "moe_backend": null
      }
    }
  }
}
```

## 3. 字段说明

### `features`：页面开关的唯一回显依据

> [!IMPORTANT]
> 页面必须直接使用 `features` 中对应字段的布尔值回显开关：`true` 显示开启，`false` 显示关闭。不得使用页面原始请求值、环境变量、`variants` 或 `others` 推断开关状态。

| 页面开关 | 唯一依赖字段 | `true` | `false` |
| --- | --- | --- | --- |
| 投机推理 | `features.speculative_decode` | 开启 | 关闭 |
| KV Cache 稀疏 | `features.sparse_kv` | 开启 | 关闭 |
| KV Cache 卸载 | `features.kv_offload` | 开启 | 关闭 |
| RAG 加速 | `features.rag_acc` | 开启 | 关闭 |

`features` 表示经过模型、卡型、白名单、容量和运行条件收口后的最终有效状态。即使页面请求值为开启，只要接口返回 `false`，页面就必须回显为关闭。

### `variants`：实际策略

| 字段 | 说明 |
| --- | --- |
| `speculative_decode` | 投机策略，例如 `mtp`、`suffix`、`eagle3`、`dflash` |
| `sparse_kv` | 稀疏策略，例如 `fp8`、`indexcache_use_index_cache_topk4` |
| `kv_offload` | 卸载后端和模式，例如 `native_kv_offloading_backend`、`lmcache_cpu+auto`、`memcache` |

当 `features.kv_offload=false` 时，variant 仍可能包含 `floor_disabled` 等诊断信息。这些内容仅用于说明未生效原因，严禁据此把开关显示为开启。

### `others.kv_mem_offload_size`：智能推荐容量

> [!IMPORTANT]
> 内存智能推荐模式下，页面必须识别 `others.kv_mem_offload_size`，并使用该字段回显后端计算出的最终推荐容量。不得继续显示字符串 `auto`，也不得使用页面提交前的估算值。

当请求侧配置 `KV_MEM_OFFLOAD_SIZE=auto` 时，后端会完成容量计算，并将节点级推荐结果写入该字段。

| 值 | 页面含义 |
| --- | --- |
| 正整数 | 推荐容量，单位为 GB，例如 `40` 回显为 `40 GB` |
| `0` | 推荐结果低于生效下限，内存卸载未生效 |
| `null` | 当前场景没有有效容量或缺少计算输入 |

页面必须同时遵守以下规则：

1. KV 卸载开关只由 `features.kv_offload` 控制。
2. 当前为智能推荐模式、`features.kv_offload=true` 且 `kv_mem_offload_size` 为正整数时，页面必须按 `<数值> GB` 回显最终推荐容量，例如 `40` 回显为 `40 GB`。
3. `kv_mem_offload_size` 为 `0` 或 `null` 时，不得展示为已生效容量。
4. `kv_mem_offload_size` 只负责容量回显，不能反向决定 KV 卸载开关状态。

### `others.speculative_decode`

投机推理详细信息，可能包含 `method`、`num_speculative_tokens`、`moe_backend`、`enforce_eager` 和 `draft_sample_method`；未产生对应详情时为 `null`。

## 4. 依赖的 JSON 文件

接口直接读取：

```text
/shared-volume/advanced_features.json
```

可通过 `ADVANCED_FEATURES_FILE` 环境变量覆盖路径。该文件的完整格式样例如下：

```json
{
  "engine": "vllm",
  "features": {
    "speculative_decode": true,
    "sparse_kv": true,
    "kv_offload": true,
    "rag_acc": false
  },
  "variants": {
    "speculative_decode": "mtp",
    "sparse_kv": "fp8_indexcache_use_index_cache_topk4",
    "kv_offload": "lmcache_cpu+auto"
  },
  "others": {
    "kv_mem_offload_size": 40,
    "speculative_decode": {
      "method": "mtp",
      "num_speculative_tokens": 3,
      "moe_backend": null
    }
  }
}
```

接口响应中的 `data` 就是该 JSON 对象；接口外层只增加 `code` 和 `msg`。

## 5. 依赖链路

```text
页面参数/环境变量/配置文件
  -> load_and_merge_configs()
  -> apply_effective_feature_enablement() 进行白名单和有效态收口
  -> vllm_adapter 解析最终策略与容量
  -> wings_entry 写入 advanced_features.json
  -> health_service 通过 /v1/startup/accel 返回
```

主要依赖：

- FastAPI/Uvicorn health 服务，默认监听 `19000`
- launcher 与 health 服务可共同访问的共享卷
- `smart_feature_whitelist.json` 模型和卡型能力白名单
- `config_loader.py` 有效态收口
- `vllm_adapter.py` 策略和容量解析
- `wings_entry.py` 状态文件写入及运行时回退更新

该接口不依赖数据库或 Redis，也不会在请求时重新访问推理引擎或重新计算特性。

## 6. 异常边界

如果 `advanced_features.json` 不存在、损坏或不是 JSON 对象，接口目前仍返回 HTTP `200`，但 `features`、`variants` 和 `others` 可能为空对象。页面除检查 HTTP 状态外，还应确认所需布尔字段存在。

引擎触发高级特性回退时，状态文件会将投机推理、KV 稀疏和 KV 卸载更新为 `false`，接口随之回显关闭状态。
