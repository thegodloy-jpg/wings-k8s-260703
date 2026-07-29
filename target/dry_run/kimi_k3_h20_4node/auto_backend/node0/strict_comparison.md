# 启动命令对比报告：kimi-k3-h20-node0-strict

- 结果：**FAIL**
- 通过：25
- 失败：3
- 待确认：0
- 信息项：0

## 差异

| 结果 | 范围 | 字段 | 规则 | 标准值 | 实际值 | 说明 |
|---|---|---|---|---|---|---|
| FAIL | cli | `host` | exact | `"0.0.0.0"` | `"7.6.25.57"` |  |
| FAIL | cli | `port` | exact | `8000` | `17000` |  |
| FAIL | cli | `distributed-executor-backend` | extra | `null` | `"mp"` | 实际命令额外 CLI 参数 |
