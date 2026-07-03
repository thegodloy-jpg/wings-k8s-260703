## 环境信息

| 机器 | IP | 账户 | 密码 | GPU | 工作目录 |
|------|-----|------|------|-----|---------|
| a100 | 7.6.52.148 | root | xfusion@1234! | 1× A100-40GB + 1× L20-46GB | /home/zhanghui |
| ubuntu2204 | 7.6.16.150 | root | Xfusion@2026 | 2× RTX5090 + 2× L20-49GB + 1× RTX4090 | /home/zhanghui |

## 设置免密登录
## task
1. 创建一个容器，使用vllm拉起qwen3-0.6.
