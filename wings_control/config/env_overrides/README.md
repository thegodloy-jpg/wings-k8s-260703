# 环境变量挂载点

此目录用于通过 K8s ConfigMap/Secret 挂载自定义环境变量文件。

## 使用方式

将环境变量文件放置在此目录下，文件格式要求：

1. **`.env` 文件**（key=value 格式）：
   ```
   MY_CUSTOM_VAR=value1
   ANOTHER_VAR=value2
   ```

2. **`.sh` 文件**（shell export 格式，支持更复杂的逻辑）：
   ```bash
   export MY_CUSTOM_VAR="value1"
   export ANOTHER_VAR="value2"
   ```

所有 `.env` 和 `.sh` 文件会按文件名字母序排列，在 `start_command.sh`
生成时自动注入到引擎启动脚本的 **前置环节**（在引擎启动命令之前执行）。

## K8s ConfigMap 挂载示例

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: custom-engine-env
data:
  custom_env.env: |
    VLLM_ATTENTION_BACKEND=FLASH_ATTN
    CUDA_VISIBLE_DEVICES=0,1,2,3
  extra_setup.sh: |
    #!/bin/bash
    export LD_LIBRARY_PATH="/opt/custom/lib:${LD_LIBRARY_PATH:-}"
---
# 在 Pod spec 中挂载到此目录
volumes:
  - name: custom-env
    configMap:
      name: custom-engine-env
containers:
  - name: wings-control
    volumeMounts:
      - name: custom-env
        mountPath: /opt/wings-control/wings_control/config/env_overrides
```

## 注意事项

- `.env` 文件中不支持 shell 变量展开 (`$VAR`)，如需变量展开请使用 `.sh` 格式
- `.sh` 文件会通过 `source` 命令执行，确保语法正确
- 文件名以 `.` 开头的隐藏文件会被忽略
- 环境变量注入发生在引擎启动之前，不会影响 sidecar 自身的配置
