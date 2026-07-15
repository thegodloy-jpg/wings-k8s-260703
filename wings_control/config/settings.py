"""全局配置定义。

这个模块的配置会被 launcher、proxy、health 共同使用，因此这里的默认值
实际上就是 sidecar 架构的运行契约。修改端口、应用入口或共享卷路径时，
通常需要同时检查 Kubernetes 清单和 engine 容器启动逻辑是否仍然一致。

配置优先级（从高到低）：
  1. 系统环境变量 → 由 K8s ConfigMap/Secret 或 docker run -e 注入
  2. .env 文件   → 本地开发时使用（pydantic-settings 自动加载）
  3. 代码默认值   → 本模块中的 default= 参数

核心设计要点：
  - 所有端口、路径都通过环境变量驱动，便于 K8s 动态配置
  - Settings 类在模块级别实例化为单例 `settings`，进程内全局唯一
  - pydantic-settings 自动将字段名映射为同名环境变量，支持类型转换
"""

from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """全局配置类，集中管理 sidecar 运行时的所有可配置参数。

    配置项涵盖：
      - 共享卷路径与启动脚本文件名
      - 引擎类型、主机和端口
      - sidecar 三层端口规划（backend / proxy / health）
      - 子服务启动入口（uvicorn 模块路径）
      - 模型路径、TP 大小等引擎参数
      - 历史兼容字段（健康检查、NodePort 等）

    所有参数均通过环境变量驱动，支持 K8s ConfigMap/Secret 注入，
    同时通过 pydantic-settings 自动加载 .env 文件用于本地开发。
    本类在模块级别实例化为单例 ``settings``，进程内全局唯一。

    注意：pydantic-settings 会自动将字段名映射为同名环境变量读取，
    因此不需要手动调用 os.getenv()。字段默认值即为环境变量不存在时的回退值。
    """
    # 允许 MODEL_NAME 等 model_ 前缀字段不触发 Pydantic v2 保护命名空间告警；
    # env_file=".env" 启用本地开发时自动加载 .env 文件。
    model_config = {"protected_namespaces": (), "env_file": ".env"}
    # 共享卷路径：launcher 在这里写 start_command.sh，engine 容器在这里读取。
    SHARED_VOLUME_PATH: str = "/shared-volume"
    START_COMMAND_FILENAME: str = "start_command.sh"

    # log_analyzer 进度文件完整路径（由 SHARED_VOLUME_PATH 派生，见 _derive_volume_paths）
    # 若需独立覆盖，可直接设置对应环境变量 PROGRESS_FILE / ADVANCED_FEATURES_FILE。
    #   - ADVANCED_FEATURES_FILE（advanced_features.json）：页面状态汇报的「使能 + 变体」JSON 对象，
    #     是 /v1/startup/accel 的单一真相源
    PROGRESS_FILE: str = ""
    ADVANCED_FEATURES_FILE: str = ""

    @model_validator(mode="before")
    @classmethod
    def _derive_volume_paths(cls, data: Any) -> Any:
        """在所有字段验证完成前，若进度/加速相关文件路径未被环境变量覆盖，
        则根据 SHARED_VOLUME_PATH 自动生成默认路径。"""
        if not isinstance(data, dict):
            return data
        shared = data.get("SHARED_VOLUME_PATH") or "/shared-volume"
        if not data.get("PROGRESS_FILE"):
            data["PROGRESS_FILE"] = f"{shared}/progress.jsonl"
        if not data.get("ADVANCED_FEATURES_FILE"):
            data["ADVANCED_FEATURES_FILE"] = f"{shared}/advanced_features.json"
        return data

    # 引擎基础配置。ENGINE_PORT 是 backend 真实监听端口，不是对外代理端口。
    ENGINE: str = "vllm"
    ENGINE_HOST: str = "127.0.0.1"
    ENGINE_PORT: int = 17000
    ENABLE_REASON_PROXY: bool = True

    # sidecar 三层端口：
    # - backend: engine 真正服务端口
    # - proxy:   对外 API 端口
    # - health:  专用健康检查端口（同时透传监控接口）
    PORT: int = 18000
    HEALTH_PORT: int = 19000
    WINGS_PORT: int = 9000  # legacy field

    # 子服务启动入口。launcher 会调用 `python -m uvicorn <app>` 启动对应服务。
    PYTHON_BIN: str = "python"
    UVICORN_MODULE: str = "uvicorn"
    PROXY_APP: str = "proxy.gateway:app"
    HEALTH_APP: str = "proxy.health_service:app"
    PROCESS_POLL_SEC: float = 1.0

    # 与历史 wings_start 语义兼容的模型默认参数。
    MODEL_NAME: str = ""
    MODEL_PATH: str = "/usr/local/serving/models"
    SAVE_PATH: str = "/opt/wings/outputs"
    TP_SIZE: int = 1
    MAX_MODEL_LEN: int = 4096

    # ---- 分布式模式配置 ----
    # 这些字段由 _determine_role() 使用，控制 Master/Worker 模式分支。
    # K8s 仅需注入 DISTRIBUTED=true、MASTER_IP 和 NODE_IPS（逗号分隔），
    # 其余分布式参数（nnodes/node_rank/head_node_addr）由 Master 动态计算注入。
    DISTRIBUTED: bool = False
    MASTER_IP: str = ""
    NODE_IPS: str = ""  # 逗号分隔的所有节点 IP

    # ---- 历史兼容字段 ----
    # 这些字段在 v4 架构中已不再是核心配置，但仍被部分旧版部署模板
    # 和 legacy wings_start.sh 脚本引用，因此保留以保证向后兼容。
    HEALTH_CHECK_INTERVAL: int = 5  # 健康检查间隔（秒）
    HEALTH_CHECK_TIMEOUT: int = 300  # 健康检查超时（秒）
    SERVICE_CLUSTER_IP: str = ""  # K8s Service ClusterIP（可选）
    NODE_PORT: str = "30483"  # K8s NodePort 端口号
    NODE_IP: str = ""  # 当前宿主机 IP
    ENABLE_ACCEL: bool = True  # 是否启用 Accel 加速包安装片段
    WINGS_ACCEL_DIR: str = "/accel-volume"  # Accel 加速包运行时目录（initContainer 从镜像 /opt/packages 拷贝到此处）

    # ---- 全局 HTTP 超时配置 ----
    HTTP_REQUEST_TIMEOUT: float = 30.0  # 分布式节点间 HTTP 请求超时（秒）
    HEARTBEAT_TIMEOUT: float = 10.0  # 心跳请求超时（秒）
    REGISTRATION_TIMEOUT: float = 10.0  # 节点注册请求超时（秒）

    # ---- 环境变量挂载点 ----
    # 用户可通过 K8s ConfigMap 将 .env/.sh 文件挂载到此目录，
    # 这些文件会在 start_command.sh 生成时自动注入到引擎启动脚本前置环节。
    ENV_OVERRIDES_DIR: str = "config/env_overrides"


# 模块级别的全局配置单例。
# 整个 sidecar 进程内所有子系统均通过 `from config.settings import settings` 引用此实例。
settings = Settings()
