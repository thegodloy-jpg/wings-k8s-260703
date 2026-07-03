import json
import os
import time
from datetime import datetime, timezone


def create_default_progress_data(include_timestamp: bool = True) -> dict:
    """创建默认的进度数据。

    Args:
        include_timestamp: 是否包含时间戳

    Returns:
        dict: 默认进度数据
    """
    data = {
        "progress": 0,
        "phase_code": "init",
        "phase_name": "wings-control初始化",
        "status": "running",
        "key_log": "",
        "start_time": "",
        "curr_time": "",
        "elapsed_time_s": 0
    }
    if include_timestamp:
        data["start_time"] = ""
        data["curr_time"] = ""
    return data


def create_error_progress_data(error: Exception) -> dict:
    """创建错误状态的进度数据。

    Args:
        error: 异常对象

    Returns:
        dict: 错误进度数据
    """
    return {
        "progress": 0,
        "phase_code": "error",
        "phase_name": "错误",
        "status": "failed",
        "key_log": str(error),
        "start_time": "",
        "curr_time": "",
        "elapsed_time_s": 0
    }


def read_progress_file(file_path: str) -> dict:
    """读取进度文件并解析最后一行。

    Args:
        file_path: 进度文件路径

    Returns:
        dict: 解析后的进度数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if lines:
            # 获取最后一行作为当前进度
            last_line = lines[-1].strip()
            if last_line:
                return json.loads(last_line)
    return create_default_progress_data(include_timestamp=False)


def build_progress_response(progress_data: dict, message: str = "") -> dict:
    """构建进度响应。

    Args:
        progress_data: 进度数据
        message: 响应消息

    Returns:
        dict: 进度响应
    """
    return {
        "code": 200,
        "msg": message,
        "data": progress_data
    }


class StartupProgressManager:
    """启动进度管理器
    
    功能：
    1. 维护服务端的进度状态实例
    2. 每次请求时读取进度文件并更新状态
    3. 当文件进度与实例进度一致时，自动递增进度（不超过阶段最大值）
    4. 提供完整的进度信息（进度值、阶段、状态、时间、日志等）
    """
    
    def __init__(self, engine: str = "vllm"):
        """初始化进度管理器
        
        Args:
            engine: 引擎类型 (vllm, sglang, vllm_ascend 等)
        """
        # 进度状态实例
        self.current_progress = 0
        self.phase_code = "init"
        self.phase_name = "wings-control初始化"
        self.status = "running"
        self.start_time = int(time.time())  # 使用时间戳（秒），初始化时设置为当前时间
        self.elapsed_time_s = 0
        self.key_log = ""
        self.is_completed = False  # 进度是否完成
        self.engine = engine  # 引擎类型
        
        # 根据引擎类型选择阶段定义
        self.stage_definitions = self._get_stage_definitions()


    def update_from_file(self, file_progress: dict) -> dict:
        """从文件更新进度状态
        
        Args:
            file_progress: 从进度文件读取的数据
            
        Returns:
            dict: 更新后的进度数据
        """

        # 动态计算耗时（基于实例的start_time）
        if self.start_time > 0 and not self.is_completed:
            self.elapsed_time_s = int(time.time()) - self.start_time

        file_progress_value = file_progress.get("progress", 0)
        file_status = file_progress.get("status", "running")
        
        # 判断进度是否完成（completed或failed）
        self.is_completed = file_status in ["completed", "failed"]
        
        # 同步所有字段（除了 progress, start_time, elapsed_time_s）
        self.phase_code = file_progress.get("phase_code", "init")
        self.phase_name = file_progress.get("phase_name", "")
        self.status = file_status
        self.key_log = file_progress.get("key_log", "")
        
        # 判断文件进度值与实例进度值的关系
        if file_progress_value > self.current_progress:
            # 文件进度值大于实例进度值，以进度文件为准
            self.current_progress = file_progress_value
        else:
            # 文件进度值小于等于实例进度值，实例进度值自增1（避免卡顿）
            if self._should_auto_increment():
                self._increment_progress()
        
        # 返回更新后的进度数据
        return self._build_progress_dict()
        
    def get_initial_progress_data(self) -> dict:
        """获取进度实例的初始化信息
        
        Returns:
            dict: 初始化进度数据
        """
        return {
            "progress": self.current_progress,
            "phase_code": self.phase_code,
            "phase_name": self.phase_name,
            "status": self.status,
            "key_log": self.key_log,
            "start_time": self.start_time,
            "elapsed_time_s": self.elapsed_time_s
        }

    def _get_stage_definitions(self) -> dict:
        """根据引擎类型获取阶段定义
        
        Args:
            engine: 引擎类型
            
        Returns:
            dict: 阶段定义字典
        """
        # vLLM 和 vLLM Ascend 的阶段定义（包含 model_compiling）
        if self.engine in ["vllm", "vllm_ascend"]:
            return {
                "init": {"min": 0, "max": 5, "name": "wings-control初始化"},
                "accel_enabling": {"min": 5, "max": 20, "name": "加速特性注入"},
                "engine_booting": {"min": 20, "max": 30, "name": "engine初始化"},
                "model_loading": {"min": 30, "max": 60, "name": "模型加载"},
                "model_compiling": {"min": 60, "max": 75, "name": "模型编译"},
                "cuda_graph_capturing": {"min": 75, "max": 98, "name": "CUDA图捕获"},
                "server_checking": {"min": 98, "max": 99, "name": "健康探测"},
                "ready": {"min": 99, "max": 100, "name": "启动终态"}
            }
        # SGLang 的阶段定义（包含 kv_cache_allocating，不包含 model_compiling）
        elif self.engine == "sglang":
            return {
                "init": {"min": 0, "max": 5, "name": "wings-control初始化"},
                "accel_enabling": {"min": 5, "max": 20, "name": "加速特性注入"},
                "engine_booting": {"min": 20, "max": 30, "name": "engine初始化"},
                "model_loading": {"min": 30, "max": 60, "name": "模型加载"},
                "kv_cache_allocating": {"min": 60, "max": 70, "name": "KV缓存分配"},
                "cuda_graph_capturing": {"min": 70, "max": 98, "name": "CUDA图捕获"},
                "server_checking": {"min": 98, "max": 99, "name": "健康探测"},
                "ready": {"min": 99, "max": 100, "name": "启动终态"}
            }
        # 默认使用 vLLM 的阶段定义
        else:
            return {
                "init": {"min": 0, "max": 5, "name": "wings-control初始化"},
                "accel_enabling": {"min": 5, "max": 20, "name": "加速特性注入"},
                "engine_booting": {"min": 20, "max": 30, "name": "engine初始化"},
                "model_loading": {"min": 30, "max": 60, "name": "模型加载"},
                "model_compiling": {"min": 60, "max": 75, "name": "模型编译"},
                "cuda_graph_capturing": {"min": 75, "max": 98, "name": "CUDA图捕获"},
                "server_checking": {"min": 98, "max": 99, "name": "健康探测"},
                "ready": {"min": 99, "max": 100, "name": "启动终态"}
            }

    
    def _should_auto_increment(self) -> bool:
        """判断是否应该自动递增进度
        
        Returns:
            bool: 是否需要递增
        """
        # 只有在running状态且进度未完成时才递增
        if self.status != "running" or self.is_completed:
            return False
        
        # 获取当前阶段的最大值
        stage_info = self.stage_definitions.get(self.phase_code)
        if not stage_info:
            return False
        
        # 如果进度已经达到阶段最大值，不再递增
        if self.current_progress >= stage_info["max"]:
            return False
        
        return True
    
    def _increment_progress(self):
        """递增进度值（不超过当前阶段最大值）"""
        stage_info = self.stage_definitions.get(self.phase_code)
        if stage_info:
            self.current_progress = min(self.current_progress + 1, stage_info["max"])
    
    def _build_progress_dict(self) -> dict:
        """构建进度字典
        
        Returns:
            dict: 进度数据字典
        """
        return self.get_initial_progress_data()


class StreamCollector:
    """RAG 流式收集器（多路 chunk 并行推理 → 合并流式输出）。

    对外接口：
        collector = StreamCollector(num_chunks, chunk_request, combine_request)
        async for chunk in collector.collect_and_stream():
            ...
    """

    def __init__(self, num_chunks: int, chunk_request, combine_request):
        self.num_chunks = num_chunks
        self.chunk_request = chunk_request
        self.combine_request = combine_request

    @staticmethod
    def _parse_sse_line(line: str) -> str:
        """解析单行 SSE data 行，返回 delta.content 文本；无内容时返回空字符串。"""
        if not line.startswith("data: ") or line == "data: [DONE]":
            return ""
        try:
            data = json.loads(line[6:])
            choices = data.get("choices", [])
            if not choices:
                return ""
            delta = choices[0].get("delta", {})
            return delta.get("content", "")
        except (ValueError, KeyError, IndexError):
            return ""

    @staticmethod
    async def _extract_content_from_sse(resp) -> str:
        """从 SSE 流式响应中提取完整的内容文本。

        解析 `data: {...}` 行，提取 choices[0].delta.content 并拼接。
        """
        content_parts: list[str] = []
        async for line in resp.aiter_lines():
            text = StreamCollector._parse_sse_line(line.strip())
            if text:
                content_parts.append(text)
        return "".join(content_parts)

    async def collect_and_stream(self):
        """并行请求各 chunk，再发合并请求，流式返回最终结果。

        chunk_request / combine_request 均为 @asynccontextmanager 工厂，
        调用后返回异步上下文管理器，yield httpx.Response 流式响应对象。
        """
        chunk_responses: list[str] = []

        # 逐个发送 chunk 请求并收集完整响应
        for i in range(self.num_chunks):
            try:
                async with self.chunk_request(i) as resp:
                    content = await self._extract_content_from_sse(resp)
                    chunk_responses.append(content)
            except Exception as e:
                chunk_responses.append(f"[chunk {i} error: {e}]")

        # 合并所有 chunk 的分析结果为单个字符串
        preliminary_analysis = "\n".join(chunk_responses)

        # 发送合并请求，流式返回最终结果
        async with self.combine_request(preliminary_analysis) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
