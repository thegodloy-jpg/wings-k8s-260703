# -*- coding: utf-8 -*-
"""
双闸门 FIFO 排队控制器（multi-worker 版）。

设计分层:
   Gate-0: GATE0_LOCAL_CAP 个并发槽，高优先级请求
   Gate-1: GATE1_LOCAL_CAP 个并发槽，普通请求溢出
   Queue:  LOCAL_QUEUE_MAXSIZE 个等待者，FIFO 软队列

aqucire() 调用时:
    1) 尝试 Gate-0 -> Gate-1 -> 队列
    2) 队列等待直到被唤醒或超时

release() 调用时:
    1) 有等待者则移交令牌
    2) 无等待者则释放 semaphore

QUEUE_OVERFLOW_MODE: block / drop_oldest / reject
"""

import time
import asyncio
import json
from typing import Dict, Optional
from fastapi import HTTPException
from . import proxy_config as C


# =============================================================================
# 日志辅助函数
# =============================================================================


def _jlog(evt: str, **fields):
    """输出 INFO 级别结构化 JSON 日志。

    Args:
        evt: 事件名称
        **fields: 额外字段
    """
    try:
        log_entry = {"evt": evt, **fields}
        C.logger.info(json.dumps(log_entry, ensure_ascii=False))
    except (TypeError, ValueError) as e:
        C.logger.error("Failed to serialize log entry: %s", log_entry, exc_info=e)
    except Exception as e:
        C.logger.error("Unexpected error while writing log", exc_info=e)


def _elog(evt: str, **fields):
    """输出 ERROR 级别结构化 JSON 日志。

    Args:
        evt: 事件名称
        **fields: 额外字段
    """
    try:
        log_entry = {"evt": evt, **fields}
        C.logger.error(json.dumps(log_entry, ensure_ascii=False))
    except (TypeError, ValueError) as e:
        C.logger.error("Failed to serialize log entry: %s", log_entry, exc_info=e)
    except Exception as e:
        C.logger.error("Unexpected error while writing log", exc_info=e)


def _ms(sec: float) -> str:
    """将秒数格式化为毫秒字符串（例如 "12.3ms"）。

    Args:
        sec: 时间秒数

    Returns:
        str: 格式化的毫秒字符串
    """
    return f"{sec*1000:.1f}ms"


class Waiter:
    """队列等待者，封装单个挂起请求的 Future 及其元数据。

    当 Gate-0 和 Gate-1 均无空余槽位时，进入等待的请求会被包装成
    一个 Waiter 放入 asyncio.Queue，待 release() 触发唤醒后取出执行。

    Attributes:
        fut:     关联的 asyncio.Future，由 release() 调用 set_result(layer) 唤醒，
                 layer 为 0 (Gate-0) 或 1 (Gate-1)。
        enq_ts:  入队时刻（perf_counter 秒），用于计算排队等待耗时。
        pos:     入队时的队列位置编号（仅作日志参考）。
    """
    __slots__ = ("fut", "enq_ts", "pos")

    def __init__(self, fut: asyncio.Future, enq_ts: float, pos: int):
        self.fut = fut          # set_result(layer:int) 0:Gate-0, 1:Gate-1
        self.enq_ts = enq_ts    # 入队时间戳
        self.pos = pos          # 队列位置编号


class QueueGate:
    """双闸门 FIFO 排队控制器，为代理层提供背压与公平准入。

    设计分层：
    ┌─ Gate-0 ─── GATE0_LOCAL_CAP 个并发槽（高优先级请求或预留资源）
    ├─ Gate-1 ─── GATE1_LOCAL_CAP 个并发槽（普通请求溢出通道）
    └─ Queue  ─── 最多 LOCAL_QUEUE_MAXSIZE 个等待者（FIFO 软队列）

    请求进入时先尝试 Gate-0，再尝试 Gate-1；若均已满则进入队列等待。
    release() 在有等待者时优先移交令牌（handover），否则释放 semaphore。

    溢出策略（QUEUE_OVERFLOW_MODE）：
    - block      ：put() 阻塞直到队列有空位
    - drop_oldest：丢弃队列最老的一个等待者以腾出空位
    - 其它       ：直接返回 503
    """
    def __init__(self):
        # 最大并发请求数（app/healthz 接口除外）
        self.max_inflight = int(C.LOCAL_PASS_THROUGH_LIMIT)

        # Gate-0 + Gate-1 容量分配
        g0_cap = max(0, int(getattr(C, "GATE0_LOCAL_CAP", 0)))
        g1_cap = max(0, int(getattr(C, "GATE1_LOCAL_CAP", max(0, self.max_inflight - g0_cap))))

        self.g0_cap = g0_cap
        self.g1_cap = g1_cap
        self.g0 = asyncio.Semaphore(self.g0_cap) if self.g0_cap > 0 else None
        self.g1 = asyncio.Semaphore(self.g1_cap) if self.g1_cap > 0 else None

        # 队列配置
        self.max_qsize = int(C.LOCAL_QUEUE_MAXSIZE)
        self.q: Optional[asyncio.Queue[Waiter]] = (
            asyncio.Queue(maxsize=self.max_qsize) if self.max_qsize > 0 else None
        )

        # 0/1
        self._holders: Dict[int, int] = {}

        _jlog("qgate_init",
              max_inflight=self.max_inflight,
              g0_cap=self.g0_cap, g1_cap=self.g1_cap,
              qmax=self.max_qsize)
    # ── Gate-0 + Gate-1 操作 ──

    @property
    def inflight(self) -> int:
        return self._sem_inflight(self.g0, self.g0_cap) + self._sem_inflight(self.g1, self.g1_cap)


    #    #


    @staticmethod
    def _task_id() -> int:
        t = asyncio.current_task()
        return 0 if t is None else id(t)


    @staticmethod
    def _has_ticket(sem: Optional[asyncio.Semaphore]) -> bool:
        if sem is None:
            return False
        v = getattr(sem, "_value", 0)
        return v > 0


    @staticmethod
    def _sem_inflight(sem: Optional[asyncio.Semaphore], cap: int) -> int:
        if sem is None or cap <= 0:
            return 0
        rem = getattr(sem, "_value", None)
        if rem is None:
            return 0
        return max(0, cap - int(rem))


    @staticmethod
    def _queue_disabled_raise(rid: str | None) -> None:
        _elog("qgate_queue_disabled", rid=rid)
        raise HTTPException(
            status_code=503, detail="server busy: queue disabled",
            headers={"Retry-After": "1", "Connection": "close", "X-Queue-Disabled": "true"}
        )


    # QueueGate public interface methods


    def obs_headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """/"""
        hdr = {
            "X-InFlight": str(self.inflight),
            "X-Queue-Size": str(self.queue_size()),
            "X-Local-MaxInflight": str(self.max_inflight),
            "X-Local-QueueMax": str(self.max_qsize),
            "X-Workers": str(C.WORKERS),
            "X-Global-MaxInflight": str(C.GLOBAL_PASS_THROUGH_LIMIT),
            "X-Global-QueueMax": str(C.GLOBAL_QUEUE_MAXSIZE),
            "X-Queue-Timeout-Sec": str(C.QUEUE_TIMEOUT),
            #
            "X-InFlight-G0": str(self._sem_inflight(self.g0, self.g0_cap)),
            "X-InFlight-G1": str(self._sem_inflight(self.g1, self.g1_cap)),
            "X-MaxInflight-G0": str(self.g0_cap),
            "X-MaxInflight-G1": str(self.g1_cap),
        }
        if extra:
            hdr.update(extra)
        return hdr


    async def release(self):
        """

          -    semaphore
          -    sem inflight-1
        """
        # 0/1 0
        task_id = self._task_id()
        if task_id not in self._holders:
            # 未经 acquire() 就调用 release()，跳过以避免信号量溢出
            _elog("qgate_release_without_acquire", task_id=task_id)
            return
        layer = self._holders.pop(task_id)

        # Release semaphore or hand off slot to next queued waiter
        if self.q is not None:
            while not self.q.empty():
                waiter: Waiter = await self.q.get()
                if waiter.fut.cancelled() or waiter.fut.done():
                    continue
                try:
                    waiter.fut.set_result(layer)  #
                    _jlog("qgate_handover", layer=layer, remain_qsize=self.queue_size())
                except Exception as e:
                    _elog("qgate_handover_error", layer=layer, error=str(e))
                return

        #
        sem = self.g0 if layer == 0 else self.g1
        if sem is not None:
            try:
                sem.release()
                _jlog("qgate_release", layer=layer)
            except ValueError:
                #
                _elog("qgate_release_double", layer=layer)


    def queue_size(self) -> int:
        """返回当前软队列中等待的请求数；队列禁用时返回 0。"""
        return 0 if self.q is None else self.q.qsize()

    async def acquire(self, req_headers: Dict[str, str]) -> Dict[str, str]:
        """

        1)  Gate-0    Gate-0
        2)  Gate-1    Gate-1
        3)   //  503
        4)  acquire
        """
        headers_out: Dict[str, str] = {}
        t0 = time.perf_counter()
        rid = req_headers.get("x-request-id") or ""

        self._log_acquire_try(rid, t0)

        # Try direct fast-path through Gate-0, then Gate-1
        if await self._try_direct_gate(self.g0, self.g0_cap, 0, rid, t0):
            headers_out["X-Queued-Wait"] = "0.0ms"
            return headers_out

        if await self._try_direct_gate(self.g1, self.g1_cap, 1, rid, t0):
            headers_out["X-Queued-Wait"] = "0.0ms"
            return headers_out


        #
        if self.q is None:
            self._queue_disabled_raise(rid)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        waiter = Waiter(fut=fut, enq_ts=time.perf_counter(), pos=0)

        #
        if self.q.full():
            self._handle_queue_full(rid)

        #
        await self._enqueue_waiter(waiter, headers_out, rid)

        # Wait for slot handover from a concurrent release() call
        layer = await self._wait_for_wakeup(fut, waiter, rid)
        return self._inherit_occupy(layer, headers_out, waiter, rid)

    def _log_acquire_try(self, rid: str | None, t0: float) -> None:
        _jlog(
            "qgate_acquire_try",
            rid=rid,
            inflight=self.inflight,
            g0_inflight=self._sem_inflight(self.g0, self.g0_cap),
            g1_inflight=self._sem_inflight(self.g1, self.g1_cap),
            qsize=self.queue_size(),
        )

    def _handle_queue_full(self, rid: str | None) -> None:
        policy = C.QUEUE_REJECT_POLICY
        if policy == "drop_oldest":
            dropped = None
            while not self.q.empty():
                w: Waiter = self.q.get_nowait()
                if not (w.fut.cancelled() or w.fut.done()):
                    w.fut.set_exception(HTTPException(
                        status_code=503, detail="server busy: dropped oldest",
                        headers={"Retry-After": "1", "Connection": "close", "X-Queue-Dropped": "oldest"}
                    ))
                    dropped = w
                    break
            _elog("qgate_drop_oldest", rid=rid, dropped=bool(dropped))
            if dropped is None and C.QUEUE_OVERFLOW_MODE != "block":
                _elog("qgate_queue_full_reject", rid=rid)
                raise HTTPException(
                    status_code=503, detail="server busy: queue full",
                    headers={"Retry-After": "1", "Connection": "close", "X-Queue-Full": "true"}
                )
            # Overflow mode is 'block': fall through to blocking put()
        elif C.QUEUE_OVERFLOW_MODE != "block":
            _elog("qgate_queue_full_reject", rid=rid)
            raise HTTPException(
                status_code=503, detail="server busy: queue full",
                headers={"Retry-After": "1", "Connection": "close", "X-Queue-Full": "true"}
            )
        # else:  block put()

    async def _enqueue_waiter(self, waiter: "Waiter", headers_out: Dict[str, str], rid: str | None) -> None:
        pos = self.q.qsize() + 1
        waiter.pos = pos
        await self.q.put(waiter)
        headers_out["X-Queue-Position"] = str(pos)
        if C.QUEUE_OVERFLOW_MODE == "block":
            headers_out["X-Queue-Overflow"] = "block"
        _jlog("qgate_enqueued", rid=rid, pos=pos, qsize=self.queue_size())

    async def _wait_for_wakeup(self, fut: asyncio.Future, waiter: "Waiter", rid: str | None) -> int:
        try:
            layer = await asyncio.wait_for(fut, timeout=C.QUEUE_TIMEOUT)
            return int(layer) if layer in (0, 1) else 0
        except asyncio.TimeoutError as e:
            if not fut.done():
                fut.cancel()
            _elog("qgate_timeout", rid=rid, waited=_ms(time.perf_counter() - waiter.enq_ts))
            raise HTTPException(
                status_code=503,
                detail="server busy: queue timeout",
                headers={"Retry-After": "1", "Connection": "close", "X-Queue-Timeout": "true"}
            ) from e

    def _inherit_occupy(
            self, layer: int,
            headers_out: Dict[str, str],
            waiter: "Waiter",
            rid: str | None) -> Dict[str, str]:
        self._holders[self._task_id()] = layer
        headers_out["X-Queued-Wait"] = f"{(time.perf_counter() - waiter.enq_ts) * 1e3:.1f}ms"
        _jlog("qgate_wakeup", rid=rid, layer=layer, waited=headers_out["X-Queued-Wait"])
        return headers_out

    async def _try_direct_gate(
        self,
        gate: asyncio.Semaphore | None,
        cap: int,
        layer: int,
        rid: str | None,
        t0: float,
    ) -> bool:
        if cap > 0 and self._has_ticket(gate):
            await gate.acquire()
            self._holders[self._task_id()] = layer
            _jlog("qgate_acquire_direct", rid=rid, layer=layer, elapsed=_ms(time.perf_counter() - t0))
            return True
        return False