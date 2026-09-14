"""
asyncio 日志噪音过滤（Windows / Proactor 事件循环）

背景：
    Windows 上 asyncio 默认使用 ProactorEventLoop。当客户端（浏览器）主动断开长连接时
    —— 例如刷新问答页、关掉标签页、EventSource 被 close、curl 中断 —— transport 会在
    `_ProactorBasePipeTransport._call_connection_lost` 回调里执行 `sock.shutdown(SHUT_RDWR)`，
    此时对端已复位，于是抛出：
        ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接
    asyncio 的默认异常处理器会把它打成一条 `ERROR:asyncio` + 堆栈，刷在控制台上。
    这不是业务错误，SSE 通道本身已经做了断连清理。

处理方式：
    安装一个自定义 loop 异常处理器，把"对端复位/断开"这类异常降级为 DEBUG 日志；
    其余异常一律交回默认处理器，保证真实错误依然可见。
"""
import asyncio
from typing import Optional

from app.core.logger import logger

# 需要静音的异常类型（都属于"对端主动断开"）
SILENT_EXCEPTIONS = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


def _quiet_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """loop 级异常处理器：过滤客户端断开噪音，其余交给默认处理器"""
    exc = context.get("exception")
    if isinstance(exc, SILENT_EXCEPTIONS):
        # 降级为 DEBUG：既不打 ERROR/堆栈，又保留可追溯性
        logger.debug(
            f"已忽略客户端断开导致的 asyncio 异常：{type(exc).__name__}: {exc}"
        )
        return
    loop.default_exception_handler(context)


def install_asyncio_noise_filter(loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """
    在当前（或指定）事件循环上安装噪音过滤器。
    建议在应用启动时（FastAPI lifespan）调用一次。

    :param loop: 目标事件循环；为空则取当前运行中的循环
    """
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("未获取到运行中的事件循环，跳过 asyncio 噪音过滤安装")
            return
    loop.set_exception_handler(_quiet_exception_handler)
    logger.debug("已安装 asyncio 噪音过滤器（忽略 WinError 10054 等断连异常）")


if __name__ == "__main__":
    # 自测：验证 ConnectionResetError 被静音、其他异常仍走默认处理器
    calls = []

    class _FakeLoop:
        def default_exception_handler(self, context):
            calls.append(context)

    fake = _FakeLoop()
    handler = _quiet_exception_handler

    handler(fake, {"exception": ConnectionResetError("[WinError 10054] 远程主机强迫关闭了一个现有的连接")})
    assert not calls, "ConnectionResetError 不应交给默认处理器"

    handler(fake, {"exception": ValueError("真实错误")})
    assert len(calls) == 1, "其它异常必须照常上报"

    logger.info("✅ asyncio 噪音过滤器自测通过")
