#!/usr/bin/env python3
"""极简 stdlib TCP 转发（宿主机 -> medix-fix 容器内 web server）。

为什么需要它：演示页 uvicorn 跑在 medix-fix 容器的 8080，但该容器只发布了
8000(vLLM)。不重建容器、不装 socat、不要 root —— 用 asyncio 裸 TCP 透传把宿主机
端口转到容器 IP。裸 TCP 透传对 SSE(text/event-stream) 友好：不缓冲、不解析
HTTP，长连接流式直通。

用法: python3 _web_proxy.py <listen_host> <listen_port> <target_host>
<target_port>
"""
import asyncio
import sys


async def _pipe(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _handle(local_r, local_w, target_host, target_port):
    try:
        remote_r, remote_w = await asyncio.open_connection(
            target_host, target_port
        )
    except Exception:
        try:
            local_w.close()
        except Exception:
            pass
        return
    await asyncio.gather(_pipe(local_r, remote_w), _pipe(remote_r, local_w))


async def main():
    if len(sys.argv) != 5:
        print(
            "usage: _web_proxy.py <listen_host> <listen_port> <target_host> <target_port>",
            file=sys.stderr,
        )
        sys.exit(2)
    lh, lp, th, tp = (
        sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    )
    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, th, tp), lh, lp, reuse_address=True
    )
    print(f"proxy {lh}:{lp} -> {th}:{tp}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
