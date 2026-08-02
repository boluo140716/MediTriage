"""FastAPI 后端：SSE 流式输出 Swarm 内部事件 + REST 入口。

对外接口：
  POST /api/ask          { question, session_id? }  -> SSE 事件流
  GET  /                  -> 单文件前端
"""
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from loguru import logger

# 向上找仓库根（含 config.py）供 config 导入（config.py 不在包内）
ROOT = Path(__file__).resolve().parent.parent
for _d in ROOT.parents:
    if (_d / "config.py").is_file():
        sys.path.insert(0, str(_d))
        break

import re  # noqa: E402
from meditriage.swarm import process_with_swarm  # noqa: E402
from meditriage.memory import ShortTermMemory  # noqa: E402  单例；与 swarm 共享同一短期记忆

# 判断一句纯文本追问是否在"指代图片"，用于复用会话中最近上传的影像
_IMAGE_REF_PAT = re.compile(
    r"图片|图像|图中|图里|影像|这张|那张|照片|片子|X光|CT|MRI|核磁|超声|B超|扫描|截图",
    re.IGNORECASE,
)

app = FastAPI(
    title="MediTriage Agent Swarm",
    description="医疗 Agent 群体智能 + SSE 可视化",
)

# 静态前端
STATIC_INDEX = Path(__file__).parent / "index.html"


@app.get("/")
async def root():
    # no-cache：保证浏览器每次都拿到最新前端（避免 UI 改动被浏览器缓存掩盖）
    return FileResponse(
        STATIC_INDEX,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


def _fetch_vllm_meta():
    """直连 vLLM /v1/models 取真实 max_model_len（如实展示当前上下文窗口）。"""
    import urllib.request
    from config import LLM_CONFIG
    url = LLM_CONFIG["base_url"].rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=4) as r:
        d = json.loads(r.read())
    m = (d.get("data") or [{}])[0]
    return {"model": m.get("id"), "max_model_len": m.get("max_model_len")}


@app.get("/api/meta")
async def meta():
    try:
        return await asyncio.to_thread(_fetch_vllm_meta)
    except Exception as e:
        return {"model": None, "max_model_len": None, "err": str(e)[:140]}


class AskRequest(BaseModel):
    # 上限：问题 8000 字（撑爆上下文/记忆的超长输入早退 422，不进推理链路）
    question: str = Field(..., max_length=8000)
    # session_id 是记忆访问键且会拼进落盘路径，白名单字符防路径遍历
    session_id: Optional[str] = Field(
        None, max_length=64, pattern=r"^[A-Za-z0-9_-]{1,64}$"
    )
    # 多模态：前端把图片转成 data URI（data:image/png;base64,...）放这里。
    # 有图 -> 走单轮 VLM 视觉链路（vision_handler）；无图 -> 走多 Agent swarm。
    # 仅接受 data:image/*——http(s)/file 透传给 vLLM 会成为 SSRF/本地读取面
    image: Optional[str] = Field(None, max_length=8_000_000)
    image_name: Optional[str] = Field(None, max_length=255)

    @field_validator("image")
    @classmethod
    def _image_must_be_data_uri(cls, v):
        if v is not None and not v.startswith("data:image/"):
            raise ValueError("image 仅接受 data:image/* 格式的 data URI")
        return v


def _sse_pack(event_type: str, data: dict) -> str:
    """打包成 SSE 行。"""
    payload = json.dumps(
        {"type": event_type, "data": data, "ts": datetime.now().isoformat()},
        ensure_ascii=False,
        default=str,
    )
    return f"event: {event_type}\ndata: {payload}\n\n"


@app.post("/api/ask")
async def ask(req: AskRequest, request: Request):
    """SSE 流式输出 Agent 协作的全过程。"""
    session_id = req.session_id or (
        f"web-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
    )
    # 登录身份：Cloudflare Access 校验后注入的邮箱头，用于隔离各用户长期记忆。
    # 无此头（本地直连 / 未走 Access）则退回单租户默认值。
    user_id = request.headers.get("cf-access-authenticated-user-email") or None
    queue: asyncio.Queue = asyncio.Queue()

    def emitter(event_type: str, data: dict):
        """Swarm 内部事件 → 推到队列 → SSE 流。"""
        try:
            queue.put_nowait((event_type, data))
        except Exception:
            pass

    async def run_swarm():
        """运行主流程：有图走单轮 VLM 视觉链路，无图走多 Agent swarm。"""
        import time as _time
        t0 = _time.time()
        try:
            # 单例，与 swarm 同一份会话记忆
            stm = ShortTermMemory(storage_type="memory")

            # 决定是否走视觉链路：本轮带图，
            # 或"指代图片"的纯文本追问且会话存有最近影像
            image_to_use = req.image
            reused_image = False
            if (not image_to_use and req.question
                    and _IMAGE_REF_PAT.search(req.question)):
                prev = stm.get_session_image(session_id)
                if prev:
                    image_to_use = prev
                    reused_image = True
                    logger.info(f"复用会话最近影像应答图片追问 (session={session_id})")

            if image_to_use:
                # 多模态：医学图像问答（VLM）。带近期对话上下文，
                # 可应答"这张图和发烧有关吗"等追问。
                if req.image:
                    # 记住本轮上传的图，供后续文字追问复用
                    stm.set_session_image(session_id, req.image)
                from meditriage.core.vision_handler import process_image_query
                result = await process_image_query(
                    image_path=image_to_use,
                    question=req.question or "请分析这张医学影像，指出主要发现与可能的异常。",
                    session_id=session_id,
                    # 传入单例，使图像问答写入同一会话记忆
                    short_term_memory=stm,
                    event_emitter=emitter,
                )
                # 视觉链路不经 process_with_swarm，安全兜底在此单独套：
                # 危机热线 + 用药剂量提示（如拍药盒问婴儿用量）
                _va = result.get("answer", "")
                if _va:
                    from meditriage.swarm.swarm_coordinator import (
                        _ensure_crisis_support, _ensure_medication_safety)
                    _va = _ensure_crisis_support(req.question or "", _va)
                    _va = _ensure_medication_safety(req.question or "", _va)
                    result["answer"] = _va
                emitter("result", {
                    "answer": result.get("answer", ""),
                    "suggestions": [],
                    "disclaimer": result.get("disclaimer", ""),
                    "swarm_enabled": False,
                    "agents_involved": ["vision"],
                    "mode": "vision",
                    "reused_image": reused_image,
                    "session_id": session_id,
                    "total_time": round(_time.time() - t0, 2),
                    "error": result.get("error", False),
                })
            else:
                result = await process_with_swarm(
                    question=req.question,
                    session_id=session_id,
                    event_emitter=emitter,
                    user_id=user_id,
                )
                # 最终完整结果（带 suggestions、disclaimer）
                emitter("result", {
                    "answer": result.get("answer", ""),
                    "suggestions": result.get("suggestions", []),
                    "disclaimer": result.get("disclaimer", ""),
                    "swarm_enabled": result.get("swarm_enabled", False),
                    "agents_involved": result.get("agents_involved", []),
                    "session_id": session_id,
                    "total_time": result.get("total_time"),
                    "timeout_occurred": result.get("timeout_occurred", False),
                })
        except Exception as e:
            logger.exception("Swarm execution failed")
            emitter("error", {"message": str(e)})
        finally:
            # 用 sentinel 通知 SSE 流可以结束
            await queue.put(("__END__", {}))

    async def event_stream():
        task = asyncio.create_task(run_swarm())
        # 立刻给前端一个 ack
        yield _sse_pack("ack", {
            "session_id": session_id,
            "question": req.question,
            "has_image": bool(req.image),
            "image_name": req.image_name,
        })
        try:
            while True:
                event_type, data = await queue.get()
                if event_type == "__END__":
                    yield _sse_pack("end", {"session_id": session_id})
                    break
                yield _sse_pack(event_type, data)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关掉 nginx 缓冲
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
