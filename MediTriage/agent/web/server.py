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
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request
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


class DoctorReply(BaseModel):
    # 医生回复（模拟身份，无登录；认证在 P2-5）
    reply: str = Field(..., min_length=1, max_length=4000)


# 转诊闭环（P1-2）：低置信度/危机 -> 建单并附到 result 事件
async def _attach_escalation(payload: dict, question: str,
                             session_id: str, user_id: Optional[str],
                             on_created: Optional[Callable[[dict], None]] = None) -> dict:
    """评估是否转人工；命中则把转诊单信息附到 result payload。

    转诊评估失败只记日志，绝不阻断主问答链路。
    on_created：建单成功后的回调，用于立刻向患者端 SSE 推送转诊通知
    （不等最终 result，患者可尽早感知"已转人工"）。
    """
    try:
        from meditriage.workflows import get_escalation_service
        esc = await get_escalation_service().evaluate(
            question=question,
            answer=payload.get("answer", ""),
            session_id=session_id,
            user_id=user_id,
            result=payload,
        )
        if esc:
            payload["escalation"] = esc
            if on_created:
                try:
                    on_created(esc)
                except Exception:
                    logger.exception("转诊 SSE 通知推送失败")
    except Exception as ex:
        logger.exception(f"转诊评估失败: {ex}")
    return payload


def _sse_pack(event_type: str, data: dict) -> str:
    """打包成 SSE 行。"""
    payload = json.dumps(
        {"type": event_type, "data": data, "ts": datetime.now().isoformat()},
        ensure_ascii=False,
        default=str,
    )
    return f"event: {event_type}\ndata: {payload}\n\n"


# 最终回答分块流式推送（思考完成后打字机效果；前端收到 result 后替换为完整卡片）
DEFAULT_ANSWER_CHUNK_INTERVAL = 0.03


def _answer_chunk_size(text_len: int) -> int:
    """分片大小：短回答 24 字符起步，长回答自动加大，保证 ≤ ~120 片 ≈ 3.6s 流完。"""
    return max(24, (text_len + 119) // 120)


async def _stream_answer(
    emitter, text: str,
    chunk_size: Optional[int] = None,
    interval: float = DEFAULT_ANSWER_CHUNK_INTERVAL,
) -> None:
    """把最终回答按片推送给前端（SSE answer_chunk 事件）。

    - 空/过短回答直接跳过（一次性 result 无感知差异）；
    - 推流只是体验增强，任何失败都不能打断主链路（emitter 已自带 try/except）。
    """
    text = text or ""
    size = chunk_size or _answer_chunk_size(len(text))
    if len(text) <= size:
        return
    for i in range(0, len(text), size):
        emitter("answer_chunk", {"chunk": text[i:i + size], "index": i // size})
        await asyncio.sleep(interval)
    emitter("answer_chunk", {"done": True})


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
                payload = {
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
                }
                payload = await _attach_escalation(
                    payload, req.question, session_id, user_id,
                    on_created=lambda esc: emitter("escalation_created", esc),
                )
                # 思考完成：最终回答分块流式输出（打字机效果），再发完整 result
                await _stream_answer(emitter, payload.get("answer", ""))
                emitter("result", payload)
            else:
                # 轻量澄清门：信息不足直接返回追问（不跑完整 swarm，追问轮
                # 从 30-90s 降到 3-8s）；危机/信息足够/达上限 -> 走完整流程
                from meditriage.swarm.langgraph_swarm import _get_swarm
                _clarify = await _get_swarm().clarify_if_needed(
                    req.question, session_id, user_id,
                )
                if _clarify:
                    payload = {
                        "answer": "", "suggestions": [],
                        "disclaimer": "", "swarm_enabled": False,
                        "agents_involved": [], "session_id": session_id,
                        "total_time": round(_time.time() - t0, 2),
                        "timeout_occurred": False,
                        "clarify": _clarify,
                    }
                    emitter("result", payload)
                    await queue.put(("__END__", {}))
                    return
                result = await process_with_swarm(
                    question=req.question,
                    session_id=session_id,
                    event_emitter=emitter,
                    user_id=user_id,
                )
                # 最终完整结果（带 suggestions、disclaimer）
                payload = {
                    "answer": result.get("answer", ""),
                    "suggestions": result.get("suggestions", []),
                    "disclaimer": result.get("disclaimer", ""),
                    "swarm_enabled": result.get("swarm_enabled", False),
                    "agents_involved": result.get("agents_involved", []),
                    "session_id": session_id,
                    "total_time": result.get("total_time"),
                    "timeout_occurred": result.get("timeout_occurred", False),
                    "risk_level": result.get("risk_level"),
                }
                if result.get("clarify"):
                    # 澄清追问轮：不推流初步分析、不触发转诊（追问不是最终分诊），
                    # 前端直接展示澄清卡片
                    payload["clarify"] = result["clarify"]
                else:
                    await _stream_answer(emitter, payload.get("answer", ""))
                    payload = await _attach_escalation(
                        payload, req.question, session_id, user_id,
                        on_created=lambda esc: emitter("escalation_created", esc),
                    )
                emitter("result", payload)
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


# ---- 转诊闭环（P1-2）：医生端队列 ----
@app.get("/doctor")
async def doctor_page():
    """医生端转诊队列页面（模拟版，无登录；认证在 P2-5）。"""
    return FileResponse(
        Path(__file__).parent / "doctor.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/escalations")
async def list_escalations(status: Optional[str] = None):
    """转诊单列表（医生队列）。

    默认只返回「待处理」（escalated）；?status=all 看全部（含已回复）；
    也可显式传 escalated / doctor_replied / ai_processing 过滤。
    """
    from meditriage.workflows import get_escalation_service
    allowed = {"escalated", "doctor_replied", "ai_processing", "all"}
    if status is None:
        status = "escalated"
    if status not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"status 仅支持: {', '.join(sorted(allowed))}",
        )
    store_status = None if status == "all" else status
    items = await asyncio.to_thread(
        get_escalation_service().store.list, store_status
    )
    return {"items": items, "count": len(items), "status": status}


@app.get("/api/escalations/{esc_id}")
async def get_escalation(esc_id: str):
    """转诊单详情（含结构化交接摘要）。"""
    from meditriage.workflows import get_escalation_service
    row = await asyncio.to_thread(get_escalation_service().store.get, esc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="转诊单不存在")
    return row


@app.post("/api/escalations/{esc_id}/reply")
async def reply_escalation(esc_id: str, body: DoctorReply):
    """医生回复（模拟身份）：escalated -> doctor_replied。"""
    from meditriage.workflows import get_escalation_service
    try:
        row = await asyncio.to_thread(
            get_escalation_service().store.reply, esc_id, body.reply
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return row


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
