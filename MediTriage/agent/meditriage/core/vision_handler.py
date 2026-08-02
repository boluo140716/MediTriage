"""图像输入链路。

医学图像由 MediX-R1-8B（VLM）通过 vLLM OpenAI Vision 格式直接处理。
不走 Milvus（Milvus 只存文本证据），也不走多 Agent swarm（图像 VQA 是单轮推理）。

用法：
  from meditriage.core.vision_handler import process_image_query
  result = await process_image_query(
      "data/med_image_samples/.../x.png", "这张影像有什么异常？")
"""
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from meditriage.core.llm_client import LLMClient

# 向上找到含 config.py 的仓库根（与 llm_client.py 同一机制）；Path 已在文件顶部导入
import sys
for _d in Path(__file__).resolve().parents:
    if (_d / "config.py").is_file():
        sys.path.insert(0, str(_d))
        break
from config import VISION_ENABLED, VISION_LLM_CONFIG

MEDICAL_VISION_SYSTEM_PROMPT = (
    "You are MediX-R1, a medical vision-language assistant. "
    "Carefully analyze the provided medical image (radiology, pathology, or clinical photo) "
    "and answer the user's question. Be precise, clinically grounded, and note any visible "
    "abnormalities. If the image quality or content is insufficient to answer confidently, say so."
)


def _to_file_uri(image_path: str) -> str:
    """本地路径转 file:// URI；http(s)/data 原样返回。

    file:// 形式需 vLLM 启用 --allowed-local-media-path 才能访问。
    """
    if image_path.startswith(("http://", "https://", "file://", "data:")):
        return image_path
    p = Path(image_path)
    if not p.is_absolute():
        p = p.resolve()
    return p.as_uri()


async def process_image_query(
    image_path: str,
    question: str,
    session_id: Optional[str] = None,
    short_term_memory: Optional[Any] = None,
    event_emitter: Optional[Any] = None,
) -> Dict[str, Any]:
    """处理图像问答（VLM 直接推理）。

    Args:
        image_path: 本地图片路径或 URL。
        question: 文本问题。
        session_id: 会话 ID（用于记忆）。
        short_term_memory: 短期记忆管理器（可选）。
        event_emitter: 事件发射器 callable(type, data)（前端可视化用）。

    Returns:
        {"answer", "image_path", "question", "disclaimer"}。
    """
    def _emit(etype, data):
        if event_emitter:
            try:
                event_emitter(etype, data)
            except Exception:
                pass

    image_uri = _to_file_uri(image_path)
    _emit("vision_started", {"image_path": image_path, "question": question})
    logger.info(f"Vision query: {image_path} | Q: {question[:60]}")

    # 视觉模型未配置（如未设置百炼 Key）时优雅降级：返回提示信息而不是报错。
    if not VISION_ENABLED:
        msg = (
            "未配置视觉模型（缺少 MEDITRIAGE_VISION_API_KEY 或 MEDITRIAGE_DASHSCOPE_API_KEY），"
            "无法分析图像。请配置百炼视觉模型 Key，或直接以文字描述症状继续问诊。"
        )
        logger.warning(msg)
        _emit("error", {"message": msg})
        return {
            "answer": msg,
            "image_path": image_path,
            "question": question,
            "error": True,
            "disclaimer": "影像分析仅供参考，最终诊断须由专业放射科/临床医生结合完整资料判断。",
        }

    # 拉取近期对话上下文：让 VLM 能把图像与之前讨论（如"发烧"）关联起来。
    context_text = ""
    if short_term_memory and session_id:
        try:
            recent = short_term_memory.get_history(session_id, limit=4)
            lines = []
            for m in (recent or [])[-8:]:
                role = "用户" if m.get("role") == "user" else "助手"
                c = (m.get("content") or "").strip()[:300]
                if c:
                    lines.append(f"{role}：{c}")
            if lines:
                context_text = (
                    "【近期对话上下文（供参考，用户的提问可能指代其中内容）】\n"
                    + "\n".join(lines)
                    + "\n\n"
                )
        except Exception:
            pass

    user_text = (
        f"{context_text}用户的提问：{question}" if context_text else question
    )

    client = LLMClient(config=VISION_LLM_CONFIG)
    messages = [
        {"role": "system", "content": MEDICAL_VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ],
        },
    ]

    try:
        answer = await client.chat(messages, max_tokens=1024)
    except Exception as e:
        logger.error(f"Vision query failed: {e}")
        _emit("error", {"message": str(e)})
        return {
            "answer": f"图像分析失败：{e}",
            "image_path": image_path,
            "question": question,
            "error": True,
        }

    # 视觉链路不走 AgentLoop，需自行补发 llm_response 上报 token 用量，
    # 否则前端 ctx 徽标在影像轮（含图片指代复用轮）一直停在空态
    usage = getattr(client, "last_usage", None)
    if usage is not None:
        _emit("llm_response", {
            "finish_reason": "stop",
            "has_tool_calls": False,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        })

    # 记录到短期记忆（图像问答也是对话的一部分）。
    if short_term_memory and session_id:
        try:
            short_term_memory.add_message(
                session_id=session_id, role="user",
                content=f"[图像] {question}")
            short_term_memory.add_message(
                session_id=session_id, role="assistant", content=answer)
        except Exception:
            pass

    _emit("vision_answer", {"answer_preview": answer[:300]})
    _emit("session_completed", {"mode": "vision", "session_id": session_id})

    return {
        "answer": answer,
        "image_path": image_path,
        "question": question,
        "disclaimer": "影像分析仅供参考，最终诊断须由专业放射科/临床医生结合完整资料判断。",
    }
