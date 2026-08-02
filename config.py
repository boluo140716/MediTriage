# ============================================================================
# 统一配置入口（Agent 系统的 LLM 接入契约）
#   1. 推理模型 (MediTriage/agent/meditriage/core/llm_client.py) —— 经 vLLM 的 OpenAI 兼容端点
#   2. 评审模型 (Gemini / DeepSeek) —— 评测与检索质量打分（MediTriage/diag/judge_lib.py）
#
# 密钥不写在代码里：从环境变量或 ~/.config/ 下的同名文件读取（如 ~/.config/gemini_api_key）
#   - 容器内（.config 未挂载）：启动命令里 export GEMINI_API_KEY=$(cat ~/.config/gemini_api_key)
#   - 宿主机：直接读 ~/.config/gemini_api_key
#
# ---- 云端 API 运行（重要）------------------------------------------
# LLM_CONFIG 支持通过环境变量把推理底座切到任意 OpenAI 兼容端点，
# 无需修改任何业务代码（Agent / RAG / 记忆 / Web 全部零改动）：
#
#   MEDITRIAGE_LLM_BASE_URL      推理端点 base_url（例如 https://api.deepseek.com
#                               或 https://dashscope.aliyuncs.com/compatible-mode/v1）
#   MEDITRIAGE_LLM_API_KEY       API Key
#   MEDITRIAGE_LLM_MODEL         模型名
#   MEDITRIAGE_LLM_TEMPERATURE   温度（默认 0.7）
#   MEDITRIAGE_LLM_MAX_TOKENS    单次最大输出 token（默认 4096）
#   MEDITRIAGE_LLM_TIMEOUT       请求超时秒数（默认 180）
#
# 云端 API 文本问诊示例（DeepSeek，注意图像问答会优雅降级为提示信息）：
#   export MEDITRIAGE_LLM_BASE_URL=https://api.deepseek.com
#   export MEDITRIAGE_LLM_API_KEY=sk-xxx
#   export MEDITRIAGE_LLM_MODEL=deepseek-chat
#
# 云端 API 且保留影像问答示例（阿里云百炼 Qwen-VL，OpenAI 兼容）：
#   export MEDITRIAGE_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
#   export MEDITRIAGE_LLM_API_KEY=sk-xxx
#   export MEDITRIAGE_LLM_MODEL=qwen-vl-max
#
# 在线嵌入 / 重排（阿里云百炼，本地零模型下载；见 langchain_rag.py 顶部说明）：
#   export MEDITRIAGE_EMBED_PROVIDER=dashscope     # 嵌入走 text-embedding-v4（默认 1024 维）
#   export MEDITRIAGE_RERANK_PROVIDER=dashscope    # 重排走 gte-rerank-v2
#   export MEDITRIAGE_DASHSCOPE_API_KEY=sk-xxx     # 百炼 Key（或放 ~/.config/dashscope_api_key）
#   不设置时嵌入/重排仍走本地 sentence-transformers，行为不变。
## 不设置任何环境变量时，行为与原来完全一致：指向本地 vLLM（medix-r1-8b @ :8000）。
# ============================================================================
import os


def _read_secret(env_var, *paths, default=""):
    """密钥读取优先级：环境变量 > 文件路径列表 > 默认值"""
    if os.environ.get(env_var):
        return os.environ[env_var].strip()
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return f.read().strip()
            except Exception:
                pass
    return default


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_GEMINI_API_KEY = _read_secret(
    "GEMINI_API_KEY",
    "~/.config/gemini_api_key",
)

# ---------------------------------------------------------------------------
# Agent 推理底座：默认本地 vLLM（OpenAI 兼容；serve 脚本见 MediTriage/serving/）。
# 未设置 MEDITRIAGE_LLM_* 环境变量时保持原行为不变；
# 设置后即可切换到云 API / 低显存部署等任意 OpenAI 兼容端点。
# ---------------------------------------------------------------------------
LLM_CONFIG = {
    "api_key": os.environ.get("MEDITRIAGE_LLM_API_KEY", "not-needed"),
    "model_name": os.environ.get("MEDITRIAGE_LLM_MODEL", "medix-r1-8b"),
    "base_url": os.environ.get(
        "MEDITRIAGE_LLM_BASE_URL", "http://localhost:8000/v1"
    ).rstrip("/"),
    "temperature": _env_float("MEDITRIAGE_LLM_TEMPERATURE", 0.7),
    "max_tokens": _env_int("MEDITRIAGE_LLM_MAX_TOKENS", 4096),
    "timeout": _env_float("MEDITRIAGE_LLM_TIMEOUT", 180),
}

# ---------------------------------------------------------------------------
# 视觉专用模型（图片问答走这里；文字对话仍走 LLM_CONFIG）。
# 与主对话模型解耦：主模型可用纯文本（如 DeepSeek），图片自动调视觉模型。
#   MEDITRIAGE_VISION_BASE_URL  默认百炼 OpenAI 兼容端点
#   MEDITRIAGE_VISION_API_KEY   视觉模型 Key（缺省用 MEDITRIAGE_DASHSCOPE_API_KEY）
#   MEDITRIAGE_VISION_MODEL     默认 qwen-vl-max
# ---------------------------------------------------------------------------
VISION_LLM_CONFIG = {
    "api_key": (
        os.environ.get("MEDITRIAGE_VISION_API_KEY")
        or os.environ.get("MEDITRIAGE_DASHSCOPE_API_KEY", "")
    ),
    "model_name": os.environ.get("MEDITRIAGE_VISION_MODEL", "qwen-vl-max"),
    "base_url": os.environ.get(
        "MEDITRIAGE_VISION_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/"),
    "temperature": 0.3,
    "max_tokens": 1024,
    "timeout": 120,
}

# 视觉能力是否可用（未配置 Key 时图片问答优雅降级）
VISION_ENABLED = bool(VISION_LLM_CONFIG.get("api_key"))

# ---------------------------------------------------------------------------
# Benchmark 评审团（双 judge）配置：仅 benchmark_run / judge_lib 使用。
# key 绝不硬编码：env 优先（DEEPSEEK_API_KEY / GEMINI_API_KEY），否则读 ~/.config/。
# base_url / model 与 MediTriage/diag/judge_lib.py 的常量保持一致。
# ---------------------------------------------------------------------------
JUDGES = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat",
                 "api_key": _read_secret("DEEPSEEK_API_KEY", "~/.config/deepseek_api_key")},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash",
               "api_key": _read_secret("GEMINI_API_KEY", "~/.config/gemini_api_key")},
}
