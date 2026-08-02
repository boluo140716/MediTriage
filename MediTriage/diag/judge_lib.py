"""评审团库：双 judge（DeepSeek + Gemini）+ 统计工具 + 留痕。

接口契约（benchmark_run 依赖以下名字与签名）：
  - 仅用标准库 urllib，不引第三方 SDK；POST /chat/completions（OpenAI 兼容）。
  - judge 调用一律 temperature=0（可复现）；3 次重试 + 指数退避 + 超时 60s。
  - 质量两两对比内部随机交换 A/B 顺序再还原（抗位置偏置），
    prompt 显式声明"忽略答案长度与文风"（抗冗长偏置）。
  - 每次 judge 调用把原始 prompt/response append 到审计文件，便于事后复核。

密钥不写在代码里：从环境变量 DEEPSEEK_API_KEY / GEMINI_API_KEY 读取。
"""
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request

# --- 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行 ---
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / 'agent' / 'meditriage' / 'paths.py').is_file()
)
for _p in (str(_ASK / 'agent'), str(_Path(__file__).resolve().parent)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# --- end 引导 ---

# --- 后端常量 ---
DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
# 选 gemini-2.5-flash：在该 OpenAI 兼容端点稳定返回内容；gemini-2.0-flash
# 已对新用户下线，gemini-flash-latest 在该端点不返回 content，均不可用。
GEMINI_MODEL = "gemini-2.5-flash"

# 审计文件（log/ 已 gitignore）；目录需 mkdir
AUDIT_PATH = str(_paths.LOG_DIR / "benchmark/judge_raw.jsonl")


# ===========================================================================
# 统计工具
# ===========================================================================
def wilson_ci(k, n, z=1.96):
    """Wilson score 置信区间（二项比例）。返回 (lo, hi)。n==0 时返回 (0.0, 0.0)。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (center - half, center + half)


def cohen_kappa(labels_a, labels_b):
    """Cohen's κ：两位评分者对同一批样本的类别标注一致性（含偶然一致校正）。

    labels_a/labels_b 等长，元素可为任意可哈希类别（如选项字母 'A'..'E'）。
    完全一致返回 1.0；按偶然水平一致返回 ~0；空输入返回 0.0。
    """
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return 0.0
    cats = set(labels_a) | set(labels_b)
    # 观察一致率
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    # 期望（偶然）一致率
    pe = 0.0
    for c in cats:
        pa = sum(1 for a in labels_a if a == c) / n
        pb = sum(1 for b in labels_b if b == c) / n
        pe += pa * pb
    if pe >= 1.0:  # 两评分者都恒定且相同 → 完全一致
        return 1.0
    return (po - pe) / (1.0 - pe)


def extract_score(text):
    """从 judge 文本里抓 1–5 分。优先级：
        1) JSON 里的 "score": N
        2) N/5 形式
        3) 文本中首个 1–5 的孤立数字
    抓不到返回 0。
    """
    if not text:
        return 0
    m = re.search(r'"score"\s*:\s*([1-5])\b', text)
    if m:
        return int(m.group(1))
    m = re.search(r'\b([1-5])\s*/\s*5\b', text)
    if m:
        return int(m.group(1))
    m = re.search(r'\b([1-5])\b', text)
    if m:
        return int(m.group(1))
    return 0


# ===========================================================================
# 留痕
# ===========================================================================
def _audit(judge, prompt, raw_response):
    """把一次 judge 调用 append 到审计文件（不用真实时间戳，省略以保持可复现）。"""
    try:
        os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)
        rec = {
            "ts_omitted": None,
            "judge": judge,
            "prompt": prompt,
            "raw_response": raw_response,
        }
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # 审计失败不应阻断主流程
        pass


# ===========================================================================
# HTTP 后端
# ===========================================================================
def _post_openai(base_url, api_key, model, messages,
                 temperature=0, max_tokens=512):
    """urllib POST {base}/chat/completions（OpenAI 兼容）。
    3 次重试 + 指数退避 + 超时 60s。返回 message.content 字符串。
    """
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + (api_key or ""),
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=body, headers=headers,
                                         method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            # content 可能为 None：thinking 模型把 token 预算耗在推理上、
            # 或 finish_reason=length 被截断。统一回成字符串，避免 KeyError。
            return data["choices"][0]["message"].get("content") or ""
        except Exception as e:  # noqa: BLE001 网络/解析异常都重试
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1s, 2s 指数退避
    raise RuntimeError(f"_post_openai failed after 3 attempts: {last_err}")


def judge_deepseek(messages, **kw):
    """DeepSeek judge：base deepseek，model deepseek-chat，env DEEPSEEK_API_KEY。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    out = _post_openai(DEEPSEEK_BASE, key, DEEPSEEK_MODEL, messages, **kw)
    _audit("deepseek", messages, out)
    return out


def judge_gemini(messages, **kw):
    """Gemini judge：base gemini-openai，model gemini-2.5-flash，env GEMINI_API_KEY。"""
    key = os.environ.get("GEMINI_API_KEY", "")
    out = _post_openai(GEMINI_BASE, key, GEMINI_MODEL, messages, **kw)
    _audit("gemini", messages, out)
    return out


# ===========================================================================
# judge 用法（评判"系统输出对不对" / 质量两两对比）
# ===========================================================================
def _parse_json_loose(text):
    """从可能含 ```json 围栏或前后噪声的文本里抠出第一个 JSON 对象。失败返回 {}。"""
    if not text:
        return {}
    # 直接尝试
    try:
        return json.loads(text)
    except Exception:
        pass
    # 抠出第一个 {...} 块
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def _format_options(options):
    return "\n".join(f"{k}. {v}" for k, v in options.items())


def judge_mcq(judge_fn, question, options, model_answer):
    """让 judge 判断 model_answer 是否对应正确选项（用于评"系统输出对不对"）。

    judge 不预先知道标准答案，需自己从题面与选项推断正确项，再与 model_answer 比对。
    返回 bool。解析失败时退化为 False（保守）。
    """
    opts_text = _format_options(options)
    prompt = (
        "你是严格的医学考试评分助手。下面是一道单选题、其选项，以及考生给出的答案。\n"
        "请你独立判断正确选项，再判断考生答案是否正确。\n"
        "只输出 JSON：{\"correct_letter\":\"<你判断的正确选项字母>\","
        "\"is_match\":true/false}（is_match 表示考生答案是否正确）。\n\n"
        f"【题目】{question}\n【选项】\n{opts_text}\n【考生答案】{model_answer}"
    )
    messages = [{"role": "user", "content": prompt}]
    # 预算给足：thinking 模型先耗隐藏推理 token，太小会被截断成空 content。
    raw = judge_fn(messages, max_tokens=1024)
    d = _parse_json_loose(raw)
    if "is_match" in d:
        return bool(d["is_match"])
    # 兜底：比较 judge 判出的正确字母与考生答案
    cl = str(d.get("correct_letter", "")).strip().upper()[:1]
    ma = str(model_answer).strip().upper()[:1]
    return bool(cl) and cl == ma


def judge_choose_letter(judge_fn, question, options):
    """让 judge 从 options 选一个正确答案字母（用于 judge 自身校准）。返回大写字母或 ''。"""
    opts_text = _format_options(options)
    valid = "".join(options.keys())
    prompt = (
        "你是资深临床医生，正在做单项选择题。请从给定选项中选出唯一正确答案。\n"
        f"只输出一个选项字母（{'/'.join(options.keys())}），不要任何解释或多余字符。\n\n"
        f"【题目】{question}\n【选项】\n{opts_text}"
    )
    messages = [{"role": "user", "content": prompt}]
    # 预算给足：gemini-2.5-flash 是 thinking 模型，会先消耗隐藏推理 token，
    # 太小（如 16）会 finish_reason=length 且 content 为空。1024 留足余量。
    raw = judge_fn(messages, max_tokens=1024)
    # 抓首个落在合法选项内的字母
    for ch in (raw or ""):
        u = ch.upper()
        if u in valid:
            return u
    return ""


def judge_quality_pairwise(judge_fn, question, ans_a, ans_b, reference=""):
    """质量两两对比（准确性/完整性/安全性）。

    抗位置偏置：内部随机交换 A/B 喂给 judge，拿到结果后还原成调用方的 A/B。
    抗冗长偏置：prompt 显式声明"忽略答案长度与文风，只看医学正确/完整/安全"。
    返回还原顺序后的 dict：{"winner":"A|B|tie","score_a":int,"score_b":int,"reason":str}。
    """
    swap = random.random() < 0.5  # 随机决定是否交换
    first, second = (ans_b, ans_a) if swap else (ans_a, ans_b)

    ref_block = f"\n【参考答案（仅供评分参考）】{reference}\n" if reference else "\n"
    prompt = (
        "你是严格、公正的医学答案评审。给定一个医学问题与两份候选回答 A、B。\n"
        "评分维度：准确性（医学事实是否正确）、完整性（要点是否齐全）、"
        "安全性（是否避免有害/误导建议、是否给出必要的就医提示）。\n"
        "重要：请忽略答案的长度与文风，不要因为某个回答更长/更啰嗦/辞藻更华丽就给高分；"
        "只看医学上的正确、完整与安全。\n"
        "请分别给 A、B 打 1–5 分，并判定胜者。\n"
        "只输出 JSON：{\"winner\":\"A\"或\"B\"或\"tie\","
        "\"score_a\":1-5,\"score_b\":1-5,\"reason\":\"简要理由\"}。\n"
        f"\n【问题】{question}{ref_block}"
        f"【回答 A】\n{first}\n\n【回答 B】\n{second}"
    )
    messages = [{"role": "user", "content": prompt}]
    raw = judge_fn(messages, max_tokens=512)
    d = _parse_json_loose(raw)

    winner = str(d.get("winner", "tie")).strip().upper()
    if winner not in ("A", "B", "TIE"):
        winner = "TIE"
    try:
        sa = int(d.get("score_a", 0))
    except (TypeError, ValueError):
        sa = 0
    try:
        sb = int(d.get("score_b", 0))
    except (TypeError, ValueError):
        sb = 0
    reason = str(d.get("reason", ""))

    if swap:
        # judge 眼里的 A 实为调用方的 B → 全部还原
        sa, sb = sb, sa
        if winner == "A":
            winner = "B"
        elif winner == "B":
            winner = "A"

    return {
        "winner": "tie" if winner == "TIE" else winner,
        "score_a": sa,
        "score_b": sb,
        "reason": reason,
    }
