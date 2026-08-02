"""统一资产路径锚点 —— 跨机器 / 跨 checkout 可迁移，均可 env 覆盖。

所有模型 / 数据 / 向量库地址都从这里派生，规则：
  - 仓库根 = 向上首个含 config.py 的目录（不依赖固定层级或机器绝对路径）
  - 每个路径都可被环境变量覆盖（换机器 / 换布局无需改代码）

要改资产位置：改下方 ASSET_ROOT 一处，或设环境变量
MEDITRIAGE_ROOT / MEDITRIAGE_MODELS / MEDITRIAGE_DATA；其余 import 本模块的代码零改动。
"""
import os
from pathlib import Path


def _find_repo_root() -> Path:
    """向上找到含 config.py 的目录 = 仓库根；找不到则退回本文件所在目录的上一级。"""
    for d in Path(__file__).resolve().parents:
        if (d / "config.py").is_file():
            return d
    return Path(__file__).resolve().parent.parent


REPO_ROOT = _find_repo_root()

# 资产根 = MediTriage/（meditriage/paths.py 上溯三级）；env MEDITRIAGE_ROOT 可覆盖。
ASSET_ROOT = Path(
    os.environ.get(
        "MEDITRIAGE_ROOT", Path(__file__).resolve().parent.parent.parent
    )
)

MODELS_DIR = Path(os.environ.get("MEDITRIAGE_MODELS", ASSET_ROOT / "models"))
DATA_DIR = Path(os.environ.get("MEDITRIAGE_DATA", ASSET_ROOT / "data"))

EMBED_MODEL = os.environ.get(
    "MEDITRIAGE_EMBED_MODEL", str(MODELS_DIR / "bge-m3")
)
RERANKER_MODEL = os.environ.get(
    "MEDITRIAGE_RERANKER_MODEL", str(MODELS_DIR / "bge-reranker-v2-m3")
)
MILVUS_URI = os.environ.get("MILVUS_URI", "http://medical-milvus:19530")
# 运行日志（仓库根 log/，跨两组件共享）
LOG_DIR = Path(os.environ.get("MEDITRIAGE_LOG", REPO_ROOT / "log"))
# 运行期缓存（会话总结等，随时可清，不入库）——唯一锚点，绝不用 CWD 相对路径，
# 否则不同进程 CWD 会把同一缓存落到多个目录。
CACHE_DIR = Path(os.environ.get("MEDITRIAGE_CACHE", REPO_ROOT / "会话缓存.tmp"))
