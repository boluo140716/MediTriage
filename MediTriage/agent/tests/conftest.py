"""pytest 共享配置。

把 MediTriage/agent 根目录加入 sys.path，使 tests/ 下的用例能直接
`from core ... / from swarm ... / from knowledge ...`（与各用例自带的
sys.path.insert 等价，双保险：pytest 跑 & 直接 python 跑都可用）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
