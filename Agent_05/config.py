
"""
文件职责：
统一管理模型配置和两个 SQLite 数据库路径。

长期记忆：
    long_term_memory.sqlite

短期记忆：
    short_term_checkpoint.sqlite

注意：
    这里只读取配置，不负责创建 Store 和 Checkpointer。
"""

import os
from pathlib import Path


# =====================================================
# 一、确定项目根目录
# =====================================================

# 获取当前文件所在目录。
# 使用绝对路径，避免从其他工作目录启动程序时
# 找不到 .env 文件。

PROJECT_ROOT = Path(__file__).resolve().parent

# =====================================================
# 二、读取长期记忆数据库路径
# =====================================================

LONG_TERM_DB_PATH = Path(
    os.getenv(
        "LONG_TERM_DB_PATH",
        "./data/long_term_memory.sqlite"
    )
)

# 将相对路径转换成项目根目录下的绝对路径。
if not LONG_TERM_DB_PATH.is_absolute():
    LONG_TERM_DB_PATH = PROJECT_ROOT / LONG_TERM_DB_PATH


# =====================================================
# 三、读取短期记忆数据库路径
# =====================================================

CHECKPOINT_DB_PATH = Path(
    os.getenv(
        "CHECKPOINT_DB_PATH",
        "./data/short_term_checkpoint.sqlite"
    )
)

if not CHECKPOINT_DB_PATH.is_absolute():
    CHECKPOINT_DB_PATH = PROJECT_ROOT / CHECKPOINT_DB_PATH


# =====================================================
# 四、创建数据库所在目录
# =====================================================

# 两个数据库都放在 data 目录中。
# 如果目录不存在，就自动创建。

LONG_TERM_DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)
