import os

# ==================== 项目基础配置 ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "UTSD-12G-0_1")  # 输出目录

# ==================== 筛选参数 ====================
SELECTION_RATIO = 0.1    # 10% 数据
MAX_CHUNK_LEN = 250000   # 每个 .npy 最大长度

# ==================== TWED 距离参数 ====================
TWED_LAMBDA = 0.1
TWED_NU = 0.001
TWED_P = 2

# ==================== 随机种子 ====================
SEED = 42