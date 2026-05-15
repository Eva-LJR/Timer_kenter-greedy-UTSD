import os
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
import joblib
from tqdm import tqdm
import pyts
from pyts.metrics import twed  # 用于计算时间扭曲编辑距离

# ====================== 配置项 ======================
CONFIG = {
    "SELECTION_RATIO": 0.1,          # 每个标签筛选比例（10%）
    "MAX_SEQ_LENGTH": 250000,        # 单个npy文件最大长度
    "OUTPUT_ROOT": "UTSD-12G-0_1",   # 输出根目录
    "ENCODER_PATH": "trained_ts_encoder_UTSD.pth",  # 预训练编码器路径
    "SCALER_PATH": "data_minmax_scaler.pkl",        # 归一化器路径
    "FEATURE_SAVE_PATH": "utsd_features.npy",       # 特征缓存路径
    "RANDOM_STATE": 42,              # 随机种子
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # TWED参数（可根据业务调整）
    "TWED_LAMBDA": 0.1,
    "TWED_NU": 0.001,
    "TWED_P": 2
}

# ====================== 1. 模型定义 ======================
class TimeSeriesEncoder(nn.Module):
    """时间序列编码器（与原代码保持一致）"""
    def __init__(self, input_length=512, encoded_dim=64):
        super().__init__()
        self.input_length = input_length
        self.encoded_dim = encoded_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        conv_output_size = 128 * 64
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, encoded_dim)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# ====================== 2. 核心工具函数 ======================
def load_pretrained_components():
    """加载预训练编码器和数据归一化器"""
    # 加载编码器
    model = TimeSeriesEncoder(input_length=512, encoded_dim=64).to(CONFIG["DEVICE"])
    model.load_state_dict(torch.load(CONFIG["ENCODER_PATH"], map_location=CONFIG["DEVICE"]))
    model.eval()
    
    # 加载归一化器
    scaler = joblib.load(CONFIG["SCALER_PATH"])
    
    print("✅ 预训练组件加载完成")
    return model, scaler

def preprocess_data(raw_data, scaler):
    """数据归一化预处理（保持原序列长度）"""
    # 按样本维度归一化（不改变序列长度）
    normalized_data = []
    for seq in raw_data:
        seq_2d = seq.reshape(-1, 1)  # (seq_len, 1)
        normalized_seq = scaler.transform(seq_2d)
        normalized_data.append(normalized_seq)
    return normalized_data

def extract_features(normalized_data, model):
    """提取时间序列特征（适配不等长序列）"""
    if os.path.exists(CONFIG["FEATURE_SAVE_PATH"]):
        features = np.load(CONFIG["FEATURE_SAVE_PATH"])
        print(f"✅ 加载缓存特征: {CONFIG['FEATURE_SAVE_PATH']}")
        return features
    
    # 对每个序列单独提取特征
    features = []
    model.eval()
    with torch.no_grad():
        for seq in tqdm(normalized_data, desc="提取特征"):
            # 适配模型输入（补零到512长度，仅用于特征提取，不影响原始序列）
            seq_padded = np.pad(seq, ((0, max(0, 512 - len(seq))), (0, 0)), mode="constant")
            seq_tensor = torch.tensor(seq_padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(CONFIG["DEVICE"])
            feat = model(seq_tensor).cpu().numpy()[0]
            features.append(feat)
    
    features = np.array(features, dtype=np.float32)
    np.save(CONFIG["FEATURE_SAVE_PATH"], features)
    print(f"✅ 特征提取完成并缓存: {features.shape}")
    return features

def twed_distance_matrix(sequences):
    """计算TWED距离矩阵（适配不等长序列）"""
    n = len(sequences)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    
    # 预计算时间戳（按序列长度生成）
    time_stamps = [np.arange(len(seq)) for seq in sequences]
    
    # 计算两两TWED距离
    for i in tqdm(range(n), desc="计算TWED距离"):
        for j in range(i+1, n):
            dist = twed(
                sequences[i].reshape(-1),
                sequences[j].reshape(-1),
                time_stamps[i],
                time_stamps[j],
                lambda_=CONFIG["TWED_LAMBDA"],
                nu=CONFIG["TWED_NU"],
                p=CONFIG["TWED_P"]
            )
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist  # 对称矩阵
    
    return dist_matrix

def kcenter_greedy_twed(sequences, ratio, random_state=42):
    """基于TWED的K-Center贪心选择（无欧式距离/无padding）"""
    np.random.seed(random_state)
    n = len(sequences)
    budget = max(1, int(n * ratio))  # 至少选1个样本
    
    # 计算TWED距离矩阵
    dist_matrix = twed_distance_matrix(sequences)
    
    # 初始化选择
    selected = []
    unselected = list(range(n))
    
    # 随机选第一个点
    first_idx = np.random.choice(unselected)
    selected.append(first_idx)
    unselected.remove(first_idx)
    
    # 贪心选择剩余点
    for _ in tqdm(range(1, budget), desc="K-Center选择"):
        # 计算未选点到已选点的最小TWED距离
        min_dists = np.min(dist_matrix[unselected][:, selected], axis=1)
        # 选距离最大的点
        max_dist_idx = np.argmax(min_dists)
        best_idx = unselected[max_dist_idx]
        
        selected.append(best_idx)
        unselected.remove(best_idx)
    
    return sorted(selected)

def split_and_save_long_sequence(seq_list, save_dir):
    """将序列拼接成长序列并按最大长度分块保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 拼接所有序列为长序列 (total_len, 1)
    long_seq = np.concatenate(seq_list, axis=0)
    total_len = len(long_seq)
    print(f"📝 拼接后总长度: {total_len}")
    
    # 分块保存
    chunk_idx = 0
    start = 0
    while start < total_len:
        end = min(start + CONFIG["MAX_SEQ_LENGTH"], total_len)
        chunk = long_seq[start:end]
        save_path = os.path.join(save_dir, f"{chunk_idx}.npy")
        np.save(save_path, chunk)
        print(f"✅ 保存分块: {save_path} (长度: {len(chunk)})")
        
        chunk_idx += 1
        start = end

# ====================== 3. 主流程 ======================
def main(raw_data_expanded, raw_labels):
    """主执行流程"""
    # 1. 加载预训练组件
    model, scaler = load_pretrained_components()
    
    # 2. 数据预处理（保持原长度）
    normalized_data = preprocess_data(raw_data_expanded, scaler)
    
    # 3. 按标签分组
    label_to_sequences = {}
    for label, seq in zip(raw_labels, normalized_data):
        if label not in label_to_sequences:
            label_to_sequences[label] = []
        label_to_sequences[label].append(seq)
    print(f"📊 标签数量: {len(label_to_sequences)}, 标签列表: {list(label_to_sequences.keys())}")
    
    # 4. 对每个标签执行筛选和保存
    for label, sequences in label_to_sequences.items():
        print(f"\n========== 处理标签: {label} (样本数: {len(sequences)}) ==========")
        
        # 4.1 基于TWED的K-Center筛选
        selected_indices = kcenter_greedy_twed(sequences, CONFIG["SELECTION_RATIO"])
        selected_sequences = [sequences[idx] for idx in selected_indices]
        print(f"🔍 筛选后样本数: {len(selected_sequences)}")
        
        # 4.2 拼接并分块保存
        save_dir = os.path.join(CONFIG["OUTPUT_ROOT"], label)
        split_and_save_long_sequence(selected_sequences, save_dir)
    
    print(f"\n🎉 所有标签处理完成！输出目录: {CONFIG['OUTPUT_ROOT']}")

# ====================== 4. 执行入口 ======================
if __name__ == "__main__":
    # 假设 raw_data_expanded 和 raw_labels 已从 extract_data 导入
    from extract_data import raw_labels, raw_data_expanded
    
    # 执行主流程
    main(raw_data_expanded, raw_labels)