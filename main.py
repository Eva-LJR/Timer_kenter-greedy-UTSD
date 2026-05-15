from config import *
from kcenter_twed import kcenter_greedy_selection
from data_utils import group_sequences_by_label, save_by_label

# ======================================================
# 你只需要在这里导入你的 UTSD 原始数据
# ======================================================
def load_raw_utsd():
    """
    加载你的UTSD数据
    返回：
        data_list: 时间序列列表 [array( L1,1 ), array(L2,1), ...]
        label_list: 标签列表 ["energy", "nature", ...]
    """
    # ========== 你自己替换成你的数据读取代码 ==========
    from extract_data import raw_data_expanded, raw_labels
    return raw_data_expanded, raw_labels

if __name__ == "__main__":
    print("🚀 开始运行 K-Center + TWED 筛选流程")
    
    # 1. 加载数据
    data_list, label_list = load_raw_utsd()
    
    # 2. 按标签分组
    groups = group_sequences_by_label(data_list, label_list)
    
    # 3. 逐个标签筛选 + 保存
    for label, seqs in groups.items():
        print(f"\n📍 处理标签：{label}，样本数：{len(seqs)}")
        selected_idx = kcenter_greedy_selection(
            seqs,
            ratio=SELECTION_RATIO,
            lam=TWED_LAMBDA,
            nu=TWED_NU,
            p=TWED_P,
            seed=SEED
        )
        selected_seqs = [seqs[i] for i in selected_idx]
        save_by_label(label, selected_seqs)

    print("\n🎉 全部完成！输出路径：", OUTPUT_ROOT)