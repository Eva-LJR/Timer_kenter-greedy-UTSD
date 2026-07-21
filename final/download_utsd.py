from pathlib import Path

from datasets import load_dataset


# 当前脚本所在的 final 目录
FINAL_DIR = Path(__file__).resolve().parent

# UTSD-1G 专用缓存目录
CACHE_DIR = FINAL_DIR / "dataset_cache" / "UTSD-1G"


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("开始下载或加载 UTSD-1G")
    print(f"缓存目录：{CACHE_DIR}")
    print("=" * 60)

    dataset = load_dataset(
        "thuml/UTSD",
        "UTSD-1G",
        split="train",
        cache_dir=str(CACHE_DIR),
    )

    print("\n下载或加载完成")
    print("样本数量：", len(dataset))
    print("字段：", dataset.column_names)

    if len(dataset) > 0:
        first_sample = dataset[0]
        print("第一个 item_id：", first_sample["item_id"])
        print("第一个 target 长度：", len(first_sample["target"]))

    arrow_files = list(CACHE_DIR.rglob("*.arrow"))

    print("\n找到的 Arrow 文件数量：", len(arrow_files))
    for path in arrow_files[:10]:
        print(path)

    if not arrow_files:
        raise RuntimeError("没有找到 Arrow 文件，请检查下载过程")

    print("\nUTSD-1G 数据准备完成")


if __name__ == "__main__":
    main()