from pathlib import Path
import csv

root      = Path(__file__).parent
I0_dir    = root / "I0"
I45_dir   = root / "I45"
I90_dir   = root / "I90"
I135_dir  = root / "I135"
depth_dir = root / "depth"
J_dir     = root / "J"
t_dir     = root / "t"
A_dir     = root / "A"

# 获取并排序
I0_files   = sorted(I0_dir.iterdir())
I45_files  = sorted(I45_dir.iterdir())
I90_files  = sorted(I90_dir.iterdir())
I135_files = sorted(I135_dir.iterdir())
depth_files = sorted(depth_dir.iterdir())
J_files     = sorted(J_dir.iterdir())
t_files     = sorted(t_dir.iterdir())
A_files     = sorted(A_dir.iterdir())

n_total = len(I0_files)
n_train = int(n_total * 0.8)          # 1052
n_test  = n_total - n_train           # 263

# 数量校验
assert len(I0_files) == len(I45_files) == len(I90_files) == len(I135_files) == len(depth_files) == len(J_files) == len(t_files), \
    "偏振图与深度图数量不一致！"

def write_csv(slice_range, csv_name):
    with open(root / csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        for idx in slice_range:
            i0, i45, i90, i135, d, j, t,A = (
                I0_files[idx], I45_files[idx], I90_files[idx],
                I135_files[idx], depth_files[idx], J_files[idx], t_files[idx],A_files[idx]
            )
            writer.writerow([
                f"data/example_dataset/I0/{i0.name}",
                f"data/example_dataset/I45/{i45.name}",
                f"data/example_dataset/I90/{i90.name}",
                f"data/example_dataset/I135/{i135.name}",
                f"data/example_dataset/depth/{d.name}",
                f"data/example_dataset/J/{j.name}",
                f"data/example_dataset/t/{t.name}",
                f"data/example_dataset/A/{A.name}"
            ])
    print(f"{csv_name} 写入完成，共 {len(slice_range)} 条。")

# 8 成训练集
write_csv(range(n_train), "dataset.csv")
# 2 成测试集
write_csv(range(n_train, n_total), "test_dataset.csv")