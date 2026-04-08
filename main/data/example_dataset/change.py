from pathlib import Path
from PIL import Image

src_dir = Path(r"F:\encoder -2decoder\uw_depth-main\data\example_dataset\t")  # 改成你的目录

# 自然排序（按数字大小）
def nat_key(p: Path):
    return int(''.join(filter(str.isdigit, p.stem)))

files = sorted(src_dir.iterdir(), key=nat_key)

# 目标尺寸：宽度640，高度480
target_size = (640, 480)

for idx, old_path in enumerate(files):
    new_name = f"{idx:05d}.png"
    new_path = old_path.with_name(new_name)

    # 打开图片并调整尺寸
    with Image.open(old_path) as img:
        # 使用LANCZOS滤镜进行高质量缩放
        resized_img = img.resize(target_size, Image.LANCZOS)
        # resized_img = resized_img.mean(dim=0, keepdim=True)
        # 保存为PNG格式
        resized_img.save(new_path, format="png")

        old_path.unlink()

print(f"完成！共 {len(files)} 个文件 → 全部转换为640x480的PNG格式")
