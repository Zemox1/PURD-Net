import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.visualization import gray_to_magma

# ===================== 核心配置 =====================
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "saved_models/base_2_e100_lr0.0001_bs4_lrd0.9.pth"
IN_PATH = "data/example_dataset"

OUT_ROOT = "infer_depth"
DEPTH_DIR  = os.path.join(OUT_ROOT, "depth_maps")
MAGMA_DIR  = os.path.join(OUT_ROOT, "magma_maps")
RECOVER_DIR = "infer_recover"

OUT_ROOT0 = "infer_depth0"
DEPTH0_DIR  = os.path.join(OUT_ROOT0, "depth_maps")
MAGMA0_DIR  = os.path.join(OUT_ROOT0, "magma_maps")

START_NUM = 1100
END_NUM   = 1315

# ===================== 模型加载 =====================
model = UDFNet(n_bins=100).to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()
print(f"模型加载完成，使用设备: {DEVICE}")

# ===================== 创建输出目录 =====================
os.makedirs(DEPTH_DIR, exist_ok=True)
os.makedirs(MAGMA_DIR, exist_ok=True)
os.makedirs(RECOVER_DIR, exist_ok=True)
os.makedirs(DEPTH0_DIR, exist_ok=True)   # ★ 创建 depth0 输出目录
os.makedirs(MAGMA0_DIR, exist_ok=True)   # ★ 创建 depth0 输出目录
print(f"深度图输出目录：{DEPTH_DIR}")
print(f"Magma热力图输出目录：{MAGMA_DIR}")
print(f"Recover图输出目录：{RECOVER_DIR}")
print(f"Depth0图输出目录：{DEPTH0_DIR}")

# ===================== 工具函数 =====================
def load_img(path):
    """加载图片并转为 [1,3,H,W] Tensor"""
    try:
        img = Image.open(path).convert("RGB")
        tensor = transforms.ToTensor()(img).unsqueeze(0)
        return tensor.to(DEVICE)
    except Exception as e:
        print(f"加载图片失败 {path}: {e}")
        return None

# ===================== 批量推理 =====================
for num in range(START_NUM, END_NUM + 1):
    image_number = f"{num:05d}"

    img_paths = [
        f"{IN_PATH}/I0/{image_number}.png",
        f"{IN_PATH}/I45/{image_number}.png",
        f"{IN_PATH}/I90/{image_number}.png",
        f"{IN_PATH}/I135/{image_number}.png"
    ]

    missing = [p for p in img_paths if not os.path.exists(p)]
    if missing:
        print(f"跳过 {image_number}：缺失图片 {missing}")
        continue

    inputs = []
    for p in img_paths:
        img = load_img(p)
        if img is None:
            break
        img = torch.nn.functional.interpolate(
            img, size=(480, 640),
            mode='bilinear', align_corners=False
        )
        inputs.append(img)

    if len(inputs) != 4:
        print(f"跳过 {image_number}：图片加载不完整")
        continue

    # ===================== 推理 =====================
    with torch.no_grad():
        pred, bin_edges, recover, a_out, _ = model(*inputs)
        magma = gray_to_magma(pred, device=DEVICE)
        # ★ 为 depth0 也创建 magma 可视化
       

    # ===================== 保存结果 =====================
    depth_path   = os.path.join(DEPTH_DIR,  f"{image_number}.png")
    magma_path   = os.path.join(MAGMA_DIR,  f"{image_number}.png")
    recover_path = os.path.join(RECOVER_DIR, f"{image_number}.png")
   
  

    save_image(pred, depth_path, normalize=False)
    save_image(magma, magma_path)
    save_image(recover, recover_path)


    print(
        f"{image_number} 完成 | "
        f"Depth → {depth_path} | "
        f"Magma → {magma_path} | "
        f"Recover → {recover_path} | "

    )

print("\n批量处理完成！")
print(f"Depth 目录    ：{DEPTH_DIR}")
print(f"Magma 目录    ：{MAGMA_DIR}")
print(f"Recover 目录  ：{RECOVER_DIR}")
