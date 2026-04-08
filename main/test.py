import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.visualization import gray_to_inferno

# 配置
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "/data/wrj/0.5-2I+t-refine_d0/saved_models/refine_2_e100_lr0.0001_bs4_lrd0.9.pth" # 模型路径
OUT_PATH = "data/test"  # 输出目录
# 4个角度的三通道图片路径
IMG_PATHS = [
    f"{OUT_PATH}/I0.png",
    f"{OUT_PATH}/I45.png",
    f"{OUT_PATH}/I90.png",
    f"{OUT_PATH}/I135.png"
]

# 创建输出目录
os.makedirs(OUT_PATH, exist_ok=True)

# 加载三通道图片（直接保留原始3通道，不做灰度转换）
def load_img(path):
    # 1. 以RGB模式加载（确保保留3通道，避免RGBA等格式干扰）
    img = Image.open(path).convert("RGB")  # 关键：明确加载为3通道RGB
    # 2. 转为Tensor（形状：[3, H, W]），并增加batch维度（[1, 3, H, W]）
    tensor = transforms.ToTensor()(img).unsqueeze(0)  # 保持3通道
    return tensor.to(DEVICE)

# 加载模型（修复警告）
model = UDFNet(n_bins=100).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# 推理
with torch.no_grad():
    # 加载4个三通道输入（每个都是[1, 3, H, W]）
    inputs = [load_img(p) for p in IMG_PATHS]
    # 模型预测（直接传入原始三通道输入）
    # 统一 resize
    inputs = [torch.nn.functional.interpolate(
        im, size=(480, 640), mode='bilinear', align_corners=False)
        for im in inputs]
    pred, bin_edges, recover, a_out, depth0 = model(*inputs)

    # 生成热力图
    inferno = gray_to_inferno(pred, device=DEVICE)

# 保存结果
save_image(inferno, os.path.join(OUT_PATH, "result_inferno.png"))
save_image(recover, os.path.join(OUT_PATH, "result_recover.png"))
save_image(pred, os.path.join(OUT_PATH, "result_depth.png"))
save_image(depth0, os.path.join(OUT_PATH, "result_depth0.png"))
save_image(a_out, os.path.join(OUT_PATH, "result_a.png"))

print(f"图片 处理完成，结果已保存到 {OUT_PATH}")