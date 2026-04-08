import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.visualization import gray_to_inferno

# 配置
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "saved_models/I+tbase_2_e100_lr0.0001_bs8_lrd0.9.pth"  # 模型路径
IN_PATH =  "data/2dataset"
OUT_PATH = "data/2test0001"  # 输出目录
# 4个角度的三通道图片路径
IMG_PATHS = [
    f"{IN_PATH}/I0/00001.png",
    f"{IN_PATH}/I45/00001.png",
    f"{IN_PATH}/I90/00001.png",
    f"{IN_PATH}/I135/00001.png"
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

# 保存输入图片（使用角度作为文件名）
angles = ["I0", "I45", "I90", "I135"]
for i, (img_path, angle) in enumerate(zip(IMG_PATHS, angles)):
    img = Image.open(img_path).convert("RGB")
    img.save(os.path.join(OUT_PATH, f"{angle}.png"))
    print(f"保存输入图片: {angle}.png")

# 推理
with torch.no_grad():
    # 加载4个三通道输入（每个都是[1, 3, H, W]）
    inputs = [load_img(p) for p in IMG_PATHS]
    # 模型预测（直接传入原始三通道输入）
    # 统一 resize
    inputs = [torch.nn.functional.interpolate(
        im, size=(480, 640), mode='bilinear', align_corners=False)
        for im in inputs]
    pred, _, recover, j_out,t_out,a_out = model(*inputs)
    # 生成热力图
    inferno = gray_to_inferno(pred, device=DEVICE)

# 保存结果
save_image(inferno, os.path.join(OUT_PATH, "result_inferno.png"))
save_image(recover, os.path.join(OUT_PATH, "result_recover.png"))
save_image(pred, os.path.join(OUT_PATH, "result_depth.png"))
save_image(t_out, os.path.join(OUT_PATH, "result_t.png"))
save_image(a_out, os.path.join(OUT_PATH, "result_a.png"))
save_image(j_out, os.path.join(OUT_PATH, "result_j.png"))
print("测试完成，结果已保存到", OUT_PATH)