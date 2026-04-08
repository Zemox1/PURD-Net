import os
import time
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# 请根据你的实际导入路径调整
from depth_estimation.model.model import UDFNet
from depth_estimation.utils.visualization import gray_to_magma

# ===================== 配置 =====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "saved_models/base_2_e100_lr0.0001_bs4_lrd0.9.pth"
DATA_DIR = "/data/wrj/data/12.25_tju/1"

OUT_ROOT = "experiment_results"
DEPTH_DIR = os.path.join(OUT_ROOT, "depth")
MAGMA_DIR = os.path.join(OUT_ROOT, "magma")
RECOVER_DIR = os.path.join(OUT_ROOT, "recover")

IMG_SIZE = (480, 640)   # (H, W)
WARMUP_ITERS = 10

# ===================== 模型加载 =====================
model = UDFNet(n_bins=100).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

print(f"模型加载完成，使用设备: {DEVICE}")

# ===================== 创建输出目录 =====================
os.makedirs(DEPTH_DIR, exist_ok=True)
os.makedirs(MAGMA_DIR, exist_ok=True)
os.makedirs(RECOVER_DIR, exist_ok=True)

print(f"数据目录: {DATA_DIR}")
print(f"输出目录: {OUT_ROOT}")


def synchronize():
    if DEVICE == 'cuda':
        torch.cuda.synchronize()


def load_image(angle, img_name):
    img_path = os.path.join(DATA_DIR, str(angle), img_name)
    if not os.path.exists(img_path):
        return None

    img = Image.open(img_path).convert("RGB")
    img = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    img = torch.nn.functional.interpolate(
        img, size=IMG_SIZE, mode='bilinear', align_corners=False
    )
    return img


def prepare_inputs(img_name):
    inputs = []
    for angle in [0, 45, 90, 135]:
        img = load_image(angle, img_name)
        if img is None:
            print(f"跳过 {img_name}，缺少 {angle} 度图像")
            return None
        inputs.append(img)
    return inputs


def inference_once(inputs):
    with torch.no_grad():
        pred, _, recover, _, _ = model(*inputs)
        magma = gray_to_magma(pred, device=DEVICE)
    return pred, magma, recover


def save_outputs(img_name, pred, magma, recover):
    base_name = os.path.splitext(img_name)[0]

    save_image(pred, os.path.join(DEPTH_DIR, f"{base_name}_depth.png"), normalize=False)
    save_image(magma, os.path.join(MAGMA_DIR, f"{base_name}_magma.png"))
    save_image(recover, os.path.join(RECOVER_DIR, f"{base_name}_recover.png"))


def warmup(image_files):
    print(f"\n开始预热 {min(WARMUP_ITERS, len(image_files))} 次...")
    count = 0
    for img_name in image_files:
        inputs = prepare_inputs(img_name)
        if inputs is None:
            continue
        _ = inference_once(inputs)
        count += 1
        if count >= WARMUP_ITERS:
            break
    synchronize()
    print("预热完成\n")


def main():
    zero_dir = os.path.join(DATA_DIR, "0")
    if not os.path.exists(zero_dir):
        print(f"错误：找不到文件夹 {zero_dir}")
        return

    image_files = [
        f for f in os.listdir(zero_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    image_files.sort()

    print(f"找到 {len(image_files)} 张图像")

    if len(image_files) == 0:
        print("没有可处理图像")
        return

    warmup(image_files)

    processed = 0
    total_infer_time = 0.0
    total_full_time = 0.0

    for img_name in image_files:
        # ===== 全流程计时开始 =====
        full_start = time.perf_counter()

        inputs = prepare_inputs(img_name)
        if inputs is None:
            continue

        # ===== 纯推理计时开始 =====
        synchronize()
        infer_start = time.perf_counter()

        pred, magma, recover = inference_once(inputs)

        synchronize()
        infer_end = time.perf_counter()
        # ===== 纯推理计时结束 =====

        save_outputs(img_name, pred, magma, recover)

        full_end = time.perf_counter()
        # ===== 全流程计时结束 =====

        infer_time = infer_end - infer_start
        full_time = full_end - full_start

        total_infer_time += infer_time
        total_full_time += full_time
        processed += 1

        print(f"{img_name} | 推理: {infer_time*1000:.2f} ms | 全流程: {full_time*1000:.2f} ms")

    print("\n处理完成")
    print(f"成功处理: {processed}/{len(image_files)}")
    print(f"结果保存在: {OUT_ROOT}")
    print(f"  深度图: {DEPTH_DIR}")
    print(f"  Magma图: {MAGMA_DIR}")
    print(f"  恢复图: {RECOVER_DIR}")

    if processed > 0:
        avg_infer = total_infer_time / processed
        avg_full = total_full_time / processed

        infer_fps = 1.0 / avg_infer if avg_infer > 0 else 0.0
        full_fps = 1.0 / avg_full if avg_full > 0 else 0.0

        print("\n================ 速度统计 ================")
        print(f"设备: {DEVICE}")
        print(f"测试图像数: {processed}")

        print("\n[1] 纯推理速度")
        print(f"总推理时间: {total_infer_time:.4f} s")
        print(f"平均推理时间: {avg_infer * 1000:.2f} ms/image")
        print(f"吞吐率: {infer_fps:.2f} FPS")

        print("\n[2] 全流程速度")
        print(f"总全流程时间: {total_full_time:.4f} s")
        print(f"平均全流程时间: {avg_full * 1000:.2f} ms/image")
        print(f"吞吐率: {full_fps:.2f} FPS")
        print("==========================================")


if __name__ == "__main__":
    main()