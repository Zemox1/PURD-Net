import os
import cv2
import numpy as np

pred_dir = "/data/wrj/results/1-2_d0/infer_recover/"
gt_dir   = "/data/wrj/results/1-2_d0/data/example_dataset/J"

valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_img(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in valid_exts


def load_rgb(path: str) -> np.ndarray:
    # cv2 is BGR, convert to RGB for consistency (not necessary for metrics but keeps intuition)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def ssim_rgb(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Fast SSIM (Wang et al.) using cv2.GaussianBlur.
    Compute SSIM per-channel and average.
    """
    K1, K2 = 0.01, 0.03
    L = 255.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    # OpenCV GaussianBlur expects float32/float64 OK
    def ssim_single(x, y):
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

        num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = num / (den + 1e-12)
        return float(np.mean(ssim_map))

    # Split channels
    ssim_vals = []
    for c in range(3):
        ssim_vals.append(ssim_single(img1[:, :, c], img2[:, :, c]))
    return float(np.mean(ssim_vals))


pred_files = sorted([f for f in os.listdir(pred_dir) if is_img(f)])
gt_set = set([f for f in os.listdir(gt_dir) if is_img(f)])

psnr_list, ssim_list = [], []
count = 0

for name in pred_files:
    if name not in gt_set:
        print("skip(not in gt):", name)
        continue

    p_path = os.path.join(pred_dir, name)
    g_path = os.path.join(gt_dir, name)

    pred = load_rgb(p_path)
    gt = load_rgb(g_path)

    if pred.shape != gt.shape:
        print("skip(shape mismatch):", name, pred.shape, gt.shape)
        continue

    v_psnr = psnr(pred, gt)
    v_ssim = ssim_rgb(pred, gt)

    psnr_list.append(v_psnr)
    ssim_list.append(v_ssim)
    count += 1

    print(f"{name}: PSNR={v_psnr:.4f}, SSIM={v_ssim:.6f}")

print("\n====================")
print("paired images:", count)
print("average PSNR:", float(np.mean(psnr_list)) if psnr_list else "N/A")
print("average SSIM:", float(np.mean(ssim_list)) if ssim_list else "N/A")