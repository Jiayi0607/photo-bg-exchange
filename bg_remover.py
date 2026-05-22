"""证件照背景替换 — 核心逻辑

流程: 混白预处理 → MODNet推理 → 色彩去污染 → Erosion → Alpha混合
"""
import os
import sys

from typing import Optional

import numpy as np
import cv2

# ImageNet 归一化参数
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None


# ============ 标准证件照背景色 (BGR) ============
BG_COLORS_BGR = {
    "blue":   (219, 142, 67),
    "red":    (25, 31, 233),
    "white":  (255, 255, 255),
}


# ============ MODNet 模型 ============
def _get_session():
    global _session
    if _session is not None:
        return _session

    import onnxruntime as ort

    # 查找模型：打包模式 → sys._MEIPASS, 开发模式 → 脚本目录
    model_path = None
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(__file__)
    candidate = os.path.join(base, 'models', 'model.onnx')
    if os.path.exists(candidate):
        model_path = candidate
    else:
        model_path = os.path.join(
            os.environ.get("MODELSCOPE_CACHE",
                os.path.expanduser("~/.cache/modelscope/hub/models")),
            "Xenova/modnet/onnx/model.onnx"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MODNet model not found: {model_path}")

    _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"MODNet loaded ({size_mb:.1f} MB) | CPU")
    return _session


def _inference_mask(img_rgb: np.ndarray) -> np.ndarray:
    """MODNet 512×512 推理，返回原始分辨率的 alpha mask [0,1]"""
    h, w = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (512, 512))
    norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    inp = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
    out = _get_session().run(None, {"input": inp})[0][0, 0]
    return np.clip(cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


# ============ 辅助函数 ============
def estimate_bg_color(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """从 alpha < 0.05 的纯背景区域采样，估计原始背景色。返回 BGR float32 (3,)"""
    bg_pixels = image[mask < 0.05]
    if len(bg_pixels) < 100:
        return np.array([255.0, 255.0, 255.0], dtype=np.float32)
    return np.median(bg_pixels, axis=0).astype(np.float32)


def color_decontaminate(
    image: np.ndarray,
    mask: np.ndarray,
    bg_color: np.ndarray,
) -> np.ndarray:
    """
    色彩去污染: 在过渡区 (0.08 < alpha < 0.98) 洗掉原始背景色的贡献。
    原理: pixel = F_true × α + B_orig × (1-α)
          → F_true = (pixel - B_orig × (1-α)) / α
    """
    img_f = image.astype(np.float32)
    transition = (mask > 0.08) & (mask < 0.98)
    alpha_safe = np.maximum(mask[transition], 1e-6)

    clean = img_f.copy()
    for c in range(3):
        clean[:, :, c][transition] = (
            img_f[:, :, c][transition]
            - bg_color[c] * (1.0 - mask[transition])
        ) / alpha_safe

    return np.clip(clean, 0, 255).astype(np.uint8)


def make_gradient(h: int, w: int, top_bgr: tuple, bottom_bgr: tuple = (255, 255, 255)) -> np.ndarray:
    """生成垂直线性渐变背景 (BGR), top → bottom。"""
    grad = np.zeros((h, w, 3), dtype=np.float32)
    # 用 broadcasting 避免 Python 循环
    ratio = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]  # (H, 1)
    for c in range(3):
        grad[:, :, c] = top_bgr[c] * (1.0 - ratio) + bottom_bgr[c] * ratio
    return grad


def blend(image: np.ndarray, mask: np.ndarray, bg_color, bg_image: Optional[np.ndarray] = None) -> np.ndarray:
    """Alpha blending: result = fg × α + bg × (1-α). 高斯模糊让边缘更自然。

    bg_image 不为 None 时直接用作背景图（支持渐变），否则用 bg_color 纯色。
    """
    if bg_image is not None:
        bg_f = bg_image.astype(np.float32)
    else:
        if isinstance(bg_color, str):
            bg_bgr = BG_COLORS_BGR[bg_color]
        else:
            bg_bgr = bg_color
        bg_f = np.zeros_like(image, dtype=np.float32)
        bg_f[:, :, 0], bg_f[:, :, 1], bg_f[:, :, 2] = bg_bgr[0], bg_bgr[1], bg_bgr[2]

    fg_f = image.astype(np.float32)
    # 高斯模糊柔和边缘
    α = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), sigmaX=0, sigmaY=0)
    α = α[:, :, np.newaxis]
    return (fg_f * α + bg_f * (1.0 - α)).astype(np.uint8)


def _imread(path: str) -> np.ndarray:
    """读取图片 — 支持中文路径"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _imwrite(path: str, img: np.ndarray, params=None):
    """保存图片 — 支持中文路径"""
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        ext = ".png"
    buf = cv2.imencode(ext, img, params or [])[1]
    buf.tofile(path)


# ============ 主流程 ============
def process_image(
    image_path: str,
    bg_color: str = "blue",
    output_path: str = None,
    lossless: bool = True,
    erode_iters: int = 3,
) -> str:
    """
    证件照换底。
    参数:
        image_path:   输入图片路径
        bg_color:     目标背景色 (blue / red / white)
        output_path:  输出路径 (默认: {name}_{bg_color}.png)
        lossless:     PNG无损输出 (默认True)
        erode_iters:  腐蚀轮次 (默认3, 设为0可跳过)
    返回:
        输出文件路径
    """
    img = _imread(image_path)
    print(f"Input: {image_path} ({img.shape[1]}×{img.shape[0]})")

    # 1. MODNet 推理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = _inference_mask(img_rgb)

    # 2. 色彩去污染
    bg_est = estimate_bg_color(img, mask)
    img_clean = color_decontaminate(img, mask, bg_est)
    print(f"  Decontaminated (bg_est={bg_est.astype(int)})")

    # 3. Erosion 消白边
    if erode_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=erode_iters)

    # 4. Alpha blending
    result = blend(img_clean, mask, bg_color)

    # 5. 保存
    if output_path is None:
        name, _ = os.path.splitext(image_path)
        ext = ".png" if lossless else ".jpg"
        output_path = f"{name}_{bg_color}{ext}"

    if lossless:
        _imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    else:
        _imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Saved: {output_path} ({result.shape[1]}×{result.shape[0]})")
    return output_path


# ============ CLI 测试入口 ============
if __name__ == "__main__":
    print("=" * 50)
    print("证件照背景替换 — MODNet")
    print("用法: process_image('photo.jpg', 'blue')")
    print("=" * 50)