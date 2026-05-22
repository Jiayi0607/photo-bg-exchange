"""证件照换底 — PyQt6 现代桌面应用 (v2)"""
import sys, os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QColorDialog,
    QProgressBar, QMessageBox, QSizePolicy, QFrame,
    QComboBox, QStackedWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData, QRect
from PyQt6.QtGui import (
    QPixmap, QImage, QDragEnterEvent, QDropEvent,
    QColor, QPainter, QFont, QFontDatabase, QPen, QBrush, QAction,
)

WORK_DIR = Path(__file__).parent
sys.path.insert(0, str(WORK_DIR))
from bg_remover import (
    _inference_mask, estimate_bg_color, color_decontaminate,
    blend, make_gradient, BG_COLORS_BGR, _imread, _imwrite,
)

# ── 标准证件照尺寸 (宽×高 @300dpi) ──────────────────────────
PHOTO_SIZES = {
    "原始尺寸": None,
    "1寸 (25×35mm)": (295, 413),
    "小2寸 (35×45mm)": (413, 531),
    "2寸 (35×53mm)": (413, 626),
}

# ── QSS 样式 ──────────────────────────────────────────────────
QSS = """
QMainWindow { background-color: #0d1117; border: none; }
QWidget { background-color: #0d1117; color: #e6edf3; font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; font-size: 14px; }

QFrame#panel {
    background-color: #161b22; border: 1px solid #30363d; border-radius: 12px;
}
QFrame#dropZone {
    background-color: #161b22; border: 2px dashed #30363d; border-radius: 16px; padding: 20px;
}
QFrame#dropZone:hover { border-color: #58a6ff; background-color: #1a2230; }

QLabel#titleLabel { font-size: 22px; font-weight: bold; color: #f0f6fc; padding: 8px 0; }
QLabel#panelHeader { font-size: 13px; color: #8b949e; padding: 8px 12px 0 12px; background: transparent; }
QLabel#hintLabel { color: #8b949e; font-size: 14px; background: transparent; }
QLabel#previewPlaceholder { color: #484f58; font-size: 13px; background: transparent; }
QLabel#importHint { color: #484f58; font-size: 12px; background: transparent; }
QLabel#zoomInfo { color: #484f58; font-size: 12px; background: transparent; }

QPushButton {
    border-radius: 8px; padding: 10px 24px; font-size: 14px; font-weight: 600;
    border: 1px solid #30363d; background-color: #21262d; color: #c9d1d9; min-height: 20px;
}
QPushButton:hover { background-color: #30363d; border-color: #8b949e; }
QPushButton:pressed { background-color: #1a1f27; }

QPushButton#btnBlue {
    background-color: rgba(67, 142, 219, 0.25); border: 2px solid rgba(67, 142, 219, 0.6); color: #e6edf3;
}
QPushButton#btnBlue:hover, QPushButton#btnBlue[checked="true"] {
    background-color: rgba(67, 142, 219, 0.45); border-color: #58a6ff;
}
QPushButton#btnRed {
    background-color: rgba(233, 31, 25, 0.25); border: 2px solid rgba(233, 31, 25, 0.6); color: #e6edf3;
}
QPushButton#btnRed:hover, QPushButton#btnRed[checked="true"] {
    background-color: rgba(233, 31, 25, 0.45); border-color: #ff4d4d;
}
QPushButton#btnWhite {
    background-color: rgba(255, 255, 255, 0.12); border: 2px solid rgba(255, 255, 255, 0.35); color: #e6edf3;
}
QPushButton#btnWhite:hover, QPushButton#btnWhite[checked="true"] {
    background-color: rgba(255, 255, 255, 0.22); border-color: #b0b0b0;
}
QPushButton#toggleGradient {
    background-color: rgba(209, 47, 47, 0.2); border: 2px solid rgba(209, 47, 47, 0.5);
    border-radius: 8px; padding: 6px 14px; font-size: 13px; font-weight: 600;
    color: #e6edf3;
}
QPushButton#toggleGradient:hover {
    border-color: #ff4d4d;
}
QPushButton#toggleGradient:checked {
    background-color: rgba(46, 160, 67, 0.3); border: 2px solid rgba(46, 160, 67, 0.6);
    color: #e6edf3;
}
QPushButton#toggleGradient:checked:hover {
    border-color: #3fb950;
}

QPushButton#btnCustom {
    background-color: #1f252d; border: 2px dashed #484f58; color: #8b949e;
    font-size: 18px; min-width: 44px; max-width: 44px; min-height: 44px; max-height: 44px;
    padding: 0; border-radius: 22px;
}
QPushButton#btnCustom:hover { border-color: #8b949e; color: #c9d1d9; }

QPushButton#btnSmall {
    padding: 4px 12px; font-size: 12px; border-radius: 6px; min-height: 0;
}
QPushButton#btnSave {
    background-color: #238636; border: 1px solid rgba(240, 246, 252, 0.15);
    color: #ffffff; font-size: 15px; padding: 12px 48px; border-radius: 10px;
}
QPushButton#btnSave:hover { background-color: #2ea043; }
QPushButton#btnSave:disabled { background-color: #21262d; color: #484f58; }

QProgressBar {
    border: none; border-radius: 6px; background-color: #21262d;
    text-align: center; color: #8b949e; font-size: 12px; min-height: 6px; max-height: 6px;
}
QProgressBar::chunk { background-color: #58a6ff; border-radius: 6px; }

QComboBox {
    background-color: #21262d; border: 1px solid #30363d; border-radius: 8px;
    padding: 6px 12px; color: #c9d1d9; font-size: 13px; min-width: 120px;
}
QComboBox:hover { border-color: #8b949e; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background-color: #161b22; border: 1px solid #30363d; selection-background-color: #1f6feb;
    color: #e6edf3; font-size: 13px; outline: none;
}
"""


# ═══════════════════════════════════════════════════════════
# 处理线程
# ═══════════════════════════════════════════════════════════
class ProcessThread(QThread):
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, image: np.ndarray, bg_color_bgr: tuple, gradient_mode: bool = False):
        super().__init__()
        self.image = image
        self.bg_color_bgr = bg_color_bgr
        self.gradient_mode = gradient_mode

    def run(self):
        try:
            self.progress.emit(15)
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # 预处理：混白淡化背景，改善 MODNet 在红底上的分割
            WHITE_MIX = 0.35
            if WHITE_MIX > 0:
                white = np.full_like(img_rgb, 255)
                img_input = cv2.addWeighted(img_rgb, 1.0 - WHITE_MIX, white, WHITE_MIX, 0)
            else:
                img_input = img_rgb

            self.progress.emit(30)
            mask = _inference_mask(img_input)
            self.progress.emit(45)
            bg_est = estimate_bg_color(self.image, mask)

            img_clean = color_decontaminate(self.image, mask, bg_est)
            self.progress.emit(70)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_final = cv2.erode(mask, kernel, iterations=3)
            self.progress.emit(80)

            if self.gradient_mode:
                bg_img = make_gradient(
                    self.image.shape[0], self.image.shape[1],
                    self.bg_color_bgr, (255, 255, 255),
                )
                result = blend(img_clean, mask_final, None, bg_image=bg_img)
            else:
                result = blend(img_clean, mask_final, self.bg_color_bgr)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════
# 图片预览组件 (支持缩放 + 拖拽平移)
# ═══════════════════════════════════════════════════════════
class ImagePreview(QWidget):
    """可缩放/平移的图片预览组件。"""

    zoomChanged = pyqtSignal(float)
    panChanged = pyqtSignal(float, float)

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0          # 1.0 = 适应容器
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._dragging = False
        self._drag_start = None
        self._placeholder = text
        self.setMinimumHeight(280)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setStyleSheet("background: transparent;")

    # ── 公共接口 ──────────────────────────────────────────

    def set_pixmap(self, pm: QPixmap):
        self._pixmap = pm
        self._reset_view()
        self.update()

    def _reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

    def set_zoom(self, zoom: float):
        self._zoom = max(1.0, zoom)
        self._clamp_pan()
        self.update()

    def fit_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def has_image(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    # ── 坐标计算 ──────────────────────────────────────────

    def _fit_scale(self) -> float:
        if self._pixmap is None:
            return 1.0
        return min(self.width() / self._pixmap.width(),
                   self.height() / self._pixmap.height())

    def _display_size(self):
        s = self._fit_scale() * self._zoom
        w = int(self._pixmap.width() * s)
        h = int(self._pixmap.height() * s)
        return w, h, s

    def _clamp_pan(self):
        if self._pixmap is None:
            return
        dw, dh, _ = self._display_size()
        mx = max(0, (dw - self.width()) // 2)
        my = max(0, (dh - self.height()) // 2)
        self._pan_x = max(-mx, min(mx, self._pan_x))
        self._pan_y = max(-my, min(my, self._pan_y))

    # ── 绘制 ──────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        bg = QColor("#161b22")
        p.fillRect(self.rect(), bg)

        if not self.has_image():
            p.setPen(QColor("#484f58"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        dw, dh, scale = self._display_size()
        scaled = self._pixmap.scaled(
            dw, dh, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        cx = (self.width() - dw) // 2 + int(self._pan_x)
        cy = (self.height() - dh) // 2 + int(self._pan_y)
        p.drawPixmap(cx, cy, scaled)

    # ── 事件 ──────────────────────────────────────────────

    def wheelEvent(self, event):
        if not self.has_image():
            return
        delta = event.angleDelta().y()
        old_zoom = self._zoom
        if delta > 0:
            self._zoom = min(self._zoom * 1.2, 10.0)
        else:
            self._zoom = max(self._zoom / 1.2, 1.0)
        if abs(self._zoom - old_zoom) > 0.001:
            self._clamp_pan()
            self.zoomChanged.emit(self._zoom)
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.has_image() and self._zoom > 1.0:
            self._dragging = True
            self._drag_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start is not None:
            dx = event.position().x() - self._drag_start.x()
            dy = event.position().y() - self._drag_start.y()
            self._pan_x += dx
            self._pan_y += dy
            self._drag_start = event.position()
            self._clamp_pan()
            self.update()
            self.panChanged.emit(self._pan_x, self._pan_y)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clamp_pan()
        self.update()


# ═══════════════════════════════════════════════════════════
# 拖拽区
# ═══════════════════════════════════════════════════════════
class DropZone(QFrame):
    fileDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        self.icon = QLabel("📂")
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon.setStyleSheet("font-size: 48px; background: transparent;")
        layout.addWidget(self.icon)

        self.label = QLabel("拖拽图片到此处\n或点击选择文件\n支持格式: JPG / PNG")
        self.label.setObjectName("hintLabel")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("background: transparent;")
        layout.addWidget(self.label)

    def dragEnterEvent(self, ev: QDragEnterEvent):
        if ev.mimeData().hasUrls():
            url = ev.mimeData().urls()[0]
            if url.toLocalFile().lower().endswith((".png", ".jpg", ".jpeg")):
                ev.acceptProposedAction()

    def dropEvent(self, ev: QDropEvent):
        paths = [u.toLocalFile() for u in ev.mimeData().urls()]
        if paths:
            self.fileDropped.emit(paths[0])

    def mousePressEvent(self, ev):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择证件照", "",
            "图片文件 (*.png *.jpg *.jpeg);;PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if path:
            self.fileDropped.emit(path)


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════
def bgr_to_qpixmap(img: np.ndarray) -> QPixmap:
    h, w = img.shape[:2]
    return QPixmap.fromImage(
        QImage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).data, w, h, 3 * w, QImage.Format.Format_RGB888)
    )


def crop_to_size(img: np.ndarray, target_size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """居中裁切 + resize 到目标尺寸。target_size=None 返回原图。"""
    if target_size is None:
        return img
    tw, th = target_size
    h, w = img.shape[:2]
    target_aspect = tw / th
    current_aspect = w / h
    if abs(current_aspect - target_aspect) < 0.001:
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        x = (w - new_w) // 2
        cropped = img[:, x:x + new_w]
    else:
        new_h = int(w / target_aspect)
        y = (h - new_h) // 2
        cropped = img[y:y + new_h]
    return cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_AREA)


# ═══════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("证件照换底")
        self.setMinimumSize(880, 640)
        self.resize(1000, 720)

        # ── 状态 ──────────────────────────────────────────
        self._image: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._bg_color_bgr = BG_COLORS_BGR["blue"]
        self._active_color = "blue"
        self._gradient_mode = False
        self._thread: ProcessThread | None = None

        self._center()
        self._build_ui()
        self.setStyleSheet(QSS)

    def _center(self):
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2,
        )

    # ══════════════════════════════════════════════════════
    # UI 构建
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(32, 20, 32, 28)
        root.setSpacing(14)

        # ── 标题 ──────────────────────────────────────────
        title = QLabel("📷 证件照换底")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # ── 双栏预览 ──────────────────────────────────────
        preview_row = QHBoxLayout()
        preview_row.setSpacing(16)
        root.addLayout(preview_row, 1)

        # 左栏 - Before
        before_frame = QFrame()
        before_frame.setObjectName("panel")
        before_col = QVBoxLayout(before_frame)
        before_col.setContentsMargins(0, 0, 0, 0)
        before_col.setSpacing(4)

        before_header = QLabel("原始图片")
        before_header.setObjectName("panelHeader")
        before_col.addWidget(before_header)

        self.before_stacked = QStackedWidget()
        self.before_stacked.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.drop_zone = DropZone()
        self.drop_zone.fileDropped.connect(self._on_file_loaded)
        self.before_preview = ImagePreview("拖拽图片到此处\n或点击左侧选择文件\n支持格式: JPG / PNG")
        self.before_preview.zoomChanged.connect(self._sync_zoom)
        self.before_preview.panChanged.connect(self._sync_pan)

        self.before_stacked.addWidget(self.drop_zone)      # index 0
        self.before_stacked.addWidget(self.before_preview)  # index 1
        before_col.addWidget(self.before_stacked, 1)

        # 左栏底部按钮 (仅在加载后显示)
        before_toolbar = QHBoxLayout()
        before_toolbar.setContentsMargins(8, 0, 8, 6)
        self.btn_change = QPushButton("📂 更换")
        self.btn_change.setObjectName("btnSmall")
        self.btn_change.clicked.connect(self._select_new_file)
        self.btn_change.setVisible(False)
        before_toolbar.addWidget(self.btn_change)
        before_toolbar.addStretch()
        before_col.addLayout(before_toolbar)

        preview_row.addWidget(before_frame, 1)

        # 右栏 - After
        after_frame = QFrame()
        after_frame.setObjectName("panel")
        after_col = QVBoxLayout(after_frame)
        after_col.setContentsMargins(0, 0, 0, 0)
        after_col.setSpacing(4)

        after_header = QLabel("处理结果")
        after_header.setObjectName("panelHeader")
        after_col.addWidget(after_header)

        self.after_preview = ImagePreview("选择图片后\n此处显示结果")
        self.after_preview.zoomChanged.connect(self._sync_zoom)
        self.after_preview.panChanged.connect(self._sync_pan)
        after_col.addWidget(self.after_preview, 1)

        after_toolbar = QHBoxLayout()
        after_toolbar.setContentsMargins(8, 0, 8, 6)
        after_toolbar.addStretch()
        after_col.addLayout(after_toolbar)

        preview_row.addWidget(after_frame, 1)

        # ── 颜色选择 ──────────────────────────────────────
        color_bar = QHBoxLayout()
        color_bar.setSpacing(10)

        color_label = QLabel("背景色")
        color_label.setStyleSheet("color: #8b949e; font-size: 14px;")
        color_bar.addWidget(color_label)

        self.btn_blue = QPushButton("蓝底")
        self.btn_blue.setObjectName("btnBlue")
        self.btn_blue.setCheckable(True)
        self.btn_blue.setChecked(True)
        self.btn_blue.clicked.connect(lambda: self._select_preset("blue"))

        self.btn_red = QPushButton("红底")
        self.btn_red.setObjectName("btnRed")
        self.btn_red.setCheckable(True)
        self.btn_red.clicked.connect(lambda: self._select_preset("red"))

        self.btn_white = QPushButton("白底")
        self.btn_white.setObjectName("btnWhite")
        self.btn_white.setCheckable(True)
        self.btn_white.clicked.connect(lambda: self._select_preset("white"))

        self.btn_custom = QPushButton("🎨")
        self.btn_custom.setObjectName("btnCustom")
        self.btn_custom.setToolTip("自定义颜色")
        self.btn_custom.clicked.connect(self._pick_custom_color)

        color_bar.addWidget(self.btn_blue)
        color_bar.addWidget(self.btn_red)
        color_bar.addWidget(self.btn_white)
        color_bar.addSpacing(10)
        color_bar.addWidget(self.btn_custom)

        # 渐变开关
        color_bar.addSpacing(16)
        grad_label = QLabel("渐变")
        grad_label.setStyleSheet("color: #8b949e; font-size: 13px;")
        color_bar.addWidget(grad_label)

        self.toggle_gradient = QPushButton("关")
        self.toggle_gradient.setObjectName("toggleGradient")
        self.toggle_gradient.setCheckable(True)
        self.toggle_gradient.setToolTip("勾选后背景从所选颜色渐变到白色")
        self.toggle_gradient.toggled.connect(self._on_gradient_toggled)
        color_bar.addWidget(self.toggle_gradient)

        color_bar.addStretch()
        root.addLayout(color_bar)

        # ── 设置行：尺寸模板 + 导出格式 + 导入提示 ─────────
        settings_row = QHBoxLayout()
        settings_row.setSpacing(16)

        # 尺寸模板
        size_label = QLabel("尺寸模板")
        size_label.setStyleSheet("color: #8b949e; font-size: 13px;")
        self.size_combo = QComboBox()
        for name in PHOTO_SIZES:
            self.size_combo.addItem(name)
        self.size_combo.currentIndexChanged.connect(self._on_size_changed)
        settings_row.addWidget(size_label)
        settings_row.addWidget(self.size_combo)

        # 分隔
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #30363d; font-size: 16px; background: transparent;")
        settings_row.addWidget(sep1)

        # 导出格式
        fmt_label = QLabel("导出格式")
        fmt_label.setStyleSheet("color: #8b949e; font-size: 13px;")
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItem("JPEG")
        self.fmt_combo.addItem("PNG")
        settings_row.addWidget(fmt_label)
        settings_row.addWidget(self.fmt_combo)

        # 缩放信息
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #30363d; font-size: 16px; background: transparent;")
        settings_row.addWidget(sep2)

        zoom_label = QLabel("视图")
        zoom_label.setStyleSheet("color: #8b949e; font-size: 13px;")
        self.zoom_info = QLabel("1.0×")
        self.zoom_info.setStyleSheet("color: #e6edf3; font-size: 13px; min-width: 36px;")
        self.btn_reset_view = QPushButton("重置")
        self.btn_reset_view.setObjectName("btnSmall")
        self.btn_reset_view.clicked.connect(self._fit_both_views)
        settings_row.addWidget(zoom_label)
        settings_row.addWidget(self.zoom_info)
        settings_row.addWidget(self.btn_reset_view)

        settings_row.addStretch()

        settings_row.addStretch()

        # 导入提示
        self.import_hint = QLabel("支持: JPG / PNG")
        self.import_hint.setObjectName("importHint")
        settings_row.addWidget(self.import_hint)

        root.addLayout(settings_row)

        # ── 进度 + 保存 ─────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(12)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        action_row.addWidget(self.progress, 1)

        self.btn_save = QPushButton("💾 保存图片")
        self.btn_save.setObjectName("btnSave")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_result)
        action_row.addWidget(self.btn_save)

        root.addLayout(action_row)

        # ── 尺寸信息标签 ─────────────────────────────────
        self.size_info = QLabel("")
        self.size_info.setStyleSheet("color: #484f58; font-size: 12px;")
        self.size_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.size_info.setVisible(False)
        root.addWidget(self.size_info)

    # ══════════════════════════════════════════════════════
    # 事件响应
    # ══════════════════════════════════════════════════════

    def _on_file_loaded(self, path: str):
        try:
            img = _imread(path)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法读取图片: {path}\n{e}")
            return

        self._image = img

        # 左侧切换到预览模式
        self.before_stacked.setCurrentIndex(1)
        self.btn_change.setVisible(True)
        # 显示原图
        self.before_preview.set_pixmap(bgr_to_qpixmap(img))

        # 更新信息
        h, w = img.shape[:2]
        basename = os.path.basename(path)
        self.size_info.setText(f"{basename}  •  {w}×{h}")
        self.size_info.setVisible(True)
        self.zoom_info.setText("1.0×")

        # 触发处理
        self._process()

    def _select_new_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择证件照", "",
            "图片文件 (*.png *.jpg *.jpeg);;PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if path:
            self._on_file_loaded(path)

    def _select_preset(self, color: str):
        self._active_color = color
        self._bg_color_bgr = BG_COLORS_BGR[color]
        for btn in [self.btn_blue, self.btn_red, self.btn_white]:
            btn.setChecked(False)
        getattr(self, f"btn_{color}").setChecked(True)
        if self._image is not None:
            self._process()

    def _pick_custom_color(self):
        c = QColorDialog.getColor(
            QColor(*reversed(BG_COLORS_BGR["blue"])),
            self, "选择背景色",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if c.isValid():
            self._bg_color_bgr = (c.blue(), c.green(), c.red())
            self._active_color = "custom"
            for btn in [self.btn_blue, self.btn_red, self.btn_white]:
                btn.setChecked(False)
            if self._image is not None:
                self._process()

    def _on_gradient_toggled(self, checked: bool):
        self._gradient_mode = checked
        self.toggle_gradient.setText("开" if checked else "关")
        if self._image is not None:
            self._process()

    def _on_size_changed(self, idx: int):
        name = self.size_combo.currentText()
        sz = PHOTO_SIZES[name]
        if sz:
            self.size_info.setText(f"导出尺寸: {sz[0]}×{sz[1]} px")
        elif self._image is not None:
            h, w = self._image.shape[:2]
            self.size_info.setText(f"原始尺寸: {w}×{h} px")

    def _process(self):
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.btn_save.setEnabled(False)

        self._thread = ProcessThread(
            self._image, self._bg_color_bgr,
            gradient_mode=self._gradient_mode,
        )
        self._thread.progress.connect(self.progress.setValue)
        self._thread.finished.connect(self._on_result)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_result(self, result: np.ndarray):
        self._result = result
        self.progress.setVisible(False)
        self.btn_save.setEnabled(True)
        self.after_preview.set_pixmap(bgr_to_qpixmap(result))
        self.zoom_info.setText("1.0×")
        self._on_size_changed(self.size_combo.currentIndex())

    def _on_error(self, msg: str):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "处理出错", msg)

    def _fit_both_views(self):
        self.before_preview.fit_view()
        self.after_preview.fit_view()
        self._update_zoom_label()

    def _update_zoom_label(self):
        z = self.before_preview._zoom
        if z <= 1.0:
            self.zoom_info.setText("1.0×")
        else:
            self.zoom_info.setText(f"{z:.1f}×")

    def _sync_zoom(self, zoom: float):
        """同步两个预览区的缩放。"""
        self.before_preview.blockSignals(True)
        self.after_preview.blockSignals(True)
        self.before_preview.set_zoom(zoom)
        self.after_preview.set_zoom(zoom)
        self.before_preview.blockSignals(False)
        self.after_preview.blockSignals(False)
        self._update_zoom_label()

    def _sync_pan(self, px: float, py: float):
        """同步两个预览区的平移。"""
        self.before_preview.blockSignals(True)
        self.after_preview.blockSignals(True)
        self.before_preview._pan_x = px
        self.before_preview._pan_y = py
        self.after_preview._pan_x = px
        self.after_preview._pan_y = py
        self.before_preview.update()
        self.after_preview.update()
        self.before_preview.blockSignals(False)
        self.after_preview.blockSignals(False)

    # ══════════════════════════════════════════════════════
    # 保存
    # ══════════════════════════════════════════════════════

    def _save_result(self):
        if self._result is None:
            return

        # 尺寸裁切
        size_name = self.size_combo.currentText()
        target = PHOTO_SIZES[size_name]
        img_to_save = crop_to_size(self._result, target)

        # 格式
        fmt = self.fmt_combo.currentText()  # "PNG" or "JPEG"
        ext = "png" if fmt == "PNG" else "jpg"
        filter_str = f"{fmt.upper()} (*.{ext})"

        default_name = f"证件照.{ext}"
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "保存证件照", default_name, filter_str,
        )
        if not path:
            return

        # 根据对话框实际选择的扩展名决定编码方式
        save_ext = path.rsplit(".", 1)[-1].lower()
        if save_ext == "png":
            _imwrite(path, img_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            _imwrite(path, img_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])

        QMessageBox.information(self, "保存成功", f"已保存到:\n{path}")


# ═══════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
