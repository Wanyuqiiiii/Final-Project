import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as e:  # noqa: BLE001
    raise RuntimeError(
        "PySide6 is not installed. Install it with: pip install PySide6"
    ) from e

from track_pipeline import RunConfig, RuntimeStats, process_video


def _imread_unicode(path: str):
    """Robust image reader for Windows paths with non-ASCII chars."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _is_image_path(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _save_first_frame_as_image(video_path: str, image_path: str) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    try:
        ext = Path(image_path).suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            return bool(cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(image_path))
        if ext == ".png":
            return bool(cv2.imencode(".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])[1].tofile(image_path))
        if ext == ".bmp":
            return bool(cv2.imencode(".bmp", frame)[1].tofile(image_path))
        if ext == ".webp":
            return bool(cv2.imencode(".webp", frame, [int(cv2.IMWRITE_WEBP_QUALITY), 95])[1].tofile(image_path))
    except Exception:
        return False
    return False


def _format_stats(s: RuntimeStats) -> str:
    d = asdict(s)
    if not d.get("stat_label"):
        return (
            f"processed={d['processed']} read={d['read_frames']} dets={d['dets']} tracks={d['tracks']} "
            f"FPS={d['fps']:.1f} YOLO={d['yolo_ms']:.1f}ms SORT={d['sort_ms']:.1f}ms"
        )
    return (
        f"processed={d['processed']} read={d['read_frames']} dets={d['dets']} tracks={d['tracks']} "
        f"累计检测框={d['total_dets']} 历史ID数={d['total_ids_ever']} "
        f"越线 in={d['line_in']} out={d['line_out']} "
        f"{d.get('stat_label','')}:轨迹={d.get('stat_in_roi_tracks',0)} 检测={d.get('stat_in_roi_dets',0)} "
        f"FPS={d['fps']:.1f} YOLO={d['yolo_ms']:.1f}ms SORT={d['sort_ms']:.1f}ms"
    )


class InferenceWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(QtGui.QImage, int, int)  # image, read_frames, processed
    log_line = QtCore.Signal(str)
    progress = QtCore.Signal(int)  # 0..100
    finished_ok = QtCore.Signal(str)  # out_path
    failed = QtCore.Signal(str)

    def __init__(self, cfg: RunConfig, out_path: Optional[str], parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.out_path = out_path
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):  # noqa: C901
        def on_progress(p: float, s: RuntimeStats):
            self.progress.emit(int(max(0, min(100, p * 100))))
            if int(s.processed) % 5 == 0:
                self.log_line.emit(_format_stats(s))

        def on_frame(frame_bgr, s: RuntimeStats):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            h0, w0 = rgb.shape[:2]
            bytes_per_line = int(rgb.strides[0])
            qimg = QtGui.QImage(
                rgb.data,
                w0,
                h0,
                bytes_per_line,
                QtGui.QImage.Format_RGB888,
            ).copy()
            self.frame_ready.emit(qimg, int(s.read_frames), int(s.processed))

        try:
            self.log_line.emit("Starting inference...")
            process_video(
                self.cfg,
                on_progress=on_progress,
                on_frame=on_frame,
                frame_every=1,
                should_stop=lambda: bool(self._stop),
            )
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
            return

        if self._stop:
            self.log_line.emit("Stopped.")
        else:
            self.log_line.emit("Done.")

        self.progress.emit(100)
        self.finished_ok.emit(self.out_path or "")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Real-time Detection (PyQt)")

        self.worker: Optional[InferenceWorker] = None
        self.last_out_path: Optional[str] = None

        # left: controls
        self.input_mode = QtWidgets.QComboBox()
        self.input_mode.addItems(["Video", "Image"])
        self.video_path = QtWidgets.QLineEdit()
        self.btn_pick_video = QtWidgets.QPushButton("Pick file...")
        self.model_path = QtWidgets.QLineEdit("yolov8n.pt")
        self.btn_pick_model = QtWidgets.QPushButton("选择权重文件...")

        self.device = QtWidgets.QComboBox()
        self.device.addItems(["cpu", "0"])
        self.algo_mode = QtWidgets.QComboBox()
        self.algo_mode.addItem("YOLOv8 only", "yolo_only")
        self.algo_mode.addItem("YOLOv8 + counting", "yolo_count")
        self.algo_mode.addItem("YOLOv8 + tracking", "yolo_track")
        self.algo_mode.addItem("YOLOv8 + tracking + counting", "all")
        self.imgsz = QtWidgets.QSpinBox()
        self.imgsz.setRange(320, 1280)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        self.conf = QtWidgets.QDoubleSpinBox()
        self.conf.setRange(0.01, 0.99)
        self.conf.setSingleStep(0.01)
        self.conf.setValue(0.25)
        self.iou = QtWidgets.QDoubleSpinBox()
        self.iou.setRange(0.01, 0.99)
        self.iou.setSingleStep(0.01)
        self.iou.setValue(0.45)
        self.vid_stride = QtWidgets.QSpinBox()
        self.vid_stride.setRange(1, 10)
        self.vid_stride.setValue(1)

        self.stat_target = QtWidgets.QComboBox()
        self.stat_target.addItem("车辆（COCO: 2,3,5,7）", "vehicle_coco")
        self.stat_target.addItem("行人（COCO: 0）", "person")
        self.stat_target.addItem("全部类别", "all")
        self.stat_target.addItem("自定义类别…", "custom")
        self.stat_target.setCurrentIndex(0)
        self.stat_custom = QtWidgets.QLineEdit()
        self.stat_custom.setPlaceholderText("custom 时填写，如 2,3,5,7")
        self.stat_custom.setEnabled(False)
        self.stat_target.currentIndexChanged.connect(
            lambda _i: self.stat_custom.setEnabled(self.stat_target.currentData() == "custom")
        )
        self.roi_ml = QtWidgets.QDoubleSpinBox()
        self.roi_ml.setRange(0.0, 0.45)
        self.roi_ml.setSingleStep(0.01)
        self.roi_ml.setValue(0.00)
        self.roi_mt = QtWidgets.QDoubleSpinBox()
        self.roi_mt.setRange(0.0, 0.45)
        self.roi_mt.setSingleStep(0.01)
        self.roi_mt.setValue(0.00)
        self.roi_mr = QtWidgets.QDoubleSpinBox()
        self.roi_mr.setRange(0.0, 0.45)
        self.roi_mr.setSingleStep(0.01)
        self.roi_mr.setValue(0.00)
        self.roi_mb = QtWidgets.QDoubleSpinBox()
        self.roi_mb.setRange(0.0, 0.45)
        self.roi_mb.setSingleStep(0.01)
        self.roi_mb.setValue(0.00)
        self.enable_line_count = QtWidgets.QCheckBox("启用越线计数（水平线）")
        self.enable_line_count.setChecked(True)
        self.line_y_frac = QtWidgets.QDoubleSpinBox()
        self.line_y_frac.setRange(0.05, 0.95)
        self.line_y_frac.setSingleStep(0.01)
        self.line_y_frac.setValue(0.55)

        # SORT：减少 ID 乱跳（与 track_pipeline.RunConfig 默认一致，可再调）
        self.trk_max_age = QtWidgets.QSpinBox()
        self.trk_max_age.setRange(1, 200)
        self.trk_max_age.setValue(60)
        self.trk_max_age.setToolTip("连续多少帧没匹配到检测仍保留该轨迹（越大越不容易换 ID）")
        self.trk_min_hits = QtWidgets.QSpinBox()
        self.trk_min_hits.setRange(1, 20)
        self.trk_min_hits.setValue(3)
        self.trk_min_hits.setToolTip("累计命中多少次才输出为确认轨迹")
        self.trk_iou = QtWidgets.QDoubleSpinBox()
        self.trk_iou.setRange(0.05, 0.95)
        self.trk_iou.setSingleStep(0.05)
        self.trk_iou.setValue(0.20)
        self.trk_iou.setToolTip("检测框与预测框 IoU 关联阈值（略低更容易在框抖动时续上原 ID）")

        self.save_output = QtWidgets.QCheckBox("Save result.mp4")
        self.save_output.setChecked(True)
        self.output_path = QtWidgets.QLineEdit(str(Path.cwd() / "result.mp4"))
        self.btn_pick_output = QtWidgets.QPushButton("Pick output...")

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_open_out = QtWidgets.QPushButton("Open output folder")
        self.btn_open_out.setEnabled(False)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)

        # right: preview + logs
        self.preview = QtWidgets.QLabel("Preview")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(800, 450)
        self.preview.setStyleSheet("background: #111; color: #ddd;")

        self.logs = QtWidgets.QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(2000)

        left = QtWidgets.QWidget()
        lf = QtWidgets.QFormLayout(left)
        lf.addRow("Input mode", self.input_mode)
        lf.addRow("Input file", self._hbox(self.video_path, self.btn_pick_video))
        lf.addRow("权重文件", self._hbox(self.model_path, self.btn_pick_model))
        lf.addRow("Device", self.device)
        lf.addRow("Algorithm mode", self.algo_mode)
        lf.addRow("imgsz", self.imgsz)
        lf.addRow("conf", self.conf)
        lf.addRow("iou", self.iou)
        lf.addRow("vid_stride", self.vid_stride)
        lf.addRow("统计/检测目标", self.stat_target)
        lf.addRow("自定义类别", self.stat_custom)
        lf.addRow("ROI margin L/T/R/B", self._hbox(self.roi_ml, self.roi_mt, self.roi_mr, self.roi_mb))
        lf.addRow(self.enable_line_count)
        lf.addRow("line_y_frac", self.line_y_frac)
        lf.addRow("SORT max_age", self.trk_max_age)
        lf.addRow("SORT min_hits", self.trk_min_hits)
        lf.addRow("SORT IoU(关联)", self.trk_iou)
        lf.addRow(self.save_output)
        lf.addRow("Output", self._hbox(self.output_path, self.btn_pick_output))
        lf.addRow(self._hbox(self.btn_start, self.btn_stop))
        lf.addRow(self.progress)
        lf.addRow(self.btn_open_out)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.addWidget(self.preview, 3)
        rv.addWidget(self.logs, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        self.btn_pick_video.clicked.connect(self.pick_video)
        self.btn_pick_model.clicked.connect(self.pick_model)
        self.btn_pick_output.clicked.connect(self.pick_output)
        self.btn_start.clicked.connect(self.start_run)
        self.btn_stop.clicked.connect(self.stop_run)
        self.btn_open_out.clicked.connect(self.open_output_folder)

    @staticmethod
    def _hbox(*widgets):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for it in widgets:
            lay.addWidget(it)
        return w

    def append_log(self, s: str):
        self.logs.appendPlainText(s)

    def pick_video(self):
        if self.input_mode.currentText() == "Image":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Pick image",
                "",
                "Image Files (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*.*)",
            )
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Pick video",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
            )
        if path:
            self.video_path.setText(path)

    def pick_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择权重文件",
            "",
            "Weights (*.pt *.pth *.onnx *.engine);;All Files (*.*)",
        )
        if path:
            self.model_path.setText(path)

    def pick_output(self):
        if self.input_mode.currentText() == "Image":
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Pick output",
                "result.jpg",
                "Image Files (*.jpg *.jpeg *.png *.bmp *.webp);;MP4 (*.mp4)",
            )
        else:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Pick output", "result.mp4", "MP4 (*.mp4)")
        if path:
            if self.input_mode.currentText() == "Video" and not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.output_path.setText(path)

    def set_running_ui(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.input_mode.setEnabled(not running)
        self.btn_pick_video.setEnabled(not running)
        self.btn_pick_model.setEnabled(not running)
        self.btn_pick_output.setEnabled(not running)
        self.save_output.setEnabled(not running)
        self.algo_mode.setEnabled(not running)
        self.stat_target.setEnabled(not running)
        self.stat_custom.setEnabled(not running and self.stat_target.currentData() == "custom")
        self.roi_ml.setEnabled(not running)
        self.roi_mt.setEnabled(not running)
        self.roi_mr.setEnabled(not running)
        self.roi_mb.setEnabled(not running)
        self.enable_line_count.setEnabled(not running)
        self.line_y_frac.setEnabled(not running)
        self.trk_max_age.setEnabled(not running)
        self.trk_min_hits.setEnabled(not running)
        self.trk_iou.setEnabled(not running)

    def start_run(self):
        if self.worker is not None:
            return

        src_path = self.video_path.text().strip()
        if not src_path:
            QtWidgets.QMessageBox.warning(self, "Missing input", "Please pick an input file.")
            return

        model = self.model_path.text().strip()
        if not model:
            self.pick_model()
            model = self.model_path.text().strip()
        if not model:
            QtWidgets.QMessageBox.warning(self, "缺少权重文件", "请先选择模型权重文件（.pt/.onnx/...）")
            return

        out_path = None
        final_image_output = None
        if self.save_output.isChecked():
            default_name = "result.jpg" if self.input_mode.currentText() == "Image" else "result.mp4"
            requested_out = self.output_path.text().strip() or str(Path.cwd() / default_name)
            if self.input_mode.currentText() == "Image" and _is_image_path(requested_out):
                # pipeline writes video; convert first frame to image after run
                tmp_dir = tempfile.mkdtemp(prefix="qt_img_out_")
                out_path = os.path.join(tmp_dir, "result.mp4")
                final_image_output = requested_out
            else:
                out_path = requested_out

        input_path_for_pipeline = src_path
        temp_single_frame_video = None
        if self.input_mode.currentText() == "Image":
            img = _imread_unicode(src_path)
            if img is None:
                QtWidgets.QMessageBox.warning(self, "Invalid image", "Cannot open selected image.")
                return
            h, w = img.shape[:2]
            temp_dir = tempfile.mkdtemp(prefix="qt_img_")
            temp_single_frame_video = os.path.join(temp_dir, "single_frame.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(temp_single_frame_video, fourcc, 1.0, (w, h))
            if not vw.isOpened():
                QtWidgets.QMessageBox.warning(self, "Video writer error", "Cannot create temp video from image.")
                return
            vw.write(img)
            vw.release()
            input_path_for_pipeline = temp_single_frame_video

        mode = str(self.algo_mode.currentData())
        enable_tracking = mode in ("yolo_track", "all")
        enable_counting = mode in ("yolo_count", "all")

        cfg = RunConfig(
            model_path=os.path.abspath(model),
            source_video=input_path_for_pipeline,
            out_path=out_path or str(Path.cwd() / "result.mp4"),
            device="cpu" if self.device.currentText() == "cpu" else 0,
            imgsz=int(self.imgsz.value()),
            conf=float(self.conf.value()),
            iou=float(self.iou.value()),
            classes=None,
            vid_stride=int(self.vid_stride.value()),
            arrow_scale=8.0,
            preview=False,
            enable_tracking=enable_tracking,
            enable_counting=enable_counting,
            max_age=int(self.trk_max_age.value()),
            min_hits=int(self.trk_min_hits.value()),
            iou_threshold=float(self.trk_iou.value()),
            only_person=False,
            stat_target=str(self.stat_target.currentData()),
            stat_custom_classes=str(self.stat_custom.text()).strip(),
            roi_margin_left=float(self.roi_ml.value()),
            roi_margin_top=float(self.roi_mt.value()),
            roi_margin_right=float(self.roi_mr.value()),
            roi_margin_bottom=float(self.roi_mb.value()),
            enable_line_count=bool(self.enable_line_count.isChecked()) and enable_tracking and enable_counting,
            line_y_frac=float(self.line_y_frac.value()),
            show_predictions=enable_tracking,
            show_trajectories=enable_tracking,
            show_overlay=True,
        )

        self.progress.setValue(0)
        self.append_log("Starting...")
        self.set_running_ui(True)
        self.btn_open_out.setEnabled(False)

        self.worker = InferenceWorker(cfg=cfg, out_path=out_path)
        # Keep temp image-video wrapper path for cleanup after run
        self.worker._temp_single_frame_video = temp_single_frame_video  # type: ignore[attr-defined]
        self.worker._final_image_output = final_image_output  # type: ignore[attr-defined]
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.log_line.connect(self.append_log)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished_ok.connect(self.on_finished_ok)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def stop_run(self):
        if self.worker is not None:
            self.append_log("Stopping...")
            self.worker.request_stop()

    def on_frame_ready(self, img: QtGui.QImage, read_frames: int, processed: int):
        pix = QtGui.QPixmap.fromImage(img)
        self.preview.setPixmap(pix.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.statusBar().showMessage(f"read={read_frames} processed={processed}")

    def on_finished_ok(self, out_path: str):
        self.append_log("Finished.")
        self.set_running_ui(False)
        final_image_out = getattr(self.worker, "_final_image_output", None) if self.worker is not None else None
        if out_path and final_image_out:
            if _save_first_frame_as_image(out_path, final_image_out):
                self.append_log(f"Saved image: {final_image_out}")
                self.last_out_path = final_image_out
                self.btn_open_out.setEnabled(True)
            else:
                self.append_log("WARN: failed to export image from result video.")
        self._cleanup_worker_temp()
        self.worker = None
        if out_path and not final_image_out:
            self.last_out_path = out_path
            self.btn_open_out.setEnabled(True)
            self.append_log(f"Saved: {out_path}")

    def on_failed(self, msg: str):
        self.append_log("FAILED:")
        self.append_log(msg)
        QtWidgets.QMessageBox.critical(self, "Run failed", msg)
        self.set_running_ui(False)
        self._cleanup_worker_temp()
        self.worker = None

    def _cleanup_worker_temp(self):
        if self.worker is None:
            return
        tmp = getattr(self.worker, "_temp_single_frame_video", None)
        if not tmp:
            tmp = None
        tmp_out = getattr(self.worker, "_cfg", None)
        try:
            if tmp and os.path.isfile(tmp):
                os.remove(tmp)
            if tmp:
                parent = os.path.dirname(tmp)
                if parent and os.path.isdir(parent):
                    os.rmdir(parent)
        except Exception:
            pass
        # cleanup temp output video used for image export
        try:
            if self.worker and getattr(self.worker, "out_path", None):
                op = str(getattr(self.worker, "out_path"))
                if "qt_img_out_" in op and os.path.isfile(op):
                    os.remove(op)
                    p = os.path.dirname(op)
                    if p and os.path.isdir(p):
                        os.rmdir(p)
        except Exception:
            pass

    def open_output_folder(self):
        path = self.last_out_path or self.output_path.text().strip()
        if not path:
            return
        folder = str(Path(path).resolve().parent)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    def closeEvent(self, event):  # noqa: N802
        if self.worker is not None:
            self.worker.request_stop()
            self.worker.wait(2000)
            self._cleanup_worker_temp()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1300, 780)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

