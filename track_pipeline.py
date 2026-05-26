from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from sort_tracker import KalmanBoxTracker, Sort

# COCO 常见车辆相关类别（用于「画面内车辆统计」预设；与具体权重 names 对齐时更准）
COCO_VEHICLE_CLASS_IDS: Tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck


@dataclass
class RunConfig:
    model_path: str
    source_video: str
    out_path: str
    device: str = "cpu"  # "cpu" or 0

    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    classes: Optional[list[int]] = None
    vid_stride: int = 1

    # viz toggles
    show_predictions: bool = True
    show_trajectories: bool = True
    arrow_scale: float = 8.0
    show_overlay: bool = True
    preview: bool = False  # True: show cv2 window while writing video
    preview_window_name: str = "YOLOv8 + SORT (Preview)"

    # trajectory controls
    traj_len: int = 50
    traj_keep_frames: Optional[int] = None  # None -> tracker.max_age + 5

    # tracker params（偏“少换 ID”：略放宽关联、允许更久无检测再删轨迹）
    enable_tracking: bool = True
    enable_counting: bool = True
    max_age: int = 60
    min_hits: int = 3
    iou_threshold: float = 0.20

    # ---------- 固定机位：ROI / 越线 / 画面内目标统计 ----------
    # 兼容旧参数：only_person=True 时等价于 stat_target="person"
    only_person: bool = False
    # 统计/检测目标：all | person | vehicle_coco | custom
    stat_target: str = "vehicle_coco"
    # custom 时生效，例如 "2,3,5,7"（车辆）或 "0"（人）
    stat_custom_classes: str = ""
    # ROI：用画面四边的「内缩比例」定义矩形检测区（0~0.49），用于屏蔽天空/远处边缘等；全 0 表示整幅画面
    roi_margin_left: float = 0.0
    roi_margin_top: float = 0.0
    roi_margin_right: float = 0.0
    roi_margin_bottom: float = 0.0
    # 越线计数：在画面高度上的归一化位置 line_y_frac∈(0,1)，画水平线；轨迹底边中心跨过线则计数
    enable_line_count: bool = False
    line_y_frac: float = 0.55


@dataclass
class RuntimeStats:
    processed: int = 0
    read_frames: int = 0
    dets: int = 0
    tracks: int = 0
    fps: float = 0.0
    yolo_ms: float = 0.0
    sort_ms: float = 0.0
    # 累计统计（便于 UI / 视频角标展示）
    total_dets: int = 0  # 全视频累计「检测框」个数（每帧 YOLO 框数之和）
    total_ids_ever: int = 0  # 曾出现在输出 tracks 中的不同 track_id 个数（历史去重）
    # 越线计数（固定机位）：以「线的一侧→另一侧」为一次，分两个方向累计
    line_in: int = 0
    line_out: int = 0
    # 画面内实时统计（由 stat_target 决定语义；车辆预设时为「车辆」）
    stat_label: str = ""  # 角标用短标签，如 车辆
    stat_in_roi_dets: int = 0  # ROI 内、且属于统计类别的检测框数（本帧）
    stat_in_roi_tracks: int = 0  # ROI 内、且归类为统计类别的轨迹数（本帧，按 track_id 去重）


ProgressCallback = Callable[[float, RuntimeStats], None]
FrameCallback = Callable[[np.ndarray, RuntimeStats], None]
StopCallback = Callable[[], bool]


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _roi_rect_xyxy(w: int, h: int, cfg: RunConfig) -> Optional[Tuple[int, int, int, int]]:
    """由四边内缩比例得到 ROI 矩形（像素）。全 0 表示不启用 ROI。"""
    ml = _clamp(float(cfg.roi_margin_left), 0.0, 0.49)
    mt = _clamp(float(cfg.roi_margin_top), 0.0, 0.49)
    mr = _clamp(float(cfg.roi_margin_right), 0.0, 0.49)
    mb = _clamp(float(cfg.roi_margin_bottom), 0.0, 0.49)
    if ml == 0.0 and mt == 0.0 and mr == 0.0 and mb == 0.0:
        return None
    x1 = int(round(w * ml))
    y1 = int(round(h * mt))
    x2 = int(round(w * (1.0 - mr)))
    y2 = int(round(h * (1.0 - mb)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2


def _foot_xy_xyxy(box: np.ndarray) -> Tuple[float, float]:
    """行人越线计数常用锚点：框底边中心（脚点近似）。"""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return (x1 + x2) * 0.5, y2


def _horizontal_line_side(px: float, py: float, y_line: float) -> int:
    """
    水平线 y=y_line 的左右侧判别（用有符号距离，避免除法）：
    返回值 >0 / <0 表示在线两侧；==0 表示在线上（数值误差时也可能出现）。
    """
    d = py - y_line
    eps = 1e-3
    if d > eps:
        return 1
    if d < -eps:
        return -1
    return 0


def _filter_dets_roi(
    dets: np.ndarray, det_classes: np.ndarray, roi: Optional[Tuple[int, int, int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    if roi is None or dets.size == 0:
        return dets, det_classes
    x1r, y1r, x2r, y2r = roi
    keep: List[int] = []
    for i in range(int(dets.shape[0])):
        fx, fy = _foot_xy_xyxy(dets[i])
        if x1r <= fx <= x2r and y1r <= fy <= y2r:
            keep.append(i)
    if not keep:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    idx = np.asarray(keep, dtype=np.int64)
    return dets[idx], det_classes[idx]


def _effective_stat_target(cfg: RunConfig) -> str:
    # 兼容旧参数
    if cfg.only_person:
        return "person"
    t = (cfg.stat_target or "all").strip().lower()
    if t not in {"all", "person", "vehicle_coco", "custom"}:
        return "all"
    return t


def _parse_int_list_csv(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _yolo_pred_classes(cfg: RunConfig) -> Optional[List[int]]:
    """
    Ultralytics predict 的 classes 参数：
    - None：不限制类别（但仍可在后处理里做统计过滤）
    - list[int]：只推理这些类（更省算力）
    """
    if cfg.classes is not None:
        return list(cfg.classes)

    target = _effective_stat_target(cfg)
    if target == "person":
        return [0]
    if target == "vehicle_coco":
        return list(COCO_VEHICLE_CLASS_IDS)
    if target == "custom":
        ids = _parse_int_list_csv(cfg.stat_custom_classes)
        return ids if ids else None
    return None


def _stat_class_set(cfg: RunConfig) -> Optional[set[int]]:
    """
    画面内统计用类别集合：
    - None 表示不限制（统计 all）
    """
    if cfg.classes is not None:
        return set(int(x) for x in cfg.classes)

    target = _effective_stat_target(cfg)
    if target == "person":
        return {0}
    if target == "vehicle_coco":
        return set(int(x) for x in COCO_VEHICLE_CLASS_IDS)
    if target == "custom":
        ids = _parse_int_list_csv(cfg.stat_custom_classes)
        return set(ids) if ids else None
    return None


def _stat_label_cn(cfg: RunConfig) -> str:
    if cfg.classes is not None:
        return "Class"
    target = _effective_stat_target(cfg)
    if target == "person":
        return "Person"
    if target == "vehicle_coco":
        return "Vehicle"
    if target == "custom":
        return "Custom"
    return "Object"


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    aw = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    bw = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = aw + bw - inter + 1e-6
    return float(inter / union)


def _assign_track_classes_by_iou(
    tracks: np.ndarray,
    dets: np.ndarray,
    det_classes: np.ndarray,
    iou_min: float = 0.15,
) -> dict[int, int]:
    """
    SORT 轨迹不带类别：用「轨迹框 vs 检测框」IoU 贪心匹配，给每个 track_id 估计一个类别。
    注意：这是工程近似；拥挤遮挡时可能偶发错配。
    """
    if tracks.size == 0 or dets.size == 0:
        return {}
    used = set()
    out: dict[int, int] = {}
    trk_list = tracks.tolist()
    det_list = list(range(int(dets.shape[0])))
    for trk in trk_list:
        x1, y1, x2, y2, tid = trk
        tid_i = int(tid)
        tb = np.array([x1, y1, x2, y2], dtype=np.float32)
        best_j = -1
        best_iou = 0.0
        for j in det_list:
            if j in used:
                continue
            iou = _iou_xyxy(tb, dets[j, :4])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_min:
            used.add(best_j)
            out[tid_i] = int(det_classes[best_j])
    return out


def process_video(
    cfg: RunConfig,
    on_progress: Optional[ProgressCallback] = None,
    on_frame: Optional[FrameCallback] = None,
    frame_every: int = 1,
    should_stop: Optional[StopCallback] = None,
) -> str:
    """
    Run YOLOv8 detection + SORT tracking and write an annotated output video.
    Returns output video path.
    """
    out_dir = os.path.dirname(cfg.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # tracker (optional)
    # Reset per-run ID counter so IDs do not keep increasing across runs.
    KalmanBoxTracker._count = 0
    tracker = Sort(max_age=cfg.max_age, min_hits=cfg.min_hits, iou_threshold=cfg.iou_threshold) if cfg.enable_tracking else None

    # load model (lazy import so Streamlit UI can start even if torch/cudnn fails)
    try:
        from ultralytics import YOLO  # local import
    except OSError as e:
        raise OSError(
            "Failed to import Ultralytics/torch (often WinError 1455: pagefile too small). "
            "Fix: increase Windows virtual memory (page file) or install CPU-only PyTorch."
        ) from e
    except Exception as e:
        raise RuntimeError("Failed to import Ultralytics/torch.") from e

    model = YOLO(cfg.model_path)
    names = getattr(model, "names", None) or {}

    if _effective_stat_target(cfg) == "custom" and cfg.classes is None:
        if not str(cfg.stat_custom_classes or "").strip():
            raise ValueError("stat_target=custom 时需要填写 stat_custom_classes（例如 2,3,5,7）")

    # open video
    cap = cv2.VideoCapture(cfg.source_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {cfg.source_video}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cfg.out_path, fourcc, fps_src / max(1, cfg.vid_stride), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {cfg.out_path}")

    # trajectory state
    traj: dict[int, list[tuple[int, int]]] = {}
    last_seen: dict[int, int] = {}

    # timing / fps
    t_last = time.perf_counter()
    fps_smooth = 0.0
    yolo_ms_smooth = 0.0
    sort_ms_smooth = 0.0
    alpha = 0.1

    stats = RuntimeStats()
    # 统计「历史上出现过的轨迹 ID」（只统计已输出到 tracks 的 id，与画面红框一致）
    seen_track_ids: set[int] = set()
    # 越线计数：记录每个 track_id 上一帧相对计数线的侧别（-1/0/+1）
    line_prev_side: dict[int, int] = {}
    # 轨迹类别估计：track_id -> 最近匹配到的检测类别（SORT 不存 cls，只能近似）
    track_last_cls: dict[int, int] = {}

    stat_set = _stat_class_set(cfg)
    stat_label = _stat_label_cn(cfg)

    # ROI disabled by user requirement: always use full frame.
    roi_rect = None
    y_line: Optional[float] = None
    if cfg.enable_line_count:
        y_line = float(h) * _clamp(float(cfg.line_y_frac), 0.05, 0.95)

    frame_idx = 0
    processed = 0
    stopped_by_user = False
    while True:
        if should_stop is not None and should_stop():
            # UI（例如 PyQt）请求停止时，走“优雅停止”：
            # - 跳出循环，释放 VideoCapture / VideoWriter
            # - 最终仍会触发一次 on_progress(1.0, stats)（用于 UI 收尾）
            stopped_by_user = True
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        stats.read_frames = frame_idx

        if cfg.vid_stride > 1 and (frame_idx - 1) % cfg.vid_stride != 0:
            continue

        processed += 1
        stats.processed = processed

        # YOLO 检测（逐帧推理）
        # 说明：
        # - Ultralytics 的 predict 会返回一组 results，这里取 results[0] 表示当前帧的检测结果
        # - 这里的输出框是 xyxy（左上右下）+ 置信度 + 类别
        t0 = time.perf_counter()
        pred_classes = _yolo_pred_classes(cfg)

        results = model.predict(
            source=frame_bgr,
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            classes=pred_classes,
            device=cfg.device,
            verbose=False,
        )
        yolo_ms = (time.perf_counter() - t0) * 1000.0
        r0 = results[0]

        dets = np.zeros((0, 5), dtype=np.float32)
        det_classes = np.zeros((0,), dtype=np.int32)
        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = r0.boxes.conf.cpu().numpy().astype(np.float32)
            cls = r0.boxes.cls.cpu().numpy().astype(np.int32)
            dets = np.concatenate([xyxy, scores[:, None]], axis=1)
            det_classes = cls

        # ROI disabled: keep detections from full frame.

        # SORT 跟踪（卡尔曼预测 + 匈牙利匹配）可选
        if cfg.enable_tracking and tracker is not None:
            # tracker.update 内部做的事：
            # - 对现有轨迹做卡尔曼 predict() 得到预测框
            # - 以 IoU 构建代价矩阵 cost=1-IoU，匈牙利算法做一对一匹配
            # - 用匹配到的检测框做卡尔曼 update()
            # - 维护轨迹生命周期（max_age/min_hits）
            t1 = time.perf_counter()
            tracks, preds, vels, meta = tracker.update(dets, return_predictions=True, return_meta=True)
            sort_ms = (time.perf_counter() - t1) * 1000.0
        else:
            tracks = np.zeros((0, 5), dtype=np.float32)
            preds = np.zeros((0, 5), dtype=np.float32)
            vels = np.zeros((0, 3), dtype=np.float32)
            meta = np.zeros((0, 4), dtype=np.float32)
            sort_ms = 0.0

        # 画面内实时统计：ROI 内「检测框数」+「轨迹数（按类别过滤）」
        if cfg.enable_counting:
            stats.stat_label = stat_label
            det_cnt = 0
            if dets.size:
                for c in det_classes.tolist():
                    ci = int(c)
                    if stat_set is None or ci in stat_set:
                        det_cnt += 1
            stats.stat_in_roi_dets = int(det_cnt)

            if tracks.size:
                m = _assign_track_classes_by_iou(tracks, dets, det_classes, iou_min=0.15)
                active_tids: set[int] = set()
                trk_cnt = 0
                for x1, y1, x2, y2, tid in tracks:
                    tid_i = int(tid)
                    active_tids.add(tid_i)
                    cls_i = m.get(tid_i, track_last_cls.get(tid_i, -1))
                    if cls_i >= 0:
                        track_last_cls[tid_i] = int(cls_i)
                    if cls_i >= 0 and (stat_set is None or int(cls_i) in stat_set):
                        trk_cnt += 1
                stats.stat_in_roi_tracks = int(trk_cnt)
                for old_id in list(track_last_cls.keys()):
                    if old_id not in active_tids:
                        track_last_cls.pop(old_id, None)
            else:
                stats.stat_in_roi_tracks = 0
                track_last_cls.clear()
        else:
            stats.stat_label = ""
            stats.stat_in_roi_dets = 0
            stats.stat_in_roi_tracks = 0
            track_last_cls.clear()

        # 全视频累计：检测框总数 + 出现过的轨迹 ID 种类数
        stats.total_dets += int(dets.shape[0])
        for _x1, _y1, _x2, _y2, tid in tracks:
            seen_track_ids.add(int(tid))
        stats.total_ids_ever = len(seen_track_ids)

        # 固定机位：越线计数（基于轨迹框底边中心跨水平线）
        if cfg.enable_counting and cfg.enable_tracking and cfg.enable_line_count and y_line is not None and tracks.size:
            active_ids: set[int] = set()
            for x1, y1, x2, y2, tid in tracks:
                tid_i = int(tid)
                active_ids.add(tid_i)
                fx, fy = _foot_xy_xyxy(np.array([x1, y1, x2, y2], dtype=np.float32))
                cur = _horizontal_line_side(float(fx), float(fy), float(y_line))
                prev = line_prev_side.get(tid_i, 0)
                if prev != 0 and cur != 0 and prev != cur:
                    # 从负侧跨到正侧记 in，反向记 out（方向可按现场把线反过来理解）
                    if prev < 0 and cur > 0:
                        stats.line_in += 1
                    elif prev > 0 and cur < 0:
                        stats.line_out += 1
                if cur != 0:
                    line_prev_side[tid_i] = cur
            # 清理丢失的 id，避免 SORT id 复用导致错误继承 prev_side
            for old_id in list(line_prev_side.keys()):
                if old_id not in active_ids:
                    line_prev_side.pop(old_id, None)

        # 性能统计（平滑）
        # 由于逐帧耗时会抖动，这里用指数滑动平均（EMA）让 UI 上显示更稳定
        yolo_ms_smooth = yolo_ms if yolo_ms_smooth == 0.0 else (1 - alpha) * yolo_ms_smooth + alpha * yolo_ms
        sort_ms_smooth = sort_ms if sort_ms_smooth == 0.0 else (1 - alpha) * sort_ms_smooth + alpha * sort_ms
        now = time.perf_counter()
        dt = now - t_last
        if dt > 1e-6:
            inst_fps = 1.0 / dt
            fps_smooth = inst_fps if fps_smooth == 0.0 else (1 - alpha) * fps_smooth + alpha * inst_fps
        t_last = now

        stats.yolo_ms = float(yolo_ms_smooth)
        stats.sort_ms = float(sort_ms_smooth)
        stats.fps = float(fps_smooth)
        stats.dets = int(dets.shape[0])
        stats.tracks = int(tracks.shape[0])

        # =========================
        # 渲染可视化（画到 vis 上）
        # =========================
        vis = frame_bgr.copy()

        # vels/meta 是 tracker.update 返回的辅助信息：
        # - vels: [vx, vy, track_id]，来自卡尔曼状态的速度项（像素/帧）
        # - meta: [time_since_update, hit_streak, hits, track_id]
        #   time_since_update 越大表示越久没匹配到检测；hit_streak 表示连续命中次数
        vel_map = {int(tid): (float(vx), float(vy)) for vx, vy, tid in (vels.tolist() if len(vels) else [])}
        meta_map: dict[int, tuple[float, float, float]] = {}
        if meta is not None and len(meta) > 0:
            for tsu, hs, hits, tid in meta.tolist():
                meta_map[int(tid)] = (float(tsu), float(hs), float(hits))

        # 1) 画检测框（颜色随置信度变化）
        if dets.size:
            det_cls_list = det_classes.tolist() if det_classes.size else [0] * int(dets.shape[0])
            for (x1, y1, x2, y2, s), c in zip(dets, det_cls_list):
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                sc = float(np.clip((float(s) - cfg.conf) / max(1e-6, (1.0 - cfg.conf)), 0.0, 1.0))
                color = (0, int(80 + 175 * sc), int(255 - 175 * sc))  # BGR
                label = f"{names.get(int(c), int(c))} {float(s):.2f}"
                cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color, 2)
                cv2.putText(vis, label, (x1i, max(0, y1i - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 2) 画跟踪框（红色）+ 速度 + 稳定度 q
        # q 的经验定义（0~1）：
        # - hit_streak 越大、time_since_update 越小 => q 越接近 1（更稳定）
        # - 刚出现/刚丢失/频繁断续 => q 趋近 0
        for x1, y1, x2, y2, tid in tracks:
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)

            vx, vy = vel_map.get(int(tid), (0.0, 0.0))
            speed_px_per_s = float(np.hypot(vx, vy) * (fps_src / max(1, cfg.vid_stride)))
            tsu, hs, cum_hits = meta_map.get(int(tid), (0.0, 0.0, 0.0))
            q = float(np.clip(hs / max(1.0, hs + tsu * 5.0), 0.0, 1.0))
            # cum_hits：SORT 内部该轨迹累计与检测匹配更新次数（卡尔曼 update 次数），可作「该目标累计被检测到」的近似
            text = f"ID {int(tid)} v={speed_px_per_s:.1f}px/s q={q:.2f} hits={int(cum_hits)}"
            cv2.putText(vis, text, (x1i, min(h - 5, y2i + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 轨迹点：记录每个 track 的中心点，用于画轨迹线
            cx = int((x1i + x2i) / 2)
            cy = int((y1i + y2i) / 2)
            key = int(tid)
            last_seen[key] = processed
            traj.setdefault(key, []).append((cx, cy))
            if len(traj[key]) > cfg.traj_len:
                traj[key] = traj[key][-cfg.traj_len :]

        # 3) 画“预测框 + 方向箭头”
        # preds 来自卡尔曼预测（即未用本帧检测更新之前的预测位置），能直观看到跟踪器的运动模型效果
        if cfg.enable_tracking and cfg.show_predictions and preds is not None and len(preds) > 0:
            # 预测框可视化里的“LOST”必须尽量贴近真实业务语义：
            # - 预测发生在本帧检测关联之前，因此 tsu 往往>=1，并不代表已经丢失
            # - 真实场景更合理的是：区分“未确认轨迹 / 短暂漂移 / 明显丢失(接近删除)”
            max_age = int(getattr(tracker, "max_age", 30))
            min_hits = int(getattr(tracker, "min_hits", 3))
            # 未确认：累计命中次数还没达到 min_hits（SORT 里 hits 是累计命中，不是连续 streak）
            init_hits = float(min_hits)
            # 漂移：连续若干帧没匹配到检测（仍可能恢复），阈值与 max_age 成比例
            drift_th = max(2, int(round(max_age * 0.25)))
            # 明显丢失：接近 SORT 的删除边界 max_age（再标 LOST 更符合“真的要没了”）
            lost_th = max(drift_th + 1, int(round(max_age * 0.85)))

            for x1, y1, x2, y2, tid in preds:
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                tsu, _hs, cum_hits = meta_map.get(int(tid), (0.0, 0.0, 0.0))
                # 注意：这里的 preds/meta 是在 SORT 的 predict() 阶段统计出来的。
                # predict() 会把每个轨迹的 time_since_update 先 +1，因此“本帧尚未用检测更新前”，
                # 绝大多数轨迹都会出现 tsu>=1 —— 这不等价于“已经跟踪丢失”，否则会把几乎所有目标都标成 LOST。
                #
                # 因此这里用“分级状态”，而不是单一阈值：
                # - TENT：轨迹还在建立（命中次数不足）
                # - PRED：正常预测（可能只是本帧暂时没匹配，或匹配发生在后续步骤）
                # - DRIFT：连续多帧未匹配，进入高风险漂移
                # - LOST：接近 max_age 仍未匹配，基本等价于“即将被删除的丢失”
                if cum_hits < init_hits:
                    state, color = "TENT", (0, 255, 255)  # 青色：未确认
                elif tsu < drift_th:
                    state, color = "PRED", (255, 0, 0)  # 红色：常规预测
                elif tsu < lost_th:
                    state, color = "DRIFT", (255, 128, 0)  # 橙色：漂移/高风险
                else:
                    state, color = "LOST", (0, 165, 255)  # 橙红偏色：明显丢失（BGR）

                cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color, 1)
                cx = int((x1i + x2i) / 2)
                cy = int((y1i + y2i) / 2)
                vx, vy = vel_map.get(int(tid), (0.0, 0.0))
                ex = int(cx + vx * cfg.arrow_scale)
                ey = int(cy + vy * cfg.arrow_scale)
                cv2.arrowedLine(vis, (cx, cy), (ex, ey), color, 2, tipLength=0.3)
                # 文本里带上 tsu，便于你对照视频理解“为什么进入 DRIFT/LOST”
                cv2.putText(
                    vis,
                    f"{state} {int(tid)} tsu={int(tsu)}",
                    (x1i, max(15, y1i - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        # trajectories + pruning
        if cfg.enable_tracking and cfg.show_trajectories:
            keep = cfg.traj_keep_frames if cfg.traj_keep_frames is not None else (int(getattr(tracker, "max_age", 30)) + 5)
            for tid in list(traj.keys()):
                last = last_seen.get(int(tid), None)
                if last is None or (processed - last) > keep:
                    traj.pop(int(tid), None)
                    last_seen.pop(int(tid), None)
            for tid, pts in traj.items():
                if len(pts) < 2:
                    continue
                if (processed - last_seen.get(int(tid), processed)) > keep:
                    continue
                for p0, p1 in zip(pts[:-1], pts[1:]):
                    cv2.line(vis, p0, p1, (0, 255, 255), 2)

        # 固定机位：画出 ROI 与计数线（不依赖 show_overlay，避免“关了角标就看不到线”）
        if cfg.enable_counting and roi_rect is not None:
            rx1, ry1, rx2, ry2 = roi_rect
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.putText(vis, "ROI", (rx1 + 5, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if cfg.enable_line_count and y_line is not None:
            yli = int(round(float(y_line)))
            cv2.line(vis, (0, yli), (w - 1, yli), (255, 0, 255), 2)
            cv2.putText(vis, "COUNT LINE", (10, max(20, yli - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # overlay
        if cfg.show_overlay:
            line1 = f"conf>={cfg.conf:.2f} iou={cfg.iou:.2f} imgsz={cfg.imgsz} stride={cfg.vid_stride} device={cfg.device}"
            line2 = f"FPS {fps_smooth:.1f} | YOLO {yolo_ms_smooth:.1f}ms | SORT {sort_ms_smooth:.1f}ms"
            line3 = f"Frame dets={stats.dets} tracks={stats.tracks}"
            cv2.putText(vis, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, line3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
            if cfg.enable_counting:
                line4 = (
                    f"Total det boxes={stats.total_dets} | Total track IDs={stats.total_ids_ever}"
                )
                line5 = (
                    f"Cross A->B (in)={stats.line_in} | Cross B->A (out)={stats.line_out} | net={stats.line_in - stats.line_out}"
                )
                line6 = (
                    f"In-frame {stats.stat_label} (tracks)={stats.stat_in_roi_tracks} | "
                    f"{stats.stat_label} (dets)={stats.stat_in_roi_dets}"
                )
                cv2.putText(vis, line4, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
                cv2.putText(vis, line5, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2)
                cv2.putText(vis, line6, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 220, 255), 2)
        else:
            # Keep one in-frame stat line even when overlay is off
            if cfg.enable_counting:
                mini = (
                    f"{stats.stat_label}: tracks={stats.stat_in_roi_tracks} dets={stats.stat_in_roi_dets} "
                    f"| dets={stats.dets} tracks={stats.tracks}"
                )
                cv2.putText(vis, mini, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        writer.write(vis)

        if on_frame is not None and frame_every > 0 and (processed % frame_every == 0):
            on_frame(vis, stats)

        if cfg.preview:
            cv2.imshow(cfg.preview_window_name, vis)
            # ESC to stop
            if cv2.waitKey(1) & 0xFF == 27:
                stopped_by_user = True
                break

        if on_progress is not None:
            p = (frame_idx / total) if total > 0 else 0.0
            p = float(np.clip(p, 0.0, 1.0))
            on_progress(p, stats)

    cap.release()
    writer.release()
    if cfg.preview:
        try:
            cv2.destroyWindow(cfg.preview_window_name)
        except Exception:
            cv2.destroyAllWindows()

    if on_progress is not None:
        on_progress(1.0, stats)
    if stopped_by_user and on_progress is not None:
        on_progress(1.0, stats)
    return cfg.out_path

