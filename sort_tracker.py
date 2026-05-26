import numpy as np
from scipy.optimize import linear_sum_assignment

"""
本文件实现了 SORT（Simple Online and Realtime Tracking）多目标跟踪算法的核心逻辑。

关键点（与“匈牙利算法 + 卡尔曼滤波”对应）：
- 卡尔曼滤波：`KalmanBoxTracker`
  - 每个目标维护一个卡尔曼状态，完成“预测 predict() / 校正 update()”
  - 用于在检测偶尔丢失时保持轨迹连续，并输出下一时刻预测框（用于可视化蓝色预测框/箭头）

- 匈牙利算法：`associate_detections_to_trackers`
  - 通过 IoU 构造代价矩阵 cost = 1 - IoU
  - 使用 `linear_sum_assignment` 求解全局最优一对一匹配（即匈牙利算法）

注意：
- 这是一个轻量实现，未引入外观特征（ReID），主要依赖几何位置 + IoU 关联。
- 参数 `max_age / min_hits / iou_threshold` 会显著影响“抖动/漏检/误匹配”的表现。
"""


def _xyxy_to_z(box_xyxy: np.ndarray) -> np.ndarray:
    """
    将检测框从像素坐标系的 [x1, y1, x2, y2] 转成卡尔曼滤波使用的观测向量 z。

    z = [cx, cy, s, r]^T
    - cx, cy：框中心点
    - s：面积（w*h）
    - r：长宽比（w/h）

    说明：SORT 经典实现用 (cx, cy, area, ratio) 作为观测，便于在尺度变化时保持一定稳定性。
    """
    x1, y1, x2, y2 = box_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s = w * h
    r = w / h if h > 1e-6 else 0.0
    return np.array([[cx], [cy], [s], [r]], dtype=np.float32)


def _x_to_xyxy(x: np.ndarray) -> np.ndarray:
    """
    将卡尔曼状态向量 x 还原为像素坐标系的 [x1, y1, x2, y2]。

    这里的状态定义为：
    x = [cx, cy, s, r, vx, vy, vs]^T
    - 前 4 维：位置 + 尺度（与观测 z 一致）
    - 后 3 维：速度项（中心点速度 vx/vy、面积变化速度 vs）
    """
    cx, cy, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    if s <= 0 or r <= 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    w = np.sqrt(s * r)
    h = s / (w + 1e-6)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_batch(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """
    计算两组框之间的 IoU（Intersection over Union）。
    - a: (N,4)
    - b: (M,4)
    返回： (N, M)

    IoU 用于做“检测-轨迹”关联的相似度度量：IoU 越大，越可能属于同一目标。
    """
    if a_xyxy.size == 0 or b_xyxy.size == 0:
        return np.zeros((a_xyxy.shape[0], b_xyxy.shape[0]), dtype=np.float32)

    a = a_xyxy[:, None, :]  # (N,1,4)
    b = b_xyxy[None, :, :]  # (1,M,4)

    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(0.0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-6
    return (inter / union).astype(np.float32)


class KalmanBoxTracker:
    """
    单目标卡尔曼滤波跟踪器（SORT 风格）。

    1) 状态（state）
    x = [cx, cy, s, r, vx, vy, vs]^T

    2) 观测（measurement）
    z = [cx, cy, s, r]^T

    3) 线性卡尔曼滤波基本形式
    - 预测：
        x = F x
        P = F P F^T + Q
    - 更新：
        y = z - H x
        S = H P H^T + R
        K = P H^T S^{-1}
        x = x + K y
        P = (I - K H) P

    其中：
    - F：状态转移矩阵（这里假设 dt=1 的匀速模型）
    - H：观测矩阵（从状态中取出 [cx,cy,s,r]）
    - P：状态协方差（不确定度）
    - Q：过程噪声（模型误差）
    - R：观测噪声（检测框噪声）
    """

    _count = 0

    def __init__(self, box_xyxy: np.ndarray):
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        # 状态向量 x（7x1），初始化时只用检测框设置前 4 维，速度项先置 0
        self.x = np.zeros((7, 1), dtype=np.float32)
        self.x[:4] = _xyxy_to_z(box_xyxy)

        # 状态协方差 P：表示“我们对当前状态估计有多不确定”
        # 这里对速度部分赋予更大不确定性（乘以 100），因为初始速度更不可知
        self.P = np.eye(7, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0

        # Q/R：过程噪声与观测噪声（可视为经验参数）
        # - Q 越大：预测更“发散”，更依赖观测更新
        # - R 越大：观测更“嘈杂”，更依赖模型预测
        self.Q = np.eye(7, dtype=np.float32) * 0.01
        self.R = np.eye(4, dtype=np.float32) * 0.1

        # 状态转移矩阵 F（dt=1）
        # 位置由速度积分得到：cx += vx, cy += vy, s += vs
        self.F = np.eye(7, dtype=np.float32)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # 观测矩阵 H：从状态中取出 [cx, cy, s, r]
        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # 轨迹统计量（SORT 常用）
        # - time_since_update：距离上次匹配到检测的帧数（>0 表示“丢失中”）
        # - hits：累计命中次数
        # - hit_streak：连续命中次数（中断后清零）
        # - age：该轨迹存在的帧数
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        # 预测步骤（时间更新）：把状态推进到下一帧的“先验估计”
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        # 返回预测框（用于关联/可视化）
        return _x_to_xyxy(self.x)

    def update(self, box_xyxy: np.ndarray) -> None:
        # 更新步骤（量测更新）：用本帧检测框校正先验状态
        z = _xyxy_to_z(box_xyxy)  # (4,1)

        # 经典卡尔曼公式：y=创新/残差，S=残差协方差，K=卡尔曼增益
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(7, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self) -> np.ndarray:
        return _x_to_xyxy(self.x)

    def get_velocity(self) -> np.ndarray:
        """Return velocity (vx, vy) in image pixel units per frame (approx)."""
        return np.array([float(self.x[4]), float(self.x[5])], dtype=np.float32)


def associate_detections_to_trackers(
    dets_xyxy: np.ndarray,
    trks_xyxy: np.ndarray,
    iou_threshold: float,
):
    """
    关联“本帧检测框 dets”与“上一帧轨迹预测框 trks”。

    1) 计算 IoU 矩阵：iou[i,j] = IoU(det_i, trk_j)
    2) 转成代价矩阵：cost = 1 - iou
       - IoU 越大 cost 越小，表示越匹配
    3) 使用匈牙利算法（linear_sum_assignment）求全局最优一对一匹配
    4) 再做 IoU 阈值门控：低于阈值的不算匹配，转入未匹配集合

    返回：
    - matches: (K,2) 每行 [det_idx, trk_idx]
    - unmatched_dets: 未匹配到任何轨迹的检测框索引
    - unmatched_trks: 未匹配到任何检测框的轨迹索引
    """
    if trks_xyxy.size == 0:
        return np.empty((0, 2), dtype=np.int32), list(range(dets_xyxy.shape[0])), []

    iou = iou_batch(dets_xyxy, trks_xyxy)
    # 匈牙利算法求的是“最小总代价”，因此用 1-IoU 作为代价更直观
    cost = 1.0 - iou
    # row_ind/col_ind 给出一对一匹配：det[row_ind[k]] <-> trk[col_ind[k]]
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    unmatched_dets = set(range(dets_xyxy.shape[0]))
    unmatched_trks = set(range(trks_xyxy.shape[0]))

    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        # IoU 门控：匹配得太差就丢弃（避免强行配对导致 ID-switch）
        if iou[r, c] < iou_threshold:
            continue
        matches.append([r, c])
        unmatched_dets.discard(r)
        unmatched_trks.discard(c)

    return (
        np.asarray(matches, dtype=np.int32),
        sorted(unmatched_dets),
        sorted(unmatched_trks),
    )


class Sort:
    """
    SORT multi-object tracker.
    Input detections: ndarray (N,5) as [x1,y1,x2,y2,score]
    Output tracks: ndarray (M,5) as [x1,y1,x2,y2,track_id]
    """

    def __init__(self, max_age: int = 60, min_hits: int = 3, iou_threshold: float = 0.20):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(
        self,
        dets_xyxy_score: np.ndarray,
        return_predictions: bool = False,
        return_meta: bool = False,
    ):
        """
        主更新函数：输入当前帧检测框，输出当前帧“确认的轨迹”。

        核心流程：
        1) 对所有现存轨迹做卡尔曼预测，得到 trks_xyxy（预测框）
        2) 用匈牙利算法做 dets 与 trks 的一对一匹配（IoU 代价）
        3) 用匹配到的检测框更新对应轨迹（卡尔曼更新）
        4) 对未匹配的检测框创建新轨迹（初始化 KalmanBoxTracker）
        5) 对太久没更新的轨迹进行删除（time_since_update > max_age）
        6) 只输出“满足 min_hits 的确认轨迹”（减少刚生成轨迹的抖动）
        """
        self.frame_count += 1

        dets_xyxy_score = np.asarray(dets_xyxy_score, dtype=np.float32)
        if dets_xyxy_score.size == 0:
            dets_xyxy = dets_xyxy_score.reshape(0, 5)[:, :4]
        else:
            dets_xyxy = dets_xyxy_score[:, :4]

        # 1) 预测：对现存轨迹做卡尔曼时间更新，得到预测框 trks_xyxy
        trks_xyxy = []
        preds_with_id = []
        vel_with_id = []
        meta_with_id = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                to_del.append(i)
            trks_xyxy.append(pred)
            preds_with_id.append([pred[0], pred[1], pred[2], pred[3], float(trk.id)])
            v = trk.get_velocity()
            vel_with_id.append([float(v[0]), float(v[1]), float(trk.id)])
            meta_with_id.append([float(trk.time_since_update), float(trk.hit_streak), float(trk.hits), float(trk.id)])
        for i in reversed(to_del):
            self.trackers.pop(i)
        trks_xyxy = np.asarray(trks_xyxy, dtype=np.float32) if len(trks_xyxy) else np.zeros((0, 4), dtype=np.float32)
        preds_with_id = np.asarray(preds_with_id, dtype=np.float32) if len(preds_with_id) else np.zeros((0, 5), dtype=np.float32)
        vel_with_id = np.asarray(vel_with_id, dtype=np.float32) if len(vel_with_id) else np.zeros((0, 3), dtype=np.float32)
        meta_with_id = np.asarray(meta_with_id, dtype=np.float32) if len(meta_with_id) else np.zeros((0, 4), dtype=np.float32)

        # 2) 关联：匈牙利算法做全局最优匹配
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets_xyxy=dets_xyxy,
            trks_xyxy=trks_xyxy,
            iou_threshold=self.iou_threshold,
        )

        # 3) 更新：对匹配到的轨迹做卡尔曼量测更新
        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(dets_xyxy[det_idx])

        # 4) 新生：未匹配的检测框 -> 新建轨迹
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets_xyxy[det_idx]))

        # 5/6) 输出：只输出“确认轨迹”，并清理过期轨迹
        ret = []
        alive_trackers = []
        for trk in self.trackers:
            if trk.time_since_update < 1:
                if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                    x1, y1, x2, y2 = trk.get_state()
                    ret.append([x1, y1, x2, y2, float(trk.id)])
            # keep if not too old
            if trk.time_since_update <= self.max_age:
                alive_trackers.append(trk)

        self.trackers = alive_trackers
        tracks = np.asarray(ret, dtype=np.float32) if len(ret) else np.zeros((0, 5), dtype=np.float32)
        if return_predictions or return_meta:
            out = [tracks]
            if return_predictions:
                out.extend([preds_with_id, vel_with_id])
            if return_meta:
                out.append(meta_with_id)
            return tuple(out)
        return tracks
