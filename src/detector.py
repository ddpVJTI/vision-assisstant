"""
detector.py — YOLOv8 + MiDaS + Lucas-Kanade detection pipeline

Responsibilities:
  • Load and cache YOLOv8 model (GPU)
  • Load and cache MiDaS Small depth model (GPU)
  • Run optical flow via Lucas-Kanade PyrLK to estimate per-object relative velocity
  • Distinguish user-approaching-object vs object-approaching-user via background consensus
  • Compute YOLO-calibrated depth in feet for unknown objects
  • Compute ThreatScore = W1*(1/dist) + W2*relative_velocity
  • Compute free-space navigable corridor map (Left/Center/Right)
  • Return annotated frame, sorted detections, free-space map
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO

from config import (
    YOLO_MODEL, CONFIDENCE_THRESHOLD,
    DANGER_OBJECTS,
    COLOR_ACCENT, COLOR_DANGER, COLOR_WARNING,
    MIDAS_MODEL_TYPE, MIDAS_FRAME_SKIP,
    THREAT_W1_DISTANCE, THREAT_W2_VELOCITY,
    LK_BACKGROUND_POINTS, LK_MAX_PYRAMID_LEVEL, LK_WIN_SIZE,
    FREE_SPACE_UPPER_ZONE_FRAC, FREE_SPACE_SPIKE_PERCENTILE, FREE_SPACE_MIN_CLEAR_FT,
    CONF_HIGH, CONF_MEDIUM,
)
from indian_context import INDIAN_OBJECT_WIDTHS_FT, DEFAULT_FALLBACK_WIDTH_FT

# ── Constants ─────────────────────────────────────────────────────────────────
FOCAL_LENGTH_PX     = 600.0
CRITICAL_DISTANCE_FT = 3.0

# LK optical flow parameters
_LK_PARAMS = dict(
    winSize=(LK_WIN_SIZE, LK_WIN_SIZE),
    maxLevel=LK_MAX_PYRAMID_LEVEL,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

_GFT_PARAMS = dict(
    maxCorners=LK_BACKGROUND_POINTS,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7,
)


# ── Detection result dataclass ────────────────────────────────────────────────

class Detection:
    """Structured result for a single detected object."""
    __slots__ = (
        "label", "direction", "distance_ft", "distance_category",
        "is_danger", "conf", "conf_tier", "bbox",
        "is_in_path", "threat_score",
        "motion_scenario",   # 'approaching' | 'you_approaching' | 'both' | 'static' | 'unknown'
        "rel_velocity_fps",  # relative approach speed in ft/s
    )

    def __init__(self, label, direction, distance_ft, is_danger, conf,
                 bbox, is_in_path, threat_score, motion_scenario, rel_velocity_fps):
        self.label           = label
        self.direction       = direction
        self.distance_ft     = distance_ft
        self.is_danger       = is_danger
        self.conf            = conf
        self.bbox            = bbox
        self.is_in_path      = is_in_path
        self.threat_score    = threat_score
        self.motion_scenario = motion_scenario
        self.rel_velocity_fps = rel_velocity_fps

        # Distance category
        if distance_ft <= CRITICAL_DISTANCE_FT:
            self.distance_category = "Close"
        elif distance_ft <= CRITICAL_DISTANCE_FT + 5.0:
            self.distance_category = "Near"
        else:
            self.distance_category = "Far"

        # Confidence tier
        if conf >= CONF_HIGH:
            self.conf_tier = "high"
        elif conf >= CONF_MEDIUM:
            self.conf_tier = "medium"
        else:
            self.conf_tier = "low"

    @property
    def sentence(self) -> str:
        return f"{self.label}, {self.direction}, {self.distance_category}"

    @property
    def audio_key(self) -> str:
        return f"{self.label}|{self.direction}|{self.distance_category}"

    def __repr__(self):
        return f"<Detection {self.label} {self.direction} {self.distance_ft:.1f}ft ts={self.threat_score:.2f}>"


# ── Main Detector ─────────────────────────────────────────────────────────────

class ObjectDetector:
    """YOLOv8 + MiDaS + Lucas-Kanade integrated detector."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[Detector] Using device: {self.device}")

        # ── YOLOv8 ───────────────────────────────────────────────────────────
        print(f"[Detector] Loading YOLOv8 model: {YOLO_MODEL}")
        self.model      = YOLO(YOLO_MODEL)
        self.model.conf = CONFIDENCE_THRESHOLD
        num_classes = len(self.model.names)
        print(f"[Detector] ✅  YOLOv8 loaded — {num_classes} classes")

        # ── MiDaS ────────────────────────────────────────────────────────────
        print(f"[Detector] Loading MiDaS model: {MIDAS_MODEL_TYPE}")
        try:
            self._midas = torch.hub.load(
                "intel-isl/MiDaS", MIDAS_MODEL_TYPE, trust_repo=True
            ).to(self.device).eval()
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            self._midas_transform = midas_transforms.small_transform
            self._midas_available = True
            print(f"[Detector] ✅  MiDaS loaded on {self.device}")
        except Exception as e:
            print(f"[Detector] ⚠️  MiDaS failed to load: {e}. Falling back to geometry-only depth.")
            self._midas_available = False

        # ── State for stateful tracking ───────────────────────────────────────
        self._prev_gray          = None          # previous greyscale frame
        self._prev_centroids     = {}            # {track_id: (cx, cy)}
        self._prev_distances     = {}            # {track_id: dist_ft} for velocity calc
        self._depth_scale        = None          # MiDaS relative → feet scale factor
        self._frame_count        = 0
        self._last_depth_map     = None          # cached depth map between MiDaS skips

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray, sensitivity: float = 1.0):
        """
        Run detection on *frame*.

        Returns:
            annotated  : np.ndarray  — frame with overlays
            detections : list[Detection] — sorted by ThreatScore descending
            free_space : dict  — {'left': bool, 'center': bool, 'right': bool}
        """
        h, w = frame.shape[:2]
        self._frame_count += 1
        effective_critical = CRITICAL_DISTANCE_FT * sensitivity

        # ── Greyscale for optical flow ────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── MiDaS depth map (GPU, every N frames) ────────────────────────────
        if self._midas_available:
            if self._frame_count % MIDAS_FRAME_SKIP == 0 or self._last_depth_map is None:
                self._last_depth_map = self._run_midas(frame, h, w)
        depth_map = self._last_depth_map  # may be None if MiDaS not available

        # ── YOLO detection ────────────────────────────────────────────────────
        results      = self.model.track(frame, persist=True, verbose=False)[0]
        annotated    = frame.copy()

        # Walking path boundary
        path_left  = int(w * 0.20)
        path_right = int(w * 0.80)

        # ── Optical flow: background consensus ───────────────────────────────
        bg_flow_vec = self._compute_background_flow(gray)

        # ── Per-object processing ─────────────────────────────────────────────
        detections: list[Detection] = []
        new_centroids = {}
        new_distances = {}
        calibration_pairs = []   # (midas_val, geometric_ft) collected this frame

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf     = float(box.conf[0])
            class_id = int(box.cls[0])
            label    = self.model.names[class_id]
            track_id = int(box.id[0]) if box.id is not None else None

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ── Geometric distance (anchor) ───────────────────────────────────
            geo_dist_ft = self._get_distance_ft(x1, x2, label)

            # ── MiDaS calibration: collect anchor pair ────────────────────────
            if depth_map is not None and 0 <= cy < h and 0 <= cx < w:
                midas_val = float(depth_map[cy, cx])
                if midas_val > 0:
                    calibration_pairs.append((midas_val, geo_dist_ft))

            # ── Optical flow: per-object relative velocity ────────────────────
            rel_vel_fps, motion_scenario = self._compute_object_velocity(
                gray, track_id, cx, cy, geo_dist_ft, bg_flow_vec
            )

            # Store for next frame
            if track_id is not None:
                new_centroids[track_id] = (cx, cy)
                new_distances[track_id] = geo_dist_ft

            # ── Boundary & danger logic ───────────────────────────────────────
            is_in_path = not (x2 < path_left or x1 > path_right)
            is_danger  = is_in_path and (geo_dist_ft <= effective_critical)

            # ── ThreatScore = W1*(1/dist) + W2*max(0, relative_vel) ──────────
            dist_component = THREAT_W1_DISTANCE * (1.0 / max(geo_dist_ft, 0.5))
            vel_component  = THREAT_W2_VELOCITY * max(0.0, rel_vel_fps)
            threat_score   = dist_component + vel_component
            if not is_in_path:
                threat_score *= 0.25   # heavily penalize out-of-path objects

            det = Detection(
                label, self._get_9zone_direction(x1, y1, x2, y2, w, h),
                geo_dist_ft, is_danger, conf,
                (x1, y1, x2, y2), is_in_path, threat_score,
                motion_scenario, rel_vel_fps,
            )
            detections.append(det)
            self._draw_box(annotated, det)

        # ── Update depth scale calibration ────────────────────────────────────
        if calibration_pairs:
            # Median to avoid outlier influence
            midas_vals = np.array([p[0] for p in calibration_pairs])
            geo_fts    = np.array([p[1] for p in calibration_pairs])
            # scale_factor: depth_map_value * scale = feet
            self._depth_scale = float(np.median(geo_fts / (midas_vals + 1e-6)))

        # ── Sort descending by ThreatScore ────────────────────────────────────
        detections.sort(key=lambda d: d.threat_score, reverse=True)

        # ── Free-space detection ──────────────────────────────────────────────
        free_space = self._detect_free_space(depth_map, h, w)

        # ── Draw overlays ─────────────────────────────────────────────────────
        self._draw_region_lines(annotated, w, h, path_left, path_right)
        self._draw_free_space_overlay(annotated, free_space, w, h)

        # ── Update state ──────────────────────────────────────────────────────
        self._prev_gray      = gray
        self._prev_centroids = new_centroids
        self._prev_distances = new_distances

        # Filter: only in-path or danger objects for audio
        filtered = [d for d in detections if d.is_in_path or d.is_danger]
        return annotated, filtered, free_space

    # ── MiDaS ─────────────────────────────────────────────────────────────────

    def _run_midas(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """Run MiDaS Small on frame. Returns float32 depth map (h×w), higher = farther."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self._midas_transform(rgb).to(self.device)
            with torch.no_grad():
                raw = self._midas(input_tensor)
                # Interpolate back to frame size (1, 1, h, w)
                raw = F.interpolate(
                    raw.unsqueeze(1), size=(h, w),
                    mode="bicubic", align_corners=False,
                ).squeeze()
            depth = raw.cpu().numpy().astype(np.float32)
            # MiDaS outputs INVERSE depth (higher = closer). Invert so higher = farther.
            depth = 1.0 / (depth + 1e-6)
            # Normalize 0–255 for downstream use
            dmin, dmax = depth.min(), depth.max()
            if dmax > dmin:
                depth = (depth - dmin) / (dmax - dmin) * 255.0
            return depth
        except Exception as e:
            print(f"[MiDaS] Inference error: {e}")
            return None

    # ── Optical Flow ──────────────────────────────────────────────────────────

    def _compute_background_flow(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute mean optical flow over background feature points.
        Returns a (2,) vector [mean_vx, mean_vy] or zeros if not enough data.
        """
        if self._prev_gray is None:
            return np.zeros(2, dtype=np.float32)

        # Sample feature points from the previous frame
        pts = cv2.goodFeaturesToTrack(self._prev_gray, **_GFT_PARAMS)
        if pts is None or len(pts) < 5:
            return np.zeros(2, dtype=np.float32)

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, pts, None, **_LK_PARAMS
        )
        if new_pts is None:
            return np.zeros(2, dtype=np.float32)

        good_old = pts[status == 1]
        good_new = new_pts[status == 1]
        if len(good_old) < 3:
            return np.zeros(2, dtype=np.float32)

        flow = good_new - good_old           # shape (N, 2)
        return np.median(flow, axis=0)       # robust median

    def _compute_object_velocity(
        self, gray, track_id, cx, cy, geo_dist_ft, bg_flow_vec
    ):
        """
        Compute per-object relative velocity and classify motion scenario.

        Returns:
            rel_vel_fps      float     relative closing speed in ft/s (+ = closing)
            motion_scenario  str       'approaching' | 'you_approaching' | 'both' | 'static' | 'unknown'
        """
        # ── Physical velocity from distance change ────────────────────────────
        # (Stable and direct — use as primary velocity signal)
        if track_id is not None and track_id in self._prev_distances:
            prev_dist = self._prev_distances[track_id]
            # Positive means distance decreased = closing
            dist_change = prev_dist - geo_dist_ft
            # Approximate: assume ~15fps processing; convert Δft per frame → ft/s
            rel_vel_fps = dist_change * 15.0
        else:
            rel_vel_fps = 0.0

        # ── Motion scenario from background consensus ─────────────────────────
        bg_mag = float(np.linalg.norm(bg_flow_vec))

        if track_id is not None and track_id in self._prev_centroids:
            prev_cx, prev_cy = self._prev_centroids[track_id]
            obj_flow = np.array([cx - prev_cx, cy - prev_cy], dtype=np.float32)
            obj_mag  = float(np.linalg.norm(obj_flow))

            # Object flow RELATIVE to background (corrects for user motion)
            relative_flow     = obj_flow - bg_flow_vec
            relative_flow_mag = float(np.linalg.norm(relative_flow))

            # Thresholds (pixels per frame)
            BG_MOVING  = 4.0   # background moving = user walking
            OBJ_MOVING = 3.0   # object moving independently

            user_moving   = bg_mag > BG_MOVING
            object_moving = relative_flow_mag > OBJ_MOVING and rel_vel_fps > 0

            if object_moving and user_moving:
                motion_scenario = "both"
            elif object_moving and not user_moving:
                motion_scenario = "approaching"
            elif user_moving and not object_moving:
                motion_scenario = "you_approaching"
            else:
                motion_scenario = "static"
        else:
            motion_scenario = "unknown"

        return rel_vel_fps, motion_scenario

    # ── Free-Space Detection ──────────────────────────────────────────────────

    def _detect_free_space(self, depth_map, h: int, w: int) -> dict:
        """
        Analyse the upper zone of the depth map (higher = farther) for spike-free columns.
        Returns dict: {'left': bool, 'center': bool, 'right': bool}
        Also tries to give calibrated clearance if depth_scale available.
        """
        result = {'left': True, 'center': True, 'right': True,
                  'left_ft': None, 'center_ft': None, 'right_ft': None}

        if depth_map is None:
            return result   # Can't analyse; assume all clear (safe default)

        upper_h = int(h * FREE_SPACE_UPPER_ZONE_FRAC)
        zone    = depth_map[:upper_h, :]   # upper portion of frame

        # spike threshold: values BELOW this percentile (darker = closer) are obstacles
        # Remember: after our inversion, higher = farther, so low values = close objects
        spike_threshold = np.percentile(depth_map, 100 - FREE_SPACE_SPIKE_PERCENTILE)

        cols = {
            'left':   zone[:, :w//3],
            'center': zone[:, w//3:2*w//3],
            'right':  zone[:, 2*w//3:],
        }

        for name, col_data in cols.items():
            median_val = float(np.median(col_data))
            # A spike means many pixels are above the threshold (unusually close objects)
            spike_ratio = float(np.mean(col_data < spike_threshold))

            # If > 15% of upper-zone pixels in this column are "suspiciously close" → blocked
            result[name] = spike_ratio < 0.15

            # Calibrated clearance in feet
            if self._depth_scale is not None:
                clearance_ft = median_val / 255.0 * self._depth_scale * 15.0
                result[f'{name}_ft'] = round(clearance_ft, 1)

        return result

    # ── Distance ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_distance_ft(x1: int, x2: int, label: str) -> float:
        """Triangle-similarity distance using Indian object widths."""
        real_width  = INDIAN_OBJECT_WIDTHS_FT.get(label, DEFAULT_FALLBACK_WIDTH_FT)
        pixel_width = x2 - x1
        if pixel_width == 0:
            return 999.0
        return float((real_width * FOCAL_LENGTH_PX) / pixel_width)

    # ── Direction ─────────────────────────────────────────────────────────────

    @staticmethod
    def _get_9zone_direction(x1, y1, x2, y2, width, height) -> str:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        hz = "Left" if cx < width // 3 else ("Center" if cx < 2 * width // 3 else "Right")
        vt = "High" if cy < height // 3 else ("Mid" if cy < 2 * height // 3 else "Low")
        if hz == "Center" and vt == "Mid":
            return "Dead-Center"
        return f"{vt}-{hz}"

    # ── Drawing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_box(frame: np.ndarray, det: 'Detection'):
        x1, y1, x2, y2 = det.bbox

        # Color: red for danger, amber for medium conf, cyan for safe
        if det.is_danger:
            color = (0, 68, 255)
        elif det.conf_tier == "medium":
            color = (0, 165, 255)
        else:
            color = (0, 212, 255)

        # Glow effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 3)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Main box (dotted for low-confidence)
        if det.conf_tier == "low":
            # Draw dashed rectangle manually
            for i in range(x1, x2, 12):
                cv2.line(frame, (i, y1), (min(i + 6, x2), y1), color, 1)
                cv2.line(frame, (i, y2), (min(i + 6, x2), y2), color, 1)
            for i in range(y1, y2, 12):
                cv2.line(frame, (x1, i), (x1, min(i + 6, y2)), color, 1)
                cv2.line(frame, (x2, i), (x2, min(i + 6, y2)), color, 1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        corner, thick = 14, 3
        for cxx, cyy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            fx = 1 if cxx == x1 else -1
            fy = 1 if cyy == y1 else -1
            cv2.line(frame, (cxx, cyy), (cxx + fx * corner, cyy), color, thick)
            cv2.line(frame, (cxx, cyy), (cxx, cyy + fy * corner), color, thick)

        # Label: include motion scenario icon and velocity
        vel_str = f" {det.rel_velocity_fps:+.1f}ft/s" if abs(det.rel_velocity_fps) > 0.3 else ""
        conf_prefix = "?" if det.conf_tier == "medium" else ("??" if det.conf_tier == "low" else "")
        label_text = f"  {conf_prefix}{det.label} | {det.distance_ft:.1f}ft{vel_str}  "
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        pill_y1 = max(y1 - th - 10, 0)
        pill_y2 = max(y1 - 2, th + 4)
        cv2.rectangle(frame, (x1, pill_y1), (x1 + tw + 4, pill_y2), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, pill_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

        # Danger flash overlay
        if det.is_danger:
            dg = frame.copy()
            cv2.rectangle(dg, (x1, y1), (x2, y2), (0, 0, 200), -1)
            cv2.addWeighted(dg, 0.18, frame, 0.82, 0, frame)

    @staticmethod
    def _draw_region_lines(frame, w, h, path_left, path_right):
        alpha   = 0.35
        overlay = frame.copy()
        cv2.line(overlay, (0, h // 3), (w, h // 3), (100, 100, 100), 1)
        cv2.line(overlay, (0, 2 * h // 3), (w, 2 * h // 3), (100, 100, 100), 1)
        cv2.rectangle(overlay, (path_left, 0), (path_right, h), (0, 255, 0), 2)
        cv2.putText(overlay, "WALKING PATH", (path_left + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def _draw_free_space_overlay(frame, free_space: dict, w, h):
        """Draw semi-transparent green/red columns at the top strip."""
        strip_h = 12
        thirds  = w // 3
        mapping = [('left', 0, thirds), ('center', thirds, 2 * thirds), ('right', 2 * thirds, w)]
        overlay = frame.copy()
        for name, x_start, x_end in mapping:
            color = (0, 200, 0) if free_space.get(name, True) else (0, 0, 200)
            cv2.rectangle(overlay, (x_start, 0), (x_end, strip_h), color, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
