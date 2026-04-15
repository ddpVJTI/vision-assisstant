"""
config.py — Centralized configuration for Smart Vision Assistant
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# YOLO Model
# ─────────────────────────────────────────────
YOLO_MODEL = os.path.join(BASE_DIR, "weights", "yolov8n.pt")  # Pretrained nano model
CONFIDENCE_THRESHOLD = 0.50        # Minimum detection confidence (increased to reduce noise)

# ─────────────────────────────────────────────
# Distance Estimation (bounding box area px²)
# ─────────────────────────────────────────────
DIST_VERY_CLOSE_THRESHOLD = 50_000  # px² → "Very Close"
DIST_NEAR_THRESHOLD       = 20_000  # px² → "Near" (else "Far")

# ─────────────────────────────────────────────
# Danger Object List
# ─────────────────────────────────────────────
DANGER_OBJECTS = {
    "person", "car", "truck", "bus", "motorcycle",
    "bicycle", "chair", "dog", "cat", "horse",
    "train", "traffic light", "fire hydrant", "stop sign",
}

# ─────────────────────────────────────────────
# Audio / Noise Reduction
# ─────────────────────────────────────────────
AUDIO_COOLDOWN_SECONDS = 3.0       # Minimum gap between same-object announcements
TTS_RATE               = 155       # Speech words-per-minute (lower = clearer)
TTS_VOLUME             = 1.0       # 0.0 – 1.0

# ─────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────
CAMERA_INDEX   = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

# ─────────────────────────────────────────────
# UI Theme Palette
# ─────────────────────────────────────────────
COLOR_BG           = "#0D1117"
COLOR_SURFACE      = "#161B22"
COLOR_SURFACE2     = "#21262D"
COLOR_BORDER       = "#30363D"
COLOR_ACCENT       = "#00D4FF"
COLOR_ACCENT_DIM   = "#005F73"
COLOR_DANGER       = "#FF4444"
COLOR_WARNING      = "#FFA500"
COLOR_SUCCESS      = "#00FF88"
COLOR_TEXT_PRIMARY = "#E6EDF3"
COLOR_TEXT_MUTED   = "#8B949E"

FONT_TITLE   = ("Helvetica", 20, "bold")
FONT_HEADING = ("Helvetica", 13, "bold")
FONT_BODY    = ("Helvetica", 11)
FONT_SMALL   = ("Helvetica", 9)
FONT_MONO    = ("Courier", 10)

# ─────────────────────────────────────────────
# MiDaS Depth Estimation
# ─────────────────────────────────────────────
MIDAS_MODEL_TYPE   = "MiDaS_small"   # Fastest variant; runs on GPU
MIDAS_FRAME_SKIP   = 2               # Run MiDaS every N frames to cap GPU load

# ─────────────────────────────────────────────
# ThreatScore Weights
# ─────────────────────────────────────────────
THREAT_W1_DISTANCE = 1.0    # Weight for proximity component (1/distance)
THREAT_W2_VELOCITY = 1.5    # Weight for relative approach velocity

# ─────────────────────────────────────────────
# Lucas-Kanade Optical Flow
# ─────────────────────────────────────────────
LK_BACKGROUND_POINTS = 50   # Random background points for consensus flow
LK_MAX_PYRAMID_LEVEL = 3    # LK pyramid depth (higher = better for fast motion)
LK_WIN_SIZE          = 21   # Window size for LK search

# ─────────────────────────────────────────────
# Free-Space / Navigable Corridor Detection
# ─────────────────────────────────────────────
FREE_SPACE_UPPER_ZONE_FRAC   = 0.40   # Check upper 40% of frame for obstacle spikes
FREE_SPACE_SPIKE_PERCENTILE  = 75     # Pixels brighter than 75th pct = suspiciously close
FREE_SPACE_MIN_CLEAR_FT      = 8.0   # Min calibrated clearance to call a corridor "clear"

# ─────────────────────────────────────────────
# Confidence Audio Tiers
# ─────────────────────────────────────────────
CONF_HIGH   = 0.85   # >= this → full alert audio
CONF_MEDIUM = 0.60   # >= this → softened "possible X" audio; below = audio suppressed
