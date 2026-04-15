"""
streamlit_app.py — Smart Vision Assistant Web UI (v2)
Upgraded with: MiDaS depth, LK velocity, ThreatScore, Free-Space navigation,
               Confidence-weighted audio, and updated Threat Matrix sidebar.
"""
import cv2
import streamlit as st
import time
import numpy as np
import tempfile
import os
from PIL import Image

from detector import ObjectDetector
from audio_engine import AudioEngine
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    THREAT_W1_DISTANCE, THREAT_W2_VELOCITY,
    CONF_HIGH, CONF_MEDIUM,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Vision Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.threat-card {
    background: #161B22; border-radius: 8px;
    padding: 8px 12px; margin-bottom: 6px;
    border-left: 4px solid;
}
.threat-high   { border-color: #FF4444; }
.threat-medium { border-color: #FFA500; }
.threat-low    { border-color: #00D4FF; }
.nav-bar { display: flex; gap: 8px; margin-top: 8px; }
.nav-ok  { background:#003d00; color:#00FF88; padding:4px 12px;
           border-radius:20px; font-weight:bold; font-size:13px; }
.nav-blocked { background:#3d0000; color:#FF4444; padding:4px 12px;
               border-radius:20px; font-weight:bold; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_backend():
    detector = ObjectDetector()
    audio    = AudioEngine()
    return detector, audio

detector, audio = get_backend()

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    input_mode    = st.radio("Input Mode", ["Live Camera", "Upload Photo", "Upload Video"])
    voice_enabled = st.toggle("🔊 Voice Narration", value=True)
    audio.set_enabled(voice_enabled)

    st.markdown("---")
    st.subheader("Threat Sensitivity")
    sensitivity = st.slider("Detection Sensitivity", 0.3, 2.5, 1.0, 0.1,
                             help="Scales the critical danger distance threshold.")
    w1 = st.slider("W1 — Distance Weight", 0.0, 3.0, float(THREAT_W1_DISTANCE), 0.1,
                    help="How much raw proximity contributes to ThreatScore.")
    w2 = st.slider("W2 — Velocity Weight", 0.0, 3.0, float(THREAT_W2_VELOCITY), 0.1,
                    help="How much approach speed contributes to ThreatScore.")

    st.markdown("---")
    st.subheader("Audio Confidence Filter")
    st.caption(f"High (full alert): conf ≥ {CONF_HIGH}")
    st.caption(f"Medium (softened): conf ≥ {CONF_MEDIUM}")
    st.caption(f"Low (visual only): conf < {CONF_MEDIUM}")

    danger_only = st.checkbox("⚠ Only Alert on Danger Objects", value=False)

    st.markdown("---")

    uploaded_photo = None
    uploaded_video = None
    video_toggle   = False

    if input_mode == "Live Camera":
        camera_toggle = st.checkbox("▶ Start Camera", value=st.session_state.camera_running)
        if camera_toggle != st.session_state.camera_running:
            st.session_state.camera_running = camera_toggle
            st.rerun()
        st.caption("Check 'Start Camera' to begin.")
    elif input_mode == "Upload Photo":
        uploaded_photo = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        st.session_state.camera_running = False
    elif input_mode == "Upload Video":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            video_toggle = st.checkbox("▶ Play Uploaded Video")
        st.session_state.camera_running = False

# ── Main Layout ───────────────────────────────────────────────────────────────
st.title("🎯 Smart Vision Assistant")
st.markdown("Real-time obstacle detection, depth estimation, and velocity-aware threat ranking for the visually impaired. Powered by **YOLOv8 + MiDaS + Lucas-Kanade**.")

col_video, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("🏆 Threat Matrix")
    threat_container = st.empty()

    st.markdown("---")
    st.subheader("🧭 Path Status")
    nav_container = st.empty()

    st.markdown("---")
    st.subheader("📊 System Status")
    status_alert = st.empty()
    fps_metric   = st.empty()
    depth_metric = st.empty()


# ── Helper: Render Threat Matrix ──────────────────────────────────────────────
def render_threat_matrix(detections, free_space):
    """Render the threat leaderboard and navigation bar to the sidebar containers."""
    # Navigation bar
    l_cls = "nav-ok" if free_space.get("left", True) else "nav-blocked"
    c_cls = "nav-ok" if free_space.get("center", True) else "nav-blocked"
    r_cls = "nav-ok" if free_space.get("right", True) else "nav-blocked"
    l_lbl = "◀ Left ✓"  if free_space.get("left", True)   else "◀ Left ✗"
    c_lbl = "● Ahead ✓" if free_space.get("center", True) else "● Ahead ✗"
    r_lbl = "Right ▶ ✓" if free_space.get("right", True)  else "Right ▶ ✗"

    nav_container.markdown(
        f'<div class="nav-bar">'
        f'<span class="{l_cls}">{l_lbl}</span>'
        f'<span class="{c_cls}">{c_lbl}</span>'
        f'<span class="{r_cls}">{r_lbl}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Threat matrix
    if not detections:
        threat_container.info("No relevant objects detected")
        return

    html = ""
    for det in detections[:6]:   # top 6 threats
        if det.is_danger:
            card_cls = "threat-high"
            icon     = "🔴"
        elif det.conf_tier == "medium":
            card_cls = "threat-medium"
            icon     = "🟡"
        else:
            card_cls = "threat-low"
            icon     = "🔵"

        vel_str = ""
        if abs(det.rel_velocity_fps) > 0.3:
            arrow = "↑" if det.rel_velocity_fps > 0 else "↓"
            vel_str = f" | {arrow} {abs(det.rel_velocity_fps):.1f} ft/s"

        scenario_tag = {
            "approaching":    "⚡ Approaching",
            "you_approaching": "→ You Approaching",
            "both":           "🚨 Both Closing",
            "static":         "— Static",
            "unknown":        "",
        }.get(det.motion_scenario, "")

        conf_tag = {"high": "", "medium": " (possible)", "low": " (unconfirmed)"}.get(det.conf_tier, "")

        html += (
            f'<div class="threat-card {card_cls}">'
            f'<strong>{icon} {det.label.title()}{conf_tag}</strong><br>'
            f'<small>{det.direction} · {det.distance_ft:.1f} ft{vel_str}</small><br>'
            f'<small>{scenario_tag} · TS={det.threat_score:.2f}</small>'
            f'</div>'
        )
    threat_container.markdown(html, unsafe_allow_html=True)


# ── Helper: Process Frame ─────────────────────────────────────────────────────
def update_ui_and_audio(frame, sensitivity, w1_val, w2_val):
    # Temporarily patch weights from UI sliders (simple override via module)
    import config as _cfg
    _cfg.THREAT_W1_DISTANCE = w1_val
    _cfg.THREAT_W2_VELOCITY = w2_val

    annotated, detections, free_space = detector.process(frame, sensitivity)

    if danger_only:
        detections = [d for d in detections if d.is_danger]

    danger_dets = [d for d in detections if d.is_danger]

    # ── Audio: Danger first ───────────────────────────────────────────────────
    for det in danger_dets:
        audio.speak_danger(det.label, det.direction, det.distance_ft, det.motion_scenario)

    # ── Audio: Scenario narration for non-danger ──────────────────────────────
    for det in detections:
        if not det.is_danger:
            audio.speak_scenario(
                det.label, det.direction, det.distance_ft,
                det.motion_scenario, det.conf_tier,
            )

    # ── Audio: Navigation cue ─────────────────────────────────────────────────
    top_label = detections[0].label if detections else None
    audio.speak_navigation(free_space, obstacle_label=top_label)

    # ── UI: Status ────────────────────────────────────────────────────────────
    if danger_dets:
        names = ", ".join(d.label for d in danger_dets[:3])
        status_alert.error(f"⚠ OBSTACLE AHEAD: {names.upper()}")
    else:
        status_alert.success("🟢 Monitoring...")

    # ── UI: Threat matrix + nav ───────────────────────────────────────────────
    render_threat_matrix(detections, free_space)

    # ── Display annotated frame ───────────────────────────────────────────────
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    video_container.image(rgb, channels="RGB", use_container_width=True)


# ── Main Processing ───────────────────────────────────────────────────────────
with col_video:
    video_container = st.empty()

    if input_mode == "Live Camera":
        if st.session_state.camera_running:
            status_alert.success("🟢 Camera Active")
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            prev_time = time.time()
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    status_alert.error("⚠ Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1)
                update_ui_and_audio(frame, sensitivity, w1, w2)

                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-4)
                fps_metric.metric("FPS", f"{fps:.1f}")
                prev_time = curr_time

            cap.release()
            audio._flush()
        else:
            video_container.info("📷 Camera is OFF. Check 'Start Camera' in the sidebar.")
            threat_container.empty()
            nav_container.empty()
            status_alert.info("⏹ Application stopped.")
            fps_metric.empty()

    elif input_mode == "Upload Photo":
        if uploaded_photo is not None:
            status_alert.success("🖼️ Processing Photo")
            file_bytes = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
            frame      = cv2.imdecode(file_bytes, 1)
            update_ui_and_audio(frame, sensitivity, w1, w2)
            fps_metric.metric("FPS", "N/A")
            audio._flush()
        else:
            video_container.info("🖼️ Please upload an image using the sidebar.")
            threat_container.empty()
            nav_container.empty()
            status_alert.info("Waiting for photo...")
            fps_metric.empty()

    elif input_mode == "Upload Video":
        if uploaded_video is not None:
            if video_toggle:
                status_alert.success("🎥 Playing Uploaded Video")
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_video.read())
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                prev_time = time.time()
                while video_toggle and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        status_alert.info("✅ Video Finished.")
                        break
                    update_ui_and_audio(frame, sensitivity, w1, w2)
                    curr_time = time.time()
                    fps_metric.metric("FPS", f"{1.0 / max(curr_time - prev_time, 1e-4):.1f}")
                    prev_time = curr_time

                cap.release()
                audio._flush()
                os.remove(tfile.name)
            else:
                video_container.info("🎥 Video loaded. Check 'Play Uploaded Video'.")
                threat_container.empty()
                nav_container.empty()
                status_alert.info("Ready to play.")
                fps_metric.empty()
        else:
            video_container.info("🎥 Please upload a video using the sidebar.")
            threat_container.empty()
            nav_container.empty()
            status_alert.info("Waiting for video...")
            fps_metric.empty()
