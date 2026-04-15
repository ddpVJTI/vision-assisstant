"""
audio_engine.py — Thread-safe, queue-based TTS with scenario-aware narration.

Design:
  • A daemon thread owns the pyttsx3 engine (only that thread calls it).
  • The main/detection thread calls `speak_*` — fire-and-forget.
  • Noise reduction: same audio_key is suppressed for AUDIO_COOLDOWN seconds.
  • Audio queue is flushed before each new utterance to avoid stale speech.
  • Scenario-aware methods: speak_scenario(), speak_navigation(), speak_danger()
"""

import queue
import threading
import time

import pyttsx3

from config import (
    AUDIO_COOLDOWN_SECONDS, TTS_RATE, TTS_VOLUME,
    CONF_HIGH, CONF_MEDIUM,
)


class AudioEngine:
    def __init__(self):
        self._queue:        queue.Queue = queue.Queue(maxsize=4)
        self._last_spoken:  dict        = {}
        self._enabled:      bool        = True
        self._lock          = threading.Lock()

        self._thread = threading.Thread(target=self._worker, name="AudioWorker", daemon=True)
        self._thread.start()

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def speak(self, text: str, key: str | None = None, cooldown_override: float | None = None) -> None:
        """Queue *text* for speech if cooldown allows."""
        if not self._enabled:
            return
        cooldown = cooldown_override if cooldown_override is not None else AUDIO_COOLDOWN_SECONDS
        now = time.monotonic()
        with self._lock:
            if key and (now - self._last_spoken.get(key, 0.0)) < cooldown:
                return
            if key:
                self._last_spoken[key] = now
        self._flush()
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def speak_scenario(self, label: str, direction: str, dist_ft: float,
                       motion_scenario: str, conf_tier: str) -> None:
        """
        Scenario-aware obstacle alert. Handles confidence tiers and motion types.

        motion_scenario: 'approaching' | 'you_approaching' | 'both' | 'static' | 'unknown'
        conf_tier:       'high' | 'medium' | 'low'
        """
        if not self._enabled:
            return

        # Low confidence → suppress audio entirely
        if conf_tier == "low":
            return

        # Build label prefix based on confidence tier
        label_phrase = label if conf_tier == "high" else f"possible {label}"

        # Build motion phrase
        dist_str = f"{dist_ft:.0f} feet"
        if motion_scenario == "approaching":
            msg = f"{label_phrase} approaching, {direction}, {dist_str}!"
        elif motion_scenario == "you_approaching":
            msg = f"{label_phrase} ahead, {direction}, {dist_str}. You are approaching."
        elif motion_scenario == "both":
            msg = f"{label_phrase} closing fast, {direction}, {dist_str}!"
        else:
            # static or unknown
            msg = f"{label_phrase}, {direction}, {dist_str}."

        # Cooldown: shorter for approaching objects (more urgent → re-announce faster)
        if motion_scenario in ("approaching", "both"):
            cooldown = 1.5
        elif motion_scenario == "you_approaching":
            cooldown = 2.5
        else:
            cooldown = AUDIO_COOLDOWN_SECONDS

        key = f"SCENARIO|{label}|{direction}"
        self.speak(msg, key=key, cooldown_override=cooldown)

    def speak_danger(self, label: str, direction: str, dist_ft: float,
                     motion_scenario: str = "approaching") -> None:
        """High-priority danger warning — scenario-aware, loops every 1.5 seconds."""
        if motion_scenario == "both":
            msg = f"Alert! {label} closing very fast! {direction}, {dist_ft:.0f} feet!"
        elif motion_scenario == "you_approaching":
            msg = f"Warning! You are approaching {label}. {direction}, {dist_ft:.0f} feet!"
        else:
            msg = f"Alert! {label}, {direction}, {dist_ft:.0f} feet!"
        self.speak(msg, key=f"DANGER|{label}", cooldown_override=1.5)

    def speak_navigation(self, free_space: dict, obstacle_label: str | None = None) -> None:
        """
        Speak navigation cue based on free-space map.
        Only speaks when there's an obstacle to route around.

        free_space: {'left': bool, 'center': bool, 'right': bool}
        """
        if not self._enabled:
            return

        left    = free_space.get('left', True)
        center  = free_space.get('center', True)
        right   = free_space.get('right', True)

        # Only narrate navigation when center is blocked (otherwise no action needed)
        if center and left and right:
            return   # All clear — stay silent

        # Build navigation guidance
        if not center and left and right:
            nav = "Clear path: left and right."
        elif not center and left and not right:
            nav = "Clear path: left only."
        elif not center and not left and right:
            nav = "Clear path: right only."
        elif not left and not center and not right:
            nav = "No clear path. Stop."
        elif center and not left and not right:
            nav = "Path clear ahead."
        elif center and left and not right:
            nav = "Path clear ahead and left."
        elif center and not left and right:
            nav = "Path clear ahead and right."
        else:
            return

        key = f"NAV|{nav[:20]}"
        self.speak(nav, key=key, cooldown_override=3.0)

    def toggle(self) -> bool:
        """Toggle voice on/off; returns new state."""
        self._enabled = not self._enabled
        if not self._enabled:
            self._flush()
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = value
        if not value:
            self._flush()

    def shutdown(self) -> None:
        """Signal the worker to exit."""
        self._flush()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    # ── Private ──────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def _worker(self) -> None:
        """Runs in a dedicated thread; owns the TTS engine."""
        import platform
        is_windows = platform.system() == "Windows"
        
        # ── Windows Direct SAPI ──────────────────────────────────────────────
        if is_windows:
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                # Direct SAPI SpVoice
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                
                # Apply rate and volume from config (SAPI rate is -10 to 10)
                # TTS_RATE is usually ~150-200. SAPI default 0 is ~150.
                sapi_rate = (TTS_RATE - 150) // 10
                voice.Rate = max(-10, min(10, sapi_rate))
                voice.Volume = int(TTS_VOLUME * 100)

                while True:
                    try:
                        text = self._queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    if text is None: break
                    if self._enabled:
                        try:
                            # SVSFlagsAsync = 1 (Async)
                            voice.Speak(text, 0)
                        except Exception:
                            pass
                    self._queue.task_done()
                return # Exit worker if successful
            except Exception as e:
                print(f"[Audio] Direct SAPI failed, falling back to pyttsx3: {e}")

        # ── pyttsx3 Fallback ────────────────────────────────────────────────
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", TTS_RATE)
            engine.setProperty("volume", TTS_VOLUME)

            voices = engine.getProperty("voices")
            for v in voices:
                if "zira" in v.name.lower() or "helena" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break

            while True:
                try:
                    text = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if text is None: break
                if self._enabled:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception:
                        pass
                self._queue.task_done()
        except Exception as e:
            print(f"[Audio] pyttsx3 fallback failed: {e}")

