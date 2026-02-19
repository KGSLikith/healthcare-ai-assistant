# app.py
# 🎙️ Streamlit + Mic Recorder → Faster-Whisper (Whisper large-v3)
# - Push-to-talk mic widget (streamlit-mic-recorder)
# - File upload (wav/mp3/m4a/mp4/etc)
# - Smart compute_type fallback (CPU/GPU autodetect; no crashes)
# - Segment timestamps + transcript download

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

import streamlit as st

# --- Audio + STT deps
import ctranslate2
from faster_whisper import WhisperModel

# Try to import mic recorder widget
try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False


# ======================
# Utility: device & compute_type handling
# ======================

def best_device() -> str:
    """Return 'cuda' if available in this CTranslate2 build, else 'cpu'."""
    try:
        return "cuda" if "cuda" in ctranslate2.get_supported_devices() else "cpu"
    except Exception:
        return "cpu"

def supported_types(device: str) -> List[str]:
    """Query supported compute types for a device; safe defaults if unknown."""
    try:
        return ctranslate2.get_supported_compute_types(device)
    except Exception:
        return ["int8", "float32"] if device == "cpu" else ["float16", "float32"]

def make_candidates(device: str, preferred: Optional[str]) -> List[str]:
    """
    Build a prioritized list of compute_types we’ll try, filtered by device capabilities.
    - On CPU: prefer int8 → float32
    - On CUDA: prefer int8_float16 → float16 → int8 → float32
    """
    caps = supported_types(device)
    if device == "cuda":
        wanted = ["int8_float16", "float16", "int8", "float32"]
    else:
        wanted = ["int8", "float32"]

    if preferred and preferred in wanted:
        wanted = [preferred] + [t for t in wanted if t != preferred]

    return [t for t in wanted if t in caps]

def friendly_caps_text() -> str:
    """String with brief capability summary to show in the UI."""
    devs = []
    try:
        devs = ctranslate2.get_supported_devices()
    except Exception:
        pass
    dev = "cuda" if "cuda" in devs else "cpu"
    caps_cpu = ", ".join(supported_types("cpu"))
    caps_cuda = ", ".join(supported_types("cuda")) if "cuda" in devs else "—"
    return f"Device: **{dev}** | CPU types: {caps_cpu} | CUDA types: {caps_cuda}"


# ======================
# Model loader with fallback (cached)
# ======================

@st.cache_resource(show_spinner=True)
def load_whisper_model(name: str, preferred: Optional[str]) -> Tuple[WhisperModel, str, List[str]]:
    """
    Try several compute_types until one works. Return (model, chosen_type, tried_list).
    """
    device = best_device()
    candidates = make_candidates(device, preferred)

    last_err = None
    tried = []
    for ct in candidates:
        tried.append(ct)
        try:
            m = WhisperModel(name, device=device, compute_type=ct)
            return m, ct, tried
        except Exception as e:
            last_err = e

    # Final attempt with library default (no explicit compute_type)
    tried.append("<default>")
    try:
        m = WhisperModel(name, device=device)
        return m, "<default>", tried
    except Exception as e:
        if last_err is None:
            last_err = e
        raise RuntimeError(
            f"Failed to initialize Whisper model. Tried compute_types={tried} on device={device}."
        ) from last_err


# ======================
# Streamlit UI
# ======================

st.set_page_config(page_title="Live Mic → Whisper (Faster-Whisper)", page_icon="🎙️", layout="centered")
st.title("🎙️ Live Mic → Whisper (Faster-Whisper)")
st.caption("Push-to-talk or upload a file. Fully local once the model downloads.")

with st.sidebar:
    st.header("Model settings")
    model_name = st.selectbox("Model size", ["large-v3", "medium", "small", "base"], index=0)
    compute_choice = st.selectbox(
        "Preferred precision",
        ["auto", "int8_float16", "float16", "int8", "float32"],
        index=0,
        help="If unsupported, the app will fall back automatically."
    )
    beam_size = st.slider("Beam size", 1, 10, 5)
    lang = st.text_input("Force language (e.g. en, hi, te). Leave blank for auto.", value="")
    vad = st.toggle("VAD filter (trim silences)", value=True)
    st.info("Tip: On CPU-only machines, **int8** will be chosen automatically.")

    st.caption(friendly_caps_text())

preferred = None if compute_choice == "auto" else compute_choice

with st.spinner(f"Loading model: {model_name} (this happens only once)…"):
    model, chosen_type, tried_list = load_whisper_model(model_name, preferred)

st.success(f"Loaded **{model_name}** with compute_type **{chosen_type}** "
           f"(candidates tried: {', '.join(tried_list)})")

st.subheader("1) Record a short clip")
if HAS_MIC:
    st.write("Click to start/stop. Then press **Transcribe** below.")
    rec = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop",
        just_once=False,
        use_container_width=True,
        format="wav",  # 16-bit PCM WAV
        key="mic",
    )
else:
    st.error("Mic widget missing. Install it: `pip install streamlit-mic-recorder`")
    rec = None

st.divider()
st.subheader("2) Or upload an audio/video file")
up = st.file_uploader(
    "Supported: wav, mp3, m4a, aac, flac, ogg, mp4, mkv",
    type=["wav", "mp3", "m4a", "aac", "flac", "ogg", "mp4", "mkv"]
)

col1, col2 = st.columns(2)
with col1:
    do_transcribe = st.button("🔎 Transcribe", use_container_width=True)
with col2:
    do_clear = st.button("🧹 Clear", use_container_width=True)

if do_clear:
    for k in ("last_text", "segments", "last_audio_path"):
        st.session_state.pop(k, None)
    st.experimental_rerun()


# ======================
# Helpers
# ======================

def save_bytes_to_temp(data: bytes, suffix: str = ".wav") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name

def persist_upload_to_temp(file) -> str:
    suffix = "".join(Path(file.name).suffixes) or ".wav"
    return save_bytes_to_temp(file.read(), suffix=suffix)

def do_whisper_transcribe(path_or_audio, language: Optional[str]) -> Tuple[str, List[dict], dict]:
    """
    Run Faster-Whisper transcription.
    Returns: (full_text, segments_dict, info_dict)
    segments_dict: list of {"start": float, "end": float, "text": str}
    """
    segments, info = model.transcribe(
        path_or_audio,
        language=(language or None),
        beam_size=beam_size,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    full_text = " ".join(s["text"] for s in segs).strip()
    return full_text, segs, {"language": info.language, "prob": info.language_probability}


# ======================
# Action
# ======================

if do_transcribe:
    input_path = None
    # Prefer fresh mic recording if available
    if rec and isinstance(rec, dict) and rec.get("bytes"):
        input_path = save_bytes_to_temp(rec["bytes"], suffix=".wav")
    elif up:
        input_path = persist_upload_to_temp(up)

    if not input_path:
        st.warning("Record something or upload a file first.")
    else:
        st.session_state["last_audio_path"] = input_path
        st.audio(input_path)
        with st.spinner("Transcribing…"):
            try:
                text, segs, info = do_whisper_transcribe(input_path, lang.strip() or None)
                st.session_state["last_text"] = text
                st.session_state["segments"] = segs
                st.success(f"Detected language: {info['language']} (p≈{info['prob']:.2f})")
            except Exception as e:
                st.exception(e)


# ======================
# Results
# ======================

if st.session_state.get("last_text"):
    st.subheader("Transcript")
    st.write(st.session_state["last_text"])

    # Download button
    st.download_button(
        "⬇️ Download transcript (.txt)",
        data=st.session_state["last_text"].encode("utf-8"),
        file_name="transcript.txt",
        mime="text/plain",
        use_container_width=True
    )

    if st.toggle("Show timestamps", value=False):
        st.subheader("Segments")
        for seg in st.session_state.get("segments", []):
            st.write(f"[{seg['start']:.2f} → {seg['end']:.2f}] {seg['text']}")
