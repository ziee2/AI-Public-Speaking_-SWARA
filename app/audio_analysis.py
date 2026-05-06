"""
Audio analysis module for SWARA public speaking evaluation.

Analyses performed:
  - tempo   : speech rate and long-pause detection
  - articulation : filler-word count and clarity score
  - profanity    : inappropriate language detection
  - keywords     : topic keyword coverage (optional)
  - structure    : content structure / reference-text similarity (optional)
"""

from __future__ import annotations

import re
import tempfile
import threading
import os
from typing import Optional

import librosa
import numpy as np

# ---------------------------------------------------------------------------
# Common regex for extracting words (including Latin-extended characters)
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z\u00C0-\u024F]+")

# ---------------------------------------------------------------------------
# Whisper model singleton – loaded once to avoid repeated expensive init
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                try:
                    from faster_whisper import WhisperModel  # type: ignore
                    _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
                except Exception:
                    _whisper_model = None
    return _whisper_model

# ---------------------------------------------------------------------------
# Filler words (Indonesian + common English fillers)
# ---------------------------------------------------------------------------
_FILLER_WORDS = {
    # Indonesian
    "anu", "ee", "emmm", "hmm", "eee", "eum", "emm",
    "gitu", "gitukan", "kan", "yaa", "ya", "yak", "nah",
    "nih", "tuh", "loh", "lah", "dong", "deh", "sih",
    "kayak", "kayaknya", "gue", "gua", "mungkin",
    "sebentar", "maksudnya", "maksud", "jadi", "terus",
    # English
    "um", "uh", "like", "so", "well", "you know",
    "basically", "literally", "actually", "right",
    "okay", "ok", "yeah", "er", "err",
}

# ---------------------------------------------------------------------------
# Profanity words (Indonesian + English)
# ---------------------------------------------------------------------------
_PROFANITY_WORDS = {
    # Indonesian
    "anjing", "babi", "bangsat", "bajingan", "kontol",
    "memek", "ngentot", "tolol", "brengsek", "sialan",
    "goblok", "idiot", "setan", "tai", "keparat",
    "kurang ajar", "jancok", "jancuk", "asu", "celeng",
    # English
    "fuck", "shit", "bitch", "asshole", "bastard",
    "damn", "crap", "dick", "pussy", "cock",
}

# ---------------------------------------------------------------------------
# Ideal speech rate range (words per minute)
# ---------------------------------------------------------------------------
_WPM_MIN_IDEAL = 100
_WPM_MAX_IDEAL = 160
_WPM_MIN_ACCEPTABLE = 70
_WPM_MAX_ACCEPTABLE = 200

_LONG_PAUSE_THRESHOLD_SEC = 3.0  # silence longer than this counts as a long pause


def _load_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load audio bytes into a numpy array using librosa."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
    finally:
        os.unlink(tmp_path)
    return y, sr


def _transcribe(audio_bytes: bytes) -> str:
    """
    Transcribe audio to text using faster-whisper (small model).
    Falls back to an empty string if the model is unavailable.
    """
    model = _get_whisper_model()
    if model is None:
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            segments, _ = model.transcribe(tmp_path, beam_size=5, language="id")
            text = " ".join(seg.text for seg in segments).strip()
        finally:
            os.unlink(tmp_path)
        return text
    except Exception:
        return ""


def _analyze_tempo(y: np.ndarray, sr: int, transcript: str) -> dict:
    """
    Analyse speech tempo (words-per-minute) and detect long pauses.

    Score:
      2 – ideal WPM (100-160) and no long pause
      1 – acceptable WPM (70-200) or has a minor issue
      0 – outside acceptable range or long pause present
    """
    duration_sec = librosa.get_duration(y=y, sr=sr)
    if duration_sec < 1e-3:
        return {"score": 0, "has_long_pause": False, "wpm": 0, "duration_sec": 0}

    # --- detect silences to find long pauses ---
    intervals = librosa.effects.split(y, top_db=35)  # voiced intervals
    has_long_pause = False
    prev_end = 0
    for start_sample, end_sample in intervals:
        gap_sec = (start_sample - prev_end) / sr
        if gap_sec >= _LONG_PAUSE_THRESHOLD_SEC:
            has_long_pause = True
            break
        prev_end = end_sample

    # --- words per minute ---
    word_count = len(transcript.split()) if transcript.strip() else 0
    wpm = (word_count / duration_sec) * 60 if duration_sec > 0 else 0

    # score
    if _WPM_MIN_IDEAL <= wpm <= _WPM_MAX_IDEAL and not has_long_pause:
        score = 2
    elif _WPM_MIN_ACCEPTABLE <= wpm <= _WPM_MAX_ACCEPTABLE:
        score = 1
    else:
        score = 0

    return {
        "score": score,
        "has_long_pause": has_long_pause,
        "wpm": round(wpm, 1),
        "duration_sec": round(duration_sec, 2),
    }


def _analyze_articulation(transcript: str) -> dict:
    """
    Detect filler words in the transcript and score articulation quality.

    Score:
      2 – 0 filler words
      1 – 1-3 filler words
      0 – 4+ filler words
    """
    if not transcript.strip():
        return {"score": 1, "filler_count": 0, "fillers_found": []}

    words = _WORD_RE.findall(transcript.lower())
    text_lower = transcript.lower()

    fillers_found: list[str] = []

    # single-word fillers
    for w in words:
        if w in _FILLER_WORDS:
            fillers_found.append(w)

    # multi-word fillers (e.g. "you know")
    for phrase in _FILLER_WORDS:
        if " " in phrase and phrase in text_lower:
            fillers_found.append(phrase)

    filler_count = len(fillers_found)

    if filler_count == 0:
        score = 2
    elif filler_count <= 3:
        score = 1
    else:
        score = 0

    return {
        "score": score,
        "filler_count": filler_count,
        "fillers_found": fillers_found,
    }


def _analyze_profanity(transcript: str) -> dict:
    """Detect profanity in the transcript."""
    if not transcript.strip():
        return {"has_profanity": False, "profanity_words": []}

    text_lower = transcript.lower()
    words = _WORD_RE.findall(text_lower)
    found: list[str] = []

    for w in words:
        if w in _PROFANITY_WORDS:
            found.append(w)
    for phrase in _PROFANITY_WORDS:
        if " " in phrase and phrase in text_lower:
            found.append(phrase)

    return {
        "has_profanity": len(found) > 0,
        "profanity_words": found,
    }


def _analyze_keywords(transcript: str, custom_keywords: list[str]) -> dict:
    """
    Measure how many of the provided keywords appear in the transcript.

    Score:
      2 – ≥ 70 % of keywords found
      1 – 30-69 % of keywords found
      0 – < 30 % of keywords found
    """
    if not custom_keywords:
        return {"score": 0, "keywords_found": [], "coverage": 0.0}

    text_lower = transcript.lower()
    found = [kw for kw in custom_keywords if kw.lower() in text_lower]
    coverage = len(found) / len(custom_keywords)

    if coverage >= 0.7:
        score = 2
    elif coverage >= 0.3:
        score = 1
    else:
        score = 0

    return {
        "score": score,
        "keywords_found": found,
        "coverage": round(coverage, 2),
    }


def _analyze_structure(transcript: str, reference_text: Optional[str]) -> dict:
    """
    Evaluate speech structure.

    If a reference_text is provided we compute a simple token-overlap
    similarity.  Otherwise we check whether the transcript has an
    identifiable opening, body, and closing.

    Score:
      2 – strong similarity / clear structure
      1 – partial similarity / partial structure
      0 – low similarity / unclear structure
    """
    if not transcript.strip():
        return {"score": 0, "similarity": 0.0}

    if reference_text:
        ref_words = set(_WORD_RE.findall(reference_text.lower()))
        trans_words = set(_WORD_RE.findall(transcript.lower()))
        union = ref_words | trans_words
        if not union:
            similarity = 0.0
        else:
            intersection = ref_words & trans_words
            similarity = len(intersection) / len(union)  # Jaccard

        if similarity >= 0.5:
            score = 2
        elif similarity >= 0.25:
            score = 1
        else:
            score = 0

        return {"score": score, "similarity": round(similarity, 2)}

    # Heuristic structure check (no reference text)
    sentences = [s.strip() for s in re.split(r"[.!?]", transcript) if s.strip()]
    n = len(sentences)
    if n >= 5:
        score = 2
    elif n >= 3:
        score = 1
    else:
        score = 0

    return {"score": score, "sentence_count": n}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_audio(
    audio_bytes: bytes,
    custom_topic: Optional[str] = None,
    reference_text: Optional[str] = None,
    custom_keywords: Optional[list] = None,
) -> dict:
    """
    Run all analyses on the provided audio bytes and return a combined result.

    Returns a dict with keys: tempo, articulation, profanity, keywords, structure.
    """
    y, sr = _load_audio(audio_bytes)
    transcript = _transcribe(audio_bytes)

    result: dict = {}

    result["tempo"] = _analyze_tempo(y, sr, transcript)
    result["articulation"] = _analyze_articulation(transcript)
    result["profanity"] = _analyze_profanity(transcript)

    if custom_keywords:
        result["keywords"] = _analyze_keywords(transcript, custom_keywords)
    elif custom_topic:
        # treat topic words as fallback keywords
        topic_words = [w for w in custom_topic.lower().split() if len(w) > 3]
        result["keywords"] = _analyze_keywords(transcript, topic_words)

    if reference_text or (transcript and len(transcript.split()) > 10):
        result["structure"] = _analyze_structure(transcript, reference_text)

    result["transcript"] = transcript

    return result
