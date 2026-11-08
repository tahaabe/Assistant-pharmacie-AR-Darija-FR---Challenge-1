"""Speech-to-text. Two interchangeable backends, chosen with ASR_BACKEND:

  * groq            – Whisper large-v3(-turbo) via Groq's OpenAI-compatible API (needs GROQ_API_KEY).
                      Best quality, nothing to run locally – recommended on an 8 GB / no-GPU laptop.
  * faster_whisper  – local CPU inference (pip install faster-whisper). ASR_MODEL may be any
                      CTranslate2 Whisper checkpoint, incl. a Darija fine-tune you convert yourself.

Whisper has no "Darija" language code: it transcribes Darija as Arabic (or French when the
speaker switches), so we let it auto-detect and pass a domain prompt with drug/pharmacy words,
which noticeably helps spelling of brand names.
"""
from __future__ import annotations

import os
import tempfile

import httpx

ASR_BACKEND = os.getenv("ASR_BACKEND", "groq" if os.getenv("GROQ_API_KEY") else "faster_whisper")
ASR_MODEL = os.getenv("ASR_MODEL", "whisper-large-v3-turbo" if ASR_BACKEND == "groq" else "small")
DOMAIN_PROMPT = os.getenv(
    "ASR_PROMPT",
    "صيدلية، دوا، بغيت، واش كاين، شحال، pharmacie de garde, Doliprane, Augmentin, Spasfon, comprimé, boîte, sirop.",
)

_local_model = None


class ASRUnavailable(RuntimeError):
    pass


def transcribe(audio: bytes, filename: str = "audio.wav", language: str | None = None) -> dict:
    if ASR_BACKEND == "groq":
        return _groq(audio, filename, language)
    return _faster_whisper(audio, filename, language)


def _groq(audio: bytes, filename: str, language: str | None) -> dict:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ASRUnavailable("GROQ_API_KEY is not set")
    data = {"model": ASR_MODEL, "response_format": "verbose_json", "prompt": DOMAIN_PROMPT, "temperature": "0"}
    if language:
        data["language"] = language
    r = httpx.post("https://api.groq.com/openai/v1/audio/transcriptions",
                   headers={"Authorization": f"Bearer {key}"},
                   files={"file": (filename, audio)}, data=data, timeout=60)
    r.raise_for_status()
    j = r.json()
    return {"text": j.get("text", "").strip(), "language": j.get("language"), "backend": "groq", "model": ASR_MODEL,
            "duration_s": j.get("duration")}


def _faster_whisper(audio: bytes, filename: str, language: str | None) -> dict:
    global _local_model
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ASRUnavailable("No ASR backend: set GROQ_API_KEY or `pip install faster-whisper`") from e
    if _local_model is None:
        _local_model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(audio)
        path = fh.name
    try:
        segs, info = _local_model.transcribe(path, language=language, initial_prompt=DOMAIN_PROMPT,
                                             vad_filter=True, beam_size=5)
        text = " ".join(s.text.strip() for s in segs)
    finally:
        os.unlink(path)
    return {"text": text.strip(), "language": info.language, "language_prob": round(info.language_probability, 3),
            "backend": "faster_whisper", "model": ASR_MODEL, "duration_s": info.duration}
