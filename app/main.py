"""FastAPI service for the Arabic/Darija/French patient–pharmacy assistant.

Run:  uvicorn app.main:app --reload      ->  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from typing import Optional
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import asr, llm
from .data import DATA_DIR, canonical_city
from .dialog import Assistant

app = FastAPI(title="Pharmacy Assistant (AR / Darija / FR)", version="1.0.0",
              description="Text & voice requests → structured, CNOPS-grounded interpretation + pharmacy lookup.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
assistant = Assistant()
REQUESTS_LOG = DATA_DIR / "requests.jsonl"


class AnalyzeIn(BaseModel):
    text: str = Field(..., min_length=1, examples=["bghit jouj d les boites doliprane 1000 f rbat"])
    session_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class PharmacyRequestIn(BaseModel):
    request: dict
    pharmacy_id: str
    patient_contact: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "medications": len(assistant.cat.meds), "pharmacies": len(assistant.ph.items),
            "data_sources": {"medications": assistant.cat.source_file, "pharmacies": assistant.ph.source_file},
            "asr_backend": asr.ASR_BACKEND, "asr_model": asr.ASR_MODEL,
            "llm_fallback": {"enabled": llm.enabled(), "model": llm.MODEL, "threshold": llm.THRESHOLD}}


@app.post("/v1/analyze")
def analyze(body: AnalyzeIn):
    t0 = time.perf_counter()
    out = assistant.handle(body.text, body.session_id, body.lat, body.lon)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.post("/v1/analyze/voice")
async def analyze_voice(audio: UploadFile = File(...), session_id: Optional[str] = Form(None),
                        lat: Optional[float] = Form(None), lon: Optional[float] = Form(None),
                        language: Optional[str] = Form(None, description="optional ISO code hint: ar / fr")):
    t0 = time.perf_counter()
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    try:
        tr = asr.transcribe(raw, audio.filename or "audio.wav", language)
    except asr.ASRUnavailable as e:
        raise HTTPException(503, str(e))
    if not tr["text"]:
        raise HTTPException(422, "no speech detected")
    out = assistant.handle(tr["text"], session_id, lat, lon, channel="voice", transcript_meta=tr)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.get("/v1/medications/search")
def medications_search(q: str = Query(..., min_length=2), limit: int = 10):
    return assistant.cat.search(q, limit)


@app.get("/v1/medications/{code}")
def medication(code: str):
    m = next((m for m in assistant.cat.meds if m.code == code), None)
    if not m:
        raise HTTPException(404, "unknown code")
    return m.public() | {"equivalents": [g.public() for g in assistant.cat.generics_for(m)[:10]]}


@app.get("/v1/pharmacies")
def pharmacies(city: Optional[str] = None, garde: Optional[bool] = None, lat: Optional[float] = None,
               lon: Optional[float] = None, district: Optional[str] = None, limit: int = 10):
    return assistant.ph.search(canonical_city(city) if city else None, garde, lat, lon, district, limit)


@app.post("/v1/requests", status_code=201)
def submit_request(body: PharmacyRequestIn):
    """Downstream hand-off: the structured request is confirmed for one pharmacy and queued."""
    if not any(p.id == body.pharmacy_id for p in assistant.ph.items):
        raise HTTPException(404, "unknown pharmacy_id")
    rec = {"id": uuid.uuid4().hex[:10], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "pharmacy_id": body.pharmacy_id, "patient_contact": body.patient_contact,
           **body.request, "status": "sent_to_pharmacy"}
    with open(REQUESTS_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return (Path(__file__).parent / "demo.html").read_text(encoding="utf-8")
