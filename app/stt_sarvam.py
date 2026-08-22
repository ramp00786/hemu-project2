from __future__ import annotations

import os

import httpx


def _extract_transcript(payload: dict) -> str:
    # Support a few likely response shapes.
    for key in ("transcript", "text"):
        if isinstance(payload.get(key), str) and payload[key].strip():
            return payload[key].strip()

    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("transcript", "text"):
            if isinstance(data.get(key), str) and data[key].strip():
                return data[key].strip()

    results = payload.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            for key in ("transcript", "text"):
                if isinstance(first.get(key), str) and first[key].strip():
                    return first[key].strip()

    return ""


def _env(name: str) -> str:
    raw = os.getenv(name, "") or ""
    return raw.split("#", 1)[0].strip().strip('"').strip("'")


async def transcribe_audio(audio_bytes: bytes, filename: str, timeout_s: float = 30.0) -> str:
    api_key = _env("SARVAM_API_KEY")
    stt_url = _env("SARVAM_STT_URL") or "https://api.sarvam.ai/speech-to-text"

    if not api_key:
        raise RuntimeError("Missing SARVAM_API_KEY in environment.")

    # Sarvam Cloud uses subscription-key, not Bearer.
    headers = {"api-subscription-key": api_key}
    files = {"file": (filename or "audio.webm", audio_bytes, "application/octet-stream")}
    form = {
        "model": _env("SARVAM_STT_MODEL") or "saarika:v2.5",
        "language_code": _env("SARVAM_LANGUAGE") or "unknown",
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(stt_url, headers=headers, files=files, data=form)
        resp.raise_for_status()
        payload = resp.json()

    transcript = _extract_transcript(payload)
    if not transcript:
        raise RuntimeError("Sarvam response did not contain transcript text.")
    return transcript
