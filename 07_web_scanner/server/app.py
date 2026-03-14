#!/usr/bin/env python3
"""CCG Card ID — Web Scanner API Server.

FastAPI server that exposes card detection and identification as a REST API.
Serves the client/ SPA at /app.

Detection and gallery search are lazy-loaded on first use.

Usage:
    python 07_web_scanner/server/app.py [--host HOST] [--port PORT]
        [--detector DETECTOR] [--identifier IDENTIFIER]

    # Or via uvicorn directly:
    uvicorn app:app --reload --port 8000
    (Run from the 07_web_scanner/server/ directory.)
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Path bootstrap — project root and detector/build subdirs
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([
    str(ROOT),
    str(ROOT / "03_detector"),
    str(ROOT / "03_detector" / "detectors"),
    str(ROOT / "04_build" / "mobilevit_xxs"),
])

from ccg_card_id.config import cfg  # noqa: E402
from card_lookup import make_lookup, CardLookup  # noqa: E402
from search import GallerySearchManager  # noqa: E402

# Detectors are imported lazily inside _get_detector() to avoid importing
# PyTorch at startup if only Canny is used.

# ---------------------------------------------------------------------------
# CLI args — parsed once at module import time so uvicorn can also use them
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="CCG Card ID web scanner server")
_parser.add_argument("--host", default="127.0.0.1")
_parser.add_argument("--port", type=int, default=8000)
_parser.add_argument("--detector", default="canny",
                     help="Default detector name (e.g. 'canny', 'tinycornercnn_e50')")
_parser.add_argument("--identifier", default="phash_16x16",
                     help="Default identifier name (e.g. 'phash_16x16', 'arcface_illustration_id_e75')")
_parser.add_argument("--ssl", action="store_true",
                     help="Serve over HTTPS using a self-signed certificate. "
                          "Required when accessing from other devices on the LAN.")

# parse_known_args so uvicorn's own args don't cause errors
_args, _ = _parser.parse_known_args()

DEFAULT_DETECTOR = _args.detector
DEFAULT_IDENTIFIER = _args.identifier

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="CCG Card ID Scanner", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # local dev tool — all origins OK
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve client/ SPA at /app
_client_dir = Path(__file__).resolve().parents[1] / "client"
if _client_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_client_dir), html=True), name="client")

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_gallery_manager: GallerySearchManager | None = None
_lookup: CardLookup | None = None
_detector_cache: dict[str, Any] = {}
_available_detectors: list[dict] | None = None


def _get_gallery_manager() -> GallerySearchManager:
    global _gallery_manager
    if _gallery_manager is None:
        _gallery_manager = GallerySearchManager()
    return _gallery_manager


def _get_lookup() -> CardLookup:
    global _lookup
    if _lookup is None:
        _lookup = make_lookup()
    return _lookup


def _discover_detectors() -> list[dict]:
    """Build the list of available detector descriptors."""
    global _available_detectors
    if _available_detectors is not None:
        return _available_detectors

    detectors = [{"name": "canny", "label": "Canny edge detector"}]

    ckpt_dir = cfg.data_dir / "results" / "corner_detector_tiny"
    if ckpt_dir.exists():
        for ckpt in sorted(ckpt_dir.glob("epoch_*.pt")):
            # epoch_0050.pt → epoch number 50 → name tinycornercnn_e50
            stem = ckpt.stem  # "epoch_0050"
            try:
                epoch = int(stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            detectors.append({
                "name": f"tinycornercnn_e{epoch}",
                "label": f"Neural corner CNN (epoch {epoch})",
                "_ckpt": ckpt,
            })
        # Also include last.pt as a named option
        last = ckpt_dir / "last.pt"
        if last.exists():
            detectors.append({
                "name": "tinycornercnn_last",
                "label": "Neural corner CNN (last checkpoint)",
                "_ckpt": last,
            })

    _available_detectors = detectors
    return detectors


def _get_detector(name: str) -> Any:
    """Return (cached) detector instance for the given name.

    Raises HTTPException 404 if not found.
    """
    if name in _detector_cache:
        return _detector_cache[name]

    if name == "canny":
        from detectors.canny_poly import CannyPolyDetector  # noqa: PLC0415
        det = CannyPolyDetector()
        _detector_cache[name] = det
        return det

    # Neural CNN
    all_detectors = _discover_detectors()
    match = next((d for d in all_detectors if d["name"] == name), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Detector '{name}' not found")

    from detectors.tiny_corner_cnn.predict import NeuralCornerDetectorInference  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    det = NeuralCornerDetectorInference(match["_ckpt"], device=device)
    _detector_cache[name] = det
    return det


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _decode_image(b64: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG string to a BGR numpy array."""
    try:
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return bgr
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def _build_record_response(
    bgr: np.ndarray,
    detector_name: str,
    identifier_name: str,
) -> dict:
    """Run detection + identification on one image, return response dict."""
    # ---- Detection ----
    try:
        detector = _get_detector(detector_name)
        result = detector.detect(bgr)
    except HTTPException:
        raise
    except Exception as exc:
        return {
            "_status": {"code": 400, "text": f"Detection error: {exc}"},
            "card_present": False,
        }

    if not result.card_present or result.corners is None:
        return {
            "_status": {"code": 200, "text": "OK"},
            "card_present": False,
            "confidence": result.confidence,
            "detector_used": detector_name,
        }

    corners_list = result.corners.tolist()  # [[x,y], ...]

    # ---- Identification ----
    try:
        manager = _get_gallery_manager()
        searcher = manager.get(identifier_name)
        search_result = searcher.find(bgr, result.corners)
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        return {
            "_status": {"code": 400, "text": f"Identification error: {exc}"},
            "card_present": True,
            "corners": corners_list,
            "detector_used": detector_name,
        }

    if search_result is None:
        return {
            "_status": {"code": 503, "text": "Gallery not available"},
            "card_present": True,
            "corners": corners_list,
            "detector_used": detector_name,
        }

    # ---- Metadata lookup ----
    card_info = _get_lookup().get(search_result.card_id) or {}

    # Confidence: for pHash invert distance to a [0,1] score (lower distance =
    # higher confidence).  For cosine similarity use the score directly.
    if search_result.score_type == "hamming":
        # Max possible Hamming distance for this hash
        bits = 8 * int(len(np.packbits(
            np.zeros(16 * 16, dtype=np.uint8)  # rough upper bound via hash_size
        )))
        # Use a simpler heuristic: map distance 0→1.0, distance 64→0.0 for 16x16
        # The actual max bits in the stored hash determines the ceiling.
        confidence = max(0.0, 1.0 - search_result.score / 256.0)
    else:
        confidence = float(search_result.score)

    return {
        "_status": {"code": 200, "text": "OK"},
        "card_present": True,
        "corners": corners_list,
        "card_name": card_info.get("card_name"),
        "set_code": card_info.get("set_code"),
        "set_name": card_info.get("set_name"),
        "scryfall_id": search_result.card_id,
        "tcgplayer_id": card_info.get("tcgplayer_id"),
        "tcgplayer_price_usd": card_info.get("price_usd"),
        "confidence": round(confidence, 4),
        "identifier_used": identifier_name,
        "detector_used": detector_name,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/health")
def health() -> dict:
    manager = _get_gallery_manager()
    identifiers = manager.list_all_identifiers()
    detectors = _discover_detectors()
    gallery_loaded = any(
        manager._cache.get(i["name"]) is not None and
        getattr(manager._cache[i["name"]], "loaded", False)
        for i in identifiers
    )
    return {
        "status": "ok",
        "gallery_loaded": gallery_loaded,
        "detectors_available": len(detectors),
        "identifiers_available": len(identifiers),
    }


@app.get("/v1/detectors")
def list_detectors() -> dict:
    public_keys = {"name", "label"}
    return {
        "detectors": [
            {k: v for k, v in d.items() if k in public_keys}
            for d in _discover_detectors()
        ]
    }


@app.get("/v1/identifiers")
def list_identifiers() -> dict:
    return {"identifiers": _get_gallery_manager().list_all_identifiers()}


@app.get("/v1/defaults")
def defaults() -> dict:
    return {"detector": DEFAULT_DETECTOR, "identifier": DEFAULT_IDENTIFIER}


@app.post("/v1/identify")
async def identify(body: dict) -> dict:
    """Identify cards in one or more image records.

    Request body: {"records": [{"_base64": "...", "detector": "...", "identifier": "..."}]}
    """
    records_in = body.get("records")
    if not records_in or not isinstance(records_in, list):
        raise HTTPException(status_code=400, detail="'records' list is required")

    records_out = []
    for rec in records_in:
        # Accept both "_base64" (Ximilar style) and "base64" (plain)
        b64 = rec.get("_base64") or rec.get("base64")
        if not b64:
            records_out.append({
                "_status": {"code": 400, "text": "Missing '_base64' or 'base64' field"},
                "card_present": False,
            })
            continue

        detector_name = rec.get("detector") or DEFAULT_DETECTOR
        identifier_name = rec.get("identifier") or DEFAULT_IDENTIFIER

        try:
            bgr = _decode_image(b64)
            record_resp = _build_record_response(bgr, detector_name, identifier_name)
        except HTTPException as exc:
            record_resp = {
                "_status": {"code": exc.status_code, "text": exc.detail},
                "card_present": False,
            }
        except Exception as exc:
            record_resp = {
                "_status": {"code": 500, "text": f"Internal error: {exc}"},
                "card_present": False,
            }

        records_out.append(record_resp)

    return {
        "records": records_out,
        "_status": {"code": 200, "text": "OK"},
    }


# ---------------------------------------------------------------------------
# TLS helpers
# ---------------------------------------------------------------------------

def _ensure_self_signed_cert(cert_path: Path, key_path: Path) -> None:
    """Generate a self-signed certificate via openssl if it doesn't exist."""
    if cert_path.exists() and key_path.exists():
        return
    import subprocess
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating self-signed certificate → {cert_path}")
    subprocess.run(
        [
            "openssl", "req", "-x509",
            "-newkey", "rsa:2048",
            "-keyout", str(key_path),
            "-out", str(cert_path),
            "-days", "3650",
            "-nodes",
            "-subj", "/CN=ccg-scanner",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Certificate generated. Your browser will show a security warning —")
    print("click 'Advanced' → 'Accept the Risk' (Firefox) or 'Proceed' (Chrome).")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    _server_dir = Path(__file__).resolve().parent
    ssl_kwargs: dict = {}
    if _args.ssl:
        _cert = _server_dir / "ssl" / "cert.pem"
        _key  = _server_dir / "ssl" / "key.pem"
        _ensure_self_signed_cert(_cert, _key)
        ssl_kwargs = {"ssl_certfile": str(_cert), "ssl_keyfile": str(_key)}
        scheme = "https"
    else:
        scheme = "http"

    host_display = "localhost" if _args.host in ("127.0.0.1", "::1") else _args.host
    print(f"Scanner UI → {scheme}://{host_display}:{_args.port}/app")

    uvicorn.run(
        "app:app",
        host=_args.host,
        port=_args.port,
        reload=False,
        **ssl_kwargs,
    )
