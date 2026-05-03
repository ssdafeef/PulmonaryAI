"""
Triton Inference Client for COVID Classifier.
This client communicates with Triton Inference Server and provides HTTP endpoints for the frontend.
"""

import base64
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(PROJECT_ROOT / ".env")

# Configuration
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:8000")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "covid_classifier")
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_API_PORT = int(os.getenv("API_PORT", "8001"))


class TritonClient:
    """Client for communicating with Triton Inference Server."""

    def __init__(self, url: str, model_name: str):
        """Initialize Triton client."""
        self.url = url
        self.model_name = model_name
        try:
            self.client = InferenceServerClient(url=url)
            print(f"Connected to Triton at {url}")
        except Exception as e:
            print(f"Failed to connect to Triton at {url}: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if Triton server is ready."""
        try:
            return self.client.is_server_live() and self.client.is_model_ready(self.model_name)
        except Exception:
            return False

    def infer(self, image_bytes: bytes) -> dict[str, Any]:
        """Send inference request to Triton."""
        try:
            # Prepare input
            image_input = InferInput("image_bytes", [len(image_bytes)], "UINT8")
            image_input.set_data_from_numpy(np.frombuffer(image_bytes, dtype=np.uint8))

            # Prepare output
            output = InferRequestedOutput("result")

            # Send request
            response = self.client.infer(
                model_name=self.model_name,
                inputs=[image_input],
                outputs=[output],
            )

            # Parse response
            result_bytes = response.as_numpy("result")[0]
            result_json = json.loads(result_bytes.decode("utf-8"))
            return result_json
        except Exception as e:
            return {"error": str(e)}


# Initialize Triton client
triton_client = None

try:
    triton_client = TritonClient(TRITON_SERVER_URL, TRITON_MODEL_NAME)
except Exception as e:
    print(f"Warning: Could not initialize Triton client: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="COVID Classifier API (Triton Backend)",
    version="2.0.0",
    description="COVID Classifier using Triton Inference Server"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_json_from_text(content: str) -> dict[str, Any] | None:
    """Parse JSON from text content."""
    content = (content or "").strip()
    if not content:
        return None

    try:
        return json.loads(content)
    except Exception:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except Exception:
            return None
    return None


@app.on_event("startup")
def startup_event():
    """Check Triton connection on startup."""
    if triton_client:
        if triton_client.is_ready():
            print("✓ Triton server is ready")
        else:
            print("✗ Triton server is not ready - will retry on requests")


@app.get("/")
def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "status": "ok",
        "service": "covid-classifier-triton-api",
        "version": "2.0.0",
        "inference_engine": "triton",
        "triton_server": TRITON_SERVER_URL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint."""
    triton_ready = False
    if triton_client:
        triton_ready = triton_client.is_ready()

    return {
        "status": "ok",
        "service": "covid-classifier-triton-api",
        "triton_ready": triton_ready,
        "triton_server": TRITON_SERVER_URL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/check-llm")
def check_llm() -> dict[str, Any]:
    """Check LLM configuration."""
    configured = bool(os.getenv("GEMINI_API_KEY"))
    return {
        "configured": configured,
        "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
        "model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        "provider": "gemini",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """Predict COVID classification from medical image."""
    if not triton_client:
        return JSONResponse(
            status_code=503,
            content={"error": "Triton server not available"},
        )

    try:
        contents = await file.read()
        result = triton_client.infer(contents)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post("/llm-insight")
async def llm_insight(request: Request) -> Any:
    """Generate LLM-based insights for prediction."""
    body = await request.json()
    prediction = str(body.get("prediction", "Unknown"))

    try:
        confidence = float(body.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    patient_name = str(body.get("patientName", "Patient"))
    doctor_name = str(body.get("doctorName", "Doctor"))
    hospital_name = str(body.get("hospitalName", "Hospital"))
    heatmap_available = bool(body.get("heatmapAvailable", False))
    image_base64 = str(body.get("imageBase64", "")).strip()
    image_mime_type = str(body.get("imageMimeType", "image/jpeg")).strip() or "image/jpeg"

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    if base_url.endswith("/openai"):
        base_url = base_url[: -len("/openai")]
    model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    fallback_models = [
        item.strip()
        for item in os.getenv(
            "GEMINI_FALLBACK_MODELS",
            "gemini-flash-latest,gemini-2.0-flash,gemini-1.5-flash-8b,gemini-1.5-pro",
        ).split(",")
        if item.strip()
    ]
    timeout_seconds = float(os.getenv("GEMINI_TIMEOUT", "25"))

    start = time.perf_counter()

    if not api_key:
        return JSONResponse(
            status_code=500,
            content={
                "error": "GEMINI_API_KEY is not configured. A real Gemini LLM provider is required.",
                "required": True,
            },
        )

    system_prompt = (
        "You are a radiology AI copilot. Return only valid JSON with keys: "
        "impression (string, 2-4 sentences), narrative_paragraph (string, 4-6 sentences), "
        "evidence_points (array of 3 short strings), "
        "differentials (array of 3 strings), action_plan (array of 3 strings), "
        "patterns (array of up to 6 objects with keys name, confidence, finding, "
        "region where region has normalized x,y,w,h values in range 0.0-1.0), "
        "caution (string), uncertainty (Low|Medium|High)."
    )

    user_prompt = (
        f"Patient: {patient_name}\n"
        f"Doctor: {doctor_name}\n"
        f"Hospital: {hospital_name}\n"
        f"Prediction: {prediction}\n"
        f"Confidence: {round(confidence * 100, 2)}%\n"
        f"Heatmap available: {'yes' if heatmap_available else 'no'}\n"
        "Use professional clinical language. Keep narrative useful and clear for report inclusion. "
        "If image is provided, extract visible radiographic patterns and add region coordinates."
    )

    try:
        combined_prompt = (
            f"System instruction:\n{system_prompt}\n\n"
            f"User request:\n{user_prompt}\n\n"
            "Return only JSON."
        )
        parts_payload: list[dict[str, Any]] = [{"text": combined_prompt}]
        if image_base64:
            parts_payload.append(
                {
                    "inline_data": {
                        "mime_type": image_mime_type,
                        "data": image_base64,
                    }
                }
            )

        candidate_models = [model_name] + [item for item in fallback_models if item != model_name]
        raw_response = None
        used_model = model_name
        last_error = ""

        for candidate_model in candidate_models:
            endpoint_url = f"{base_url}/models/{candidate_model}:generateContent"
            response = requests.post(
                endpoint_url,
                headers={
                    "Content-Type": "application/json",
                    "X-goog-api-key": api_key,
                },
                json={
                    "contents": [{"parts": parts_payload}],
                    "generationConfig": {"temperature": 0.2},
                },
                timeout=timeout_seconds,
            )

            if response.ok:
                raw_response = response.json()
                used_model = candidate_model
                break

            last_error = f"Gemini API {response.status_code}: {response.text[:600]}"
            if response.status_code != 404:
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": "LLM request failed.",
                        "source": "remote-llm",
                        "model": candidate_model,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "warning": last_error,
                    },
                )

        if raw_response is None:
            return JSONResponse(
                status_code=502,
                content={
                    "error": "LLM request failed.",
                    "source": "remote-llm",
                    "model": model_name,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "warning": last_error or "No model produced a valid response.",
                },
            )

        parts = raw_response.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        content = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        parsed = _parse_json_from_text(content)
        if not parsed:
            return JSONResponse(
                status_code=502,
                content={
                    "error": "The LLM response was not valid JSON.",
                    "raw_response": content,
                },
            )

        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "source": "remote-llm",
            "model": used_model,
            "latency_ms": latency_ms,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "insight": parsed,
        }
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return JSONResponse(
            status_code=502,
            content={
                "error": "LLM request failed.",
                "source": "remote-llm",
                "model": model_name,
                "latency_ms": latency_ms,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "warning": str(exc),
            },
        )


if __name__ == "__main__":
    import uvicorn

    print(f"Starting API server on port {DEFAULT_API_PORT}")
    print(f"Connecting to Triton at {TRITON_SERVER_URL}")
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_API_PORT)
