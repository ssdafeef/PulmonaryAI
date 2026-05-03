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
import tensorflow as tf
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(PROJECT_ROOT / ".env")


CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MODEL_PORT = int(os.getenv("PORT", "8000"))


def _resolve_model_path() -> Path:
    configured_path = os.getenv("MODEL_PATH")
    if configured_path:
        candidate = Path(configured_path)
        if candidate.is_absolute():
            return candidate

        for base_dir in (BASE_DIR, PROJECT_ROOT, Path.cwd()):
            resolved = (base_dir / candidate).resolve()
            if resolved.exists():
                return resolved

        return (BASE_DIR / candidate).resolve()

    return (PROJECT_ROOT / "attention_resnet_covid_classifier.keras").resolve()


MODEL_PATH = _resolve_model_path()


app = FastAPI(title="COVID Classifier Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = None


def _parse_json_from_text(content: str) -> dict[str, Any] | None:
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


def _build_reconstructed_model() -> tf.keras.Model:
    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    output = tf.keras.layers.Dense(4, activation="softmax")(x)
    return tf.keras.Model(inputs=base_model.input, outputs=output)


def load_trained_model(path: Path) -> tf.keras.Model:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    rebuilt_model = _build_reconstructed_model()
    try:
        rebuilt_model.load_weights(str(path))
        print("SUCCESS: model architecture rebuilt and weights loaded")
        return rebuilt_model
    except Exception as rebuild_error:
        print(f"Rebuild path failed: {rebuild_error}")
        return tf.keras.models.load_model(str(path), compile=False, safe_mode=False)


def _load_model_once() -> tf.keras.Model:
    global MODEL
    if MODEL is None:
        MODEL = load_trained_model(MODEL_PATH)
    return MODEL


def _prepare_image(file_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    resized = image.resize(DEFAULT_IMAGE_SIZE)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch, resized


def _resolve_last_conv_layer(model: tf.keras.Model) -> str:
    preferred = "conv5_block3_out"
    try:
        model.get_layer(preferred)
        return preferred
    except Exception:
        pass

    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if isinstance(output_shape, tuple) and len(output_shape) == 4:
            return layer.name
    raise ValueError("Could not find a convolution layer for Grad-CAM")


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def get_gradcam_base64(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    original_image: Image.Image,
) -> str | None:
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_index = tf.argmax(preds[0])
            class_channel = preds[:, class_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = tf.reduce_sum(last_conv_layer_output * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        denominator = tf.reduce_max(heatmap)
        if float(denominator.numpy()) == 0.0:
            return None
        heatmap = heatmap / denominator
        heatmap = heatmap.numpy()

        heatmap_rgb = (_jet_colormap(heatmap) * 255).astype(np.uint8)
        overlay_base = np.asarray(original_image, dtype=np.float32)
        overlay = np.clip((heatmap_rgb.astype(np.float32) * 0.4) + (overlay_base * 0.6), 0, 255).astype(np.uint8)
        overlay_image = Image.fromarray(overlay)

        buffer = io.BytesIO()
        overlay_image.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"Grad-CAM generation failed: {exc}")
        return None


@app.on_event("startup")
def _startup() -> None:
    _load_model_once()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "covid-classifier-backend",
        "model_path": str(MODEL_PATH),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    model_loaded = MODEL is not None
    return {
        "status": "ok",
        "service": "covid-classifier-backend",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/check-llm")
def check_llm() -> dict[str, Any]:
    configured = bool(os.getenv("GEMINI_API_KEY"))
    return {
        "configured": configured,
        "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
        "model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        "provider": "gemini",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    model = _load_model_once()
    started = time.perf_counter()
    contents = await file.read()
    image_batch, resized_image = _prepare_image(contents)

    preds = model.predict(image_batch, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    try:
        last_conv_layer_name = _resolve_last_conv_layer(model)
        heatmap = get_gradcam_base64(image_batch, model, last_conv_layer_name, resized_image)
    except Exception as exc:
        print(f"Heatmap generation failed: {exc}")
        heatmap = None

    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "heatmap": heatmap,
        "latency_ms": latency_ms,
    }


@app.post("/llm-insight")
async def llm_insight(request: Request) -> Any:
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

    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_MODEL_PORT)
