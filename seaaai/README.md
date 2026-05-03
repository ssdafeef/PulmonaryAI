# PneumoAI

An AI-powered clinical diagnostic platform for chest X-ray analysis.

> **Disclaimer:** PneumoAI is intended for educational and research use only. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis.

## Overview

PneumoAI is a full-stack pneumonia detection system that analyzes chest X-ray images and generates both machine-readable and human-friendly diagnostic outputs. It combines deep learning, model serving, explainability, and LLM-based reporting into a single workflow.

The system uses a hybrid inference pipeline with:
- Two models served through NVIDIA Triton Inference Server
- One locally loaded model for fallback or supplementary analysis
- Grad-CAM heatmaps for visual explainability
- LLM-generated clinical explanations and summaries
- A React-based frontend for user interaction and report generation

## Key Features

- Multi-model inference for improved reliability
- Confidence-based ensemble voting
- Grad-CAM heatmap generation
- AI doctor chat and explanation generation
- PDF clinical report export
- Multi-language support
- Dockerized deployment for easy startup
- Triton-based model serving for scalable inference

## System Architecture

PneumoAI is organized as a layered inference system:

- The React frontend accepts chest X-ray uploads and displays results.
- The FastAPI backend handles preprocessing, orchestration, and API responses.
- NVIDIA Triton serves production inference models over HTTP and gRPC.
- A local Keras model is used for the third inference branch.
- The LLM layer generates patient-friendly or clinician-friendly explanations.

### Inference Flow

1. The user uploads a chest X-ray image.
2. The backend preprocesses the image to the expected input shape.
3. Triton is queried for the served models.
4. The local model performs its inference independently.
5. Predictions are combined using ensemble voting logic.
6. Grad-CAM heatmaps are created for explainability.
7. The LLM generates a clinical summary and explanation.
8. The frontend renders the final result for the user.

## Tech Stack

### Backend
- FastAPI
- TensorFlow
- NVIDIA Triton Client
- OpenCV
- Groq API / LLM integration
- python-dotenv

### Frontend
- React
- Vite
- Axios
- Framer Motion
- html2pdf.js

### Inference Infrastructure
- NVIDIA Triton Inference Server
- ONNX Runtime
- TensorFlow / Keras

### Deployment
- Docker
- Docker Compose

## Repository Structure

```text
PneumoAI/
├── backend/
│   ├── main.py
│   ├── utils.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── package.json
├── model_repository/
│   ├── densenet/
│   │   ├── 1/model.onnx
│   │   └── config.pbtxt
│   └── resnet/
│       ├── 1/model.onnx
│       └── config.pbtxt
├── models/
│   ├── best_model_1.keras
│   ├── best_model_2.keras
│   └── InceptionV3.keras
├── docker-compose.yml
├── run.sh
├── run.ps1
├── convert_to_onnx.py
├── fix_triton_config.py
├── run_instructions.md
└── *.ipynb
```

## Dataset

PneumoAI uses the Chest X-Ray Images (Pneumonia) dataset published by Paul Mooney on Kaggle.

Download:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Expected directory layout:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Models and Inference Pipeline

### Trained Models
- DenseNet121: served via Triton in ONNX format
- ResNet50: served via Triton in ONNX format
- InceptionV3: loaded locally in Keras format

### Output
All models perform binary classification:
- NORMAL
- PNEUMONIA

### Ensemble Logic
- If all models agree, the result is marked as high confidence.
- If predictions diverge, the result is marked for review.
- Final confidence is derived from the strongest model prediction.

## NVIDIA Triton Inference Server

PneumoAI uses Triton to serve the primary inference models in a scalable and production-ready way.

### Benefits
- Faster model serving
- Dynamic batching support
- HTTP and gRPC APIs
- Easier model versioning
- Better deployment isolation

### Model Repository Example

```text
model_repository/
├── densenet/
│   ├── 1/model.onnx
│   └── config.pbtxt
└── resnet/
    ├── 1/model.onnx
    └── config.pbtxt
```

## Docker Setup

### Prerequisites
- Docker Desktop installed and running
- At least 12 GB RAM recommended
- Internet access for initial image pull

### Quick Start

#### Windows
```powershell
.\run.ps1
```

#### Linux / macOS
```bash
chmod +x run.sh
bash run.sh
```

### Service URLs
- Frontend: http://localhost:5173
- Backend API: http://localhost:8080/docs
- Triton Health: http://localhost:8000/v2/models

## API Reference

### POST /predict
Upload a chest X-ray image for analysis.

Example:
```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@chest_xray.jpg"
```

Example response:
```json
{
  "label": "PNEUMONIA",
  "confidence": 0.9341,
  "agreement": "HIGH CONFIDENCE",
  "densenet": { "label": "PNEUMONIA", "confidence": 0.93 },
  "resnet": { "label": "PNEUMONIA", "confidence": 0.91 },
  "inception": { "label": "PNEUMONIA", "confidence": 0.94 },
  "heatmap": "<base64>",
  "original": "<base64>",
  "severity": "MODERATE",
  "quality_metrics": {
    "focus_score": 87.4
  },
  "explanation": "The AI analysis indicates..."
}
```

### POST /explain
Generate a clinical explanation from the prediction.

Example:
```bash
curl -X POST http://localhost:8080/explain \
  -H "Content-Type: application/json" \
  -d '{
    "diagnosis": "PNEUMONIA",
    "confidence": 0.93,
    "severity": "MODERATE",
    "language": "English"
  }'
```

## Local Development

### Backend
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

The frontend can be configured to point to the backend using `VITE_API_URL`.

## Troubleshooting

- If the backend exits unexpectedly, check model paths and environment variables.
- If Triton models show as unavailable, regenerate the Triton config and restart the server.
- If you see CORS issues, confirm the frontend is calling the correct backend port.
- If OpenCV fails with GUI-related errors, use the headless version.
- If the LLM feature is unavailable, verify the API key in `.env`.

## Author

PneumoAI was created to bridge the gap between deep learning research and practical clinical-style AI workflows.

Made with care for medical AI experimentation and explainable inference.
