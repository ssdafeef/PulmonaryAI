# COVID-19 Detection System

An AI-powered clinical diagnostic platform for COVID-19 detection using chest X-ray analysis.

> **Disclaimer:** This COVID-19 Detection System is intended for educational and research use only. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis.

## Overview

The COVID-19 Detection System is a full-stack diagnostic platform that analyzes chest X-ray images using deep learning and generates both machine-readable and human-friendly diagnostic outputs. It combines advanced neural networks, model serving infrastructure, explainability techniques, and LLM-based reporting into a single production-ready workflow.

The system uses a hybrid inference pipeline with:
- AttentionResNet50 model served through NVIDIA Triton Inference Server
- Locally loaded Keras model for enhanced reliability
- Grad-CAM heatmaps for visual explainability
- LLM-generated clinical explanations and summaries
- A React-based Vite frontend for user interaction and report generation

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
COVID-19-Detection-System/
├── backend/
│   ├── app.py                          # FastAPI application
│   ├── triton_client_api.py            # Triton client wrapper
│   ├── Dockerfile.client               # Backend Docker image
│   ├── requirements.txt                # Original FastAPI dependencies
│   ├── requirements_triton.txt         # Triton client dependencies
│   ├── convert_model_to_savedmodel.py  # Model conversion utility
│   └── .env                            # Environment configuration
├── covid-frontend/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── services/api.js
│   ├── Dockerfile
│   ├── package.json
│   └── public/
├── model_repository/
│   └── covid_classifier/
│       ├── 1/
│       │   ├── attention_resnet_covid_classifier.keras
│       │   ├── model.py                # Triton Python backend
│       │   └── saved_model/
│       └── config.pbtxt                # Triton model configuration
├── docker-compose.yml
├── start-triton-server.bat             # Start Triton (Windows)
├── start-triton-api.bat                # Start API (Windows)
├── start-frontend.bat                  # Start Frontend (Windows)
├── quickstart.bat                      # One-command startup (Windows)
├── quickstart.sh                       # One-command startup (Linux/macOS)
├── setup_triton_models.py              # Model repository setup
├── test_setup.py                       # Setup verification
├── TRITON_MIGRATION_GUIDE.md           # Triton deployment guide
├── TRITON_MIGRATION_SUMMARY.md         # Migration summary
├── README.md                           # This file
├── README-STARTUP.md                   # Startup instructions
├── finalupdated (1).ipynb              # Experiment notebook
└── .gitignore
```

## Dataset

The COVID-19 Detection System uses the **COVID-19 Radiography Database** published by Tawsif ur Rahman on Kaggle.

**Dataset Information:**
- **Total Size:** ~780 MB
- **Total Samples:** 21,165 X-ray images
- **Classes:** COVID-19, Normal, Viral Pneumonia, Opacity
- **Distribution:**
  - COVID-19: 3,616 images (~17%)
  - Normal: 10,192 images (~48%)
  - Viral Pneumonia: 1,345 images (~6%)
  - Opacity: 6,012 images (~29%)

**Download:**
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

**Expected directory layout:**

```text
chest_xray/
├── COVID/
│   └── *.png
├── NORMAL/
│   └── *.png
├── Viral Pneumonia/
│   └── *.png
└── Opacity/
    └── *.png
```

**Setup Instructions:**
1. Download the dataset from Kaggle (requires Kaggle account)
2. Extract to `chest_xray/` at the project root
3. The system will automatically balance classes during training

## Models and Inference Pipeline

### Trained Models
- **AttentionResNet50** (`attention_resnet_covid_classifier.keras`): Primary Keras model for COVID-19 classification
  - Architecture: ResNet50 + Attention Mechanism
  - Input: (1, 224, 224, 3) normalized RGB chest X-ray
  - Output: 4-class classification (COVID-19, Normal, Viral Pneumonia, Opacity)
  - Trained on COVID-19 Radiography Database

### Model Performance
- Accuracy: ~94% on test set
- Sensitivity/Recall: ~92%
- Specificity: ~96%
- AUC-ROC: ~0.97

### Inference Pipeline
The system performs:
1. Image preprocessing (normalization, resizing to 224×224)
2. Model inference via Triton Inference Server
3. Grad-CAM heatmap generation for explainability
4. Confidence scoring and class probability distribution
5. LLM-based clinical explanation generation

## NVIDIA Triton Inference Server

The system uses Triton to serve inference models in a scalable, production-ready manner.

### Architecture Benefits
- Enterprise-grade model serving
- HTTP (port 8000) and gRPC (port 8001) APIs
- Metrics endpoint (port 8002) for monitoring
- Dynamic batching support
- Model versioning and hot-reload capabilities
- CPU-optimized for cost-effective inference

### Model Repository Structure

```text
model_repository/
└── covid_classifier/
    ├── 1/
    │   ├── attention_resnet_covid_classifier.keras
    │   ├── model.py                    # Python backend for inference
    │   └── saved_model/                # TensorFlow SavedModel format
    └── config.pbtxt                    # Triton configuration
```

### Health Check

```bash
curl http://localhost:8000/v2/models
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

### Backend Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1
# Or (Linux/macOS)
# source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# For Triton support
pip install -r backend/requirements_triton.txt

# Run FastAPI server
cd backend
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Frontend Setup
```bash
cd covid-frontend
npm install
npm run dev
```

The frontend will run on http://localhost:5173 and calls the backend at http://localhost:8080.

### Environment Variables

Create `backend/.env`:

```bash
# Triton Configuration
TRITON_SERVER_URL=localhost:8000
TRITON_MODEL_NAME=covid_classifier
API_PORT=8080

# LLM Configuration (optional)
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-flash-latest
GEMINI_TIMEOUT=25
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend exits unexpectedly | Check model paths, verify `attention_resnet_covid_classifier.keras` exists in `model_repository/covid_classifier/1/` |
| Triton models unavailable | Run `python setup_triton_models.py` and restart Triton |
| CORS/connection errors | Verify frontend calls backend on correct port (8080 for Docker, 8080 for local) |
| Docker image pull fails | Ensure internet connection; Triton image is ~7.8 GB |
| LLM chat not working | Set `GEMINI_API_KEY` in `backend/.env` |
| Memory issues | Reduce batch size or allocate more RAM to Docker (recommend 12GB+) |
| Model loading fails | Verify TensorFlow 2.20.0+ installed; check Python version is 3.11+ |

## Performance

- **Inference Latency:** ~500-800ms per image (CPU)
- **Memory Usage:** ~2.5 GB (Triton + API)
- **Throughput:** ~3-5 images/second (single instance)

## Deployment

For production deployment:

1. **DockerHub:** Push to registry for easy pulls
   ```bash
   docker tag covid-detection YOUR_USERNAME/covid-detection:v1.0
   docker push YOUR_USERNAME/covid-detection:v1.0
   ```

2. **Kubernetes:** Use provided `docker-compose.yml` as reference

3. **Cloud:** Deploy to AWS ECR, Azure Container Registry, or Google Container Registry

## License

Educational and research use only. See LICENSE file for details.

## Citation

If you use this system in research, please cite:
- Dataset: Tawsif ur Rahman, COVID-19 Radiography Database, Kaggle
- Methodology: [Your Paper/Reference]

## Author & Support

Developed as part of AI-driven medical imaging research. 
For issues, questions, or contributions, please open a GitHub issue.

Made with care for medical AI research and explainable inference.
dat
