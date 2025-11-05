from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import datetime
import logging
import io
import os
import imghdr
from typing import List, Optional

# ======================================================
# 🧠 FastAPI + PyTorch Image Classification Inference API
# ======================================================

# ======================================================
# Security Configuration
# ======================================================
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_IMAGE_TYPES = {"jpeg", "png", "gif", "bmp", "webp"}  # Valid image formats
MAX_IMAGE_PIXELS = 178956970  # PIL default decompression bomb limit

# ======================================================
# Rate Limiter Setup
# ======================================================
limiter = Limiter(key_func=get_remote_address)

# ======================================================
# Pydantic Models for Request/Response Validation
# ======================================================
class PredictionResult(BaseModel):
    label: str = Field(..., description="The predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

class PredictionResponse(BaseModel):
    top1_prediction: PredictionResult
    top3_predictions: List[PredictionResult]

class LogsResponse(BaseModel):
    recent_predictions: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    message: str

class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    num_classes: int
    total_parameters: int
    trainable_parameters: int
    model_file: str
    model_size_mb: float
    last_modified: str
    device: str
    status: str

# ======================================================
# FastAPI App Initialization
# ======================================================
app = FastAPI(
    title="🖼️ Image Classification API",
    description="Serve a trained ResNet18 model for CIFAR-10 image classification.",
    version="1.0.0"
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ======================================================
# CORS Configuration
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# Logging Setup (add near the top of api/app.py)
# ======================================================

LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "predictions.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

# ======================================================
# 1️⃣ Model Setup — Load pretrained fine-tuned model
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 class labels
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/best_model.pt")

def load_model():
    """Load the fine-tuned ResNet18 model."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()
print("✅ Model loaded successfully and ready for inference.")

# ======================================================
# 2️⃣ Image Preprocessing Pipeline (must match training)
# ======================================================
preprocess = transforms.Compose([
    transforms.Resize((32, 32)), #CIFAR-10 size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

# ======================================================
# 2.5️⃣ File Validation Utilities
# ======================================================
async def validate_image_file(file: UploadFile) -> bytes:
    """
    Validate uploaded image file for security:
    - Check file size
    - Validate image format using imghdr
    - Prevent decompression bombs

    Returns file contents as bytes if valid, raises HTTPException otherwise.
    """
    # Check file size
    contents = await file.read()
    file_size = len(contents)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f} MB"
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Validate image format using imghdr (checks actual file content, not just extension)
    image_type = imghdr.what(None, h=contents)
    if image_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{image_type}'. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )

    # Check for decompression bombs
    try:
        img = Image.open(io.BytesIO(contents))
        img.load()  # Force load to check for decompression bombs

        # PIL will raise DecompressionBombError if image is too large
        if img.size[0] * img.size[1] > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=400,
                detail="Image resolution too large (possible decompression bomb)"
            )
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=400,
            detail="Image resolution too large (decompression bomb detected)"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {str(e)}")

    return contents

# ======================================================
# 3️⃣ Root Endpoint
# ======================================================
@app.get("/")
async def root():
    return {"message": "Welcome to the Image Classification API.  Use POST /predict to classify an image."}

# ======================================================
# 4️⃣ Predict Endpoint
# ======================================================
@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")  # Rate limit: 30 requests per minute per IP
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Classify an uploaded image using the trained ResNet18 model.

    Security features:
    - File size validation (max 10 MB)
    - MIME type validation (only image types)
    - Decompression bomb prevention
    - Rate limiting (30 requests/minute per IP)
    """
    # Validate the uploaded file
    contents = await validate_image_file(file)

    try:
        # Open and convert image
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess the image
        tensor = preprocess(img).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

            # Top-3 predictions
            top_probs, top_idxs = torch.topk(probs, k=3)
            top_probs = top_probs.cpu().numpy()
            top_idxs = top_idxs.cpu().numpy()

            top3 = []
            for i in range(3):
                top3.append({
                    "label": CLASSES[top_idxs[i]],
                    "confidence": round(float(top_probs[i]), 4)
                })

            # Top-1 prediction
            top1 = top3[0]

        # Log the prediction
        logging.info(
            f"File: {file.filename} | Top1: {top1['label']} "
            f"({top1['confidence']}) | Top3: {[(p['label'], p['confidence']) for p in top3]}"
        )

        # Return prediction with Pydantic validation
        return PredictionResponse(
            top1_prediction=PredictionResult(**top1),
            top3_predictions=[PredictionResult(**p) for p in top3]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    
# ======================================================
# 5️⃣ Logs Endpoint — View Recent Predictions
# ======================================================
@app.get("/logs", response_model=LogsResponse)
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def get_logs(
    request: Request,
    limit: int = Query(default=10, ge=1, le=100, description="Number of recent logs to retrieve (1-100)")
):
    """
    Return the last 'limit' lines from the prediction log file.
    Example: GET /logs?limit=5

    Query parameters are validated: limit must be between 1 and 100.
    """
    try:
        if not os.path.exists(LOG_FILE):
            return LogsResponse(recent_predictions=[])

        with open(LOG_FILE, "r") as f:
            lines = f.readlines()

        # Return the last N lines (most recent predictions)
        recent_logs = [line.strip() for line in lines[-limit:]]
        return LogsResponse(recent_predictions=recent_logs)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {e}")
    
# ======================================================
# 6️⃣ Health Check Endpoint — for ECS / Monitoring
# ======================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API and model readiness.
    Returns a simple JSON confirming service health.
    """
    try:
        # Minimal internal check — ensures model is in memory
        test_tensor = torch.zeros((1, 3, 32, 32)).to(device)
        with torch.no_grad():
            _ = model(test_tensor)

        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=str(device),
            message="API and model are ready for inference."
        )

    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device=str(device),
            message=f"Health check failed: {str(e)}"
        )
    
# ======================================================
# 7️⃣ Model Info Endpoint — Metadata for Debugging & Versioning
# ======================================================
@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Returns key metadata about the currently loaded model.
    Useful for debugging, version tracking, and documentation.
    """
    try:
        # Get model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Get file stats
        if os.path.exists(MODEL_PATH):
            modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
            file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        else:
            modified_time = "Unknown"
            file_size_mb = 0

        return ModelInfoResponse(
            model_name="ResNet18 (Fine-Tuned on CIFAR-10)",
            architecture="ResNet18",
            num_classes=len(CLASSES),
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_file=os.path.basename(MODEL_PATH),
            model_size_mb=round(file_size_mb, 2),
            last_modified=str(modified_time),
            device=str(device),
            status="ready"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {e}")