from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import datetime
import logging
import io
import os

# ======================================================
# 🧠 FastAPI + PyTorch Image Classification Inference API
# ======================================================

app = FastAPI(
    title="🖼️ Image Classification API",
    description="Serve a trained ResNet18 model for CIFAR-10 image classification.",
    version="1.0.0"
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
# 3️⃣ Root Endpoint
# ======================================================
@app.get("/")
async def root():
    return {"message": "Welcome to the Image Classification API.  Use POST /predict to classify an image."}

# ======================================================
# 4️⃣ Predict Endpoint
# ======================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {e}")
    
    try:
        # Preprocess the image
        tensor = preprocess(img).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

            # ✅ Top-3 predictions
            top_probs, top_idxs = torch.topk(probs, k=3)
            top_probs = top_probs.cpu().numpy()
            top_idxs = top_idxs.cpu().numpy()

            top3 = []
            for i in range(3):
                top3.append({
                    "label": CLASSES[top_idxs[i]],
                    "confidence": round(float(top_probs[i]), 4)
                })
            #predicted_label = CLASSES[top_idx.item()]
            #confidence = round(top_prob.item(), 4)

            # ✅ Top-1 prediction
            top1 = top3[0]

        # ✅ Log the prediction
        logging.info(
            f"File: {file.filename} | Top1: {top1['label']} "
            f"({top1['confidence']}) | Top3: {[(p['label'], p['confidence']) for p in top3]}"
        )

        # Return prediction as JSON
        return JSONResponse(content={
            "top1_prediction": top1,
            "top3_predictions": top3
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    
# ======================================================
# 5️⃣ Logs Endpoint — View Recent Predictions
# ======================================================
@app.get("/logs")
async def get_logs(limit: int = 10):
    """
    Return the last 'limit' lines from the prediction log file.
    Example: GET /logs?limit=5
    """
    try:
        if not os.path.exists(LOG_FILE):
            return {"message": "No logs found yet."}
        
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()

        # Return the last N lines (most recent predictions)
        recent_logs = [line.strip() for line in lines[-limit:]]
        return {"recent_predictions": recent_logs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {e}")
    
# ======================================================
# 6️⃣ Health Check Endpoint — for ECS / Monitoring
# ======================================================
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API and model readiness.
    Returns a simple JSON confirming service health.
    """
    try:
        # Minimal internal check — ensures model is in memory
        test_tensor = torch.zeros((1, 3, 32, 32)).to(device)
        with torch.no_grad():
            _=model(test_tensor)

        return {
            "status": "healthy",
            "model_loaded": True,
            "device": str(device),
            "message": "API and model are ready for inference."
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }
    
# ======================================================
# 7️⃣ Model Info Endpoint — Metadata for Debugging & Versioning
# ======================================================
@app.get("/info")
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

        return {
            "model_name": "ResNet18 (Fine-Tuned on CIFAR-10)",
            "architecture": "ResNet18",
            "num_classes": len(CLASSES),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_file": os.path.basename(MODEL_PATH),
            "model_size_mb": round(file_size_mb, 2),
            "last_modified": str(modified_time),
            "device": str(device),
            "status": "ready"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {e}")