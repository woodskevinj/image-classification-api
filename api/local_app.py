from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import torch
from torchvision import transforms
import uvicorn

app = FastAPI(title="Image Classification API", version="0.1.0")

# ---- Runtime /model placeholders ----
# In a later step we will load a real pretrained model (e.g., resnet18) and class names.
# For now, we keep them as None to demonstrate the API contract.
MODEL: Optional[torch.nn.Module] = None
CLASS_NAMES: List[str] = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic transform placdeholder (adjust during training phase to match preprocessing)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # Note: Update normlaization with the exact means/stds used during training
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225),
])

class Prediction(BaseModel):
    label: str
    score: float

class PredictResponse(BaseModel):
    top1: Prediction
    # Optionally include top_k later: List[Prediction]

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": MODEL is not None}

def _validate_image(content_type: str):
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {content_type} (use JPEG/PNG/Webp)")
    
def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Could not decode.")
    
def _dummy_predict(tensor: torch.Tensor) -> PredictResponse:
    """
    Temporary stub logic that fakes a prediction so you can test the endpoint
    end-to-end. Replace later by real model inference.
    """
    # Fake class list if empty (for quick local testing)
    labels = CLASS_NAMES or ["class_a", "class_b", "class_c"]

    # Generate a trivial, deterministic index based on the sum of pixels
    idx = int(tensor.sum().item()) % len(labels)
    return PredictResponse(top1=Prediction(label=labels[idx], score=0.33))

def _model_predict(tensor: torch.Tensor) -> PredictResponse:
    """
    Real model inference path (wired in once MODEL is loaded).
    """
    if MODEL is None:
        # Until training is done, fall back to the stub
        return _dummy_predict(tensor)
    
    MODEL.eval()
    with torch.no_grad():
        logits = Model(tensor.to(DEVICE)) # shape: [1, num_classes]
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
        top1_idx = int(torch.argmax(probs).item())
        top1_score = float(probs[top1_idx].item())

    label = CLASS_NAMES[top1_idx] if CLASS_NAMES else str(top1_idx)
    return PredictResponse(top1=Prediction(label=label, score=top1_score))

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image upload and returns a (stub) prediction.
    Replace stubbed logic with real model inference once training is complete.
    
    Test locally:
        curl -X POST -F "file=@test.jpg" http://127.0.0.1:8000/predict
    """
    _validate_image(file.content_type)
    file_bytes = await file.read()
    img = _read_image(file_bytes)

    # Preprocess -> shape [1, C, H, W]
    tensor = preprocess(img).unsqueeze(0)

    # Return dummy for now; switches to real model once MODEL is set
    return JSONResponse(content=_model_predict(tensor).model_dump())

# Optional local dev runner: `python api/app.py`
if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
