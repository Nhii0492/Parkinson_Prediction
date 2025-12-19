import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import numpy as np
from torchvision import transforms
from model import load_model
import uvicorn

MODEL_PATH = "parkinson_cbam7x7_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Healthy", "Parkinson"]

app = FastAPI(
    title="Parkinson's Disease Detection API",
    description="API to predict Parkinson's Disease from spiral drawing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

try:
    model = load_model(MODEL_PATH, DEVICE)
except Exception as e:
    model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


def predict_image(tensor: torch.Tensor) -> dict:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    tensor = tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        label = CLASS_NAMES[pred_idx]
    
    return {
        "label": label,
        "confidence": round(confidence, 4)
    }


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    import os
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Parkinson's Disease Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "frontend": "Visit /static/index.html"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (image/*)"
        )
    
    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
        result = predict_image(tensor)
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
