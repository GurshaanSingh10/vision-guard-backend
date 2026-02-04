from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from skimage.measure import shannon_entropy

app = FastAPI(title="VisionGuard AI API")

def extract_features(img):
    lap = cv2.Laplacian(img, cv2.CV_64F).var()
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    tenengrad = np.mean(gx**2 + gy**2)
    entropy = shannon_entropy(img)
    return lap, tenengrad, entropy

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        data = await file.read()
        img_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))

        lap, ten, ent = extract_features(img)

        # simple, explainable decision
        is_blur = lap < 100 and ten < 300

        results.append({
            "filename": file.filename,
            "prediction": "Blur" if is_blur else "Not Blur",
            "metrics": {
                "laplacian": round(lap, 2),
                "tenengrad": round(ten, 2),
                "entropy": round(ent, 2)
            }
        })

    return {"results": results}
