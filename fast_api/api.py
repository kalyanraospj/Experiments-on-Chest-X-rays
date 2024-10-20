from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import base64
import torch
import torchvision.transforms as T
from infrence import run_inference, load_model


# Define input and output formats
class ImageInput(BaseModel):
    image_data: str  # Base64-encoded image data


class DetectionResponse(BaseModel):
    boxes: List[List[float]]
    labels: List[int]
    scores: List[float]


# Initialize FastAPI app
app = FastAPI()


# Function to decode base64 image
def decode_base64_image(image_data: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


# Function to transform image and make prediction
def predict(image: Image.Image):
    model = load_model()
    predictions = run_inference(image=image, model=model)
    return predictions


# Route for object detection
@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(image_input: ImageInput):
    # Decode the image
    image = decode_base64_image(image_input.image_data)

    # Make predictions
    boxes, labels, scores = predict(image)

    return DetectionResponse(boxes=boxes, labels=labels, scores=scores)


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
