from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import io
from PIL import Image
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("F:\\final_project\\tumor-detection\\runs\\detect\\train\\weights\\best.pt")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*'])

# Prediction endpoint
@app.post('/predictions')
async def pred(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Load image and preprocess
    image = Image.open(io.BytesIO(contents)).convert("RGB")  # Convert to RGB if grayscale
    
    # Run the model on the image (inference mode)
    results = model(image)
    
    # Get the first result (assuming single image input)
    result = results[0]
    
    # Get the detected image with bounding boxes
    detected_image = result.plot(labels=False)  # This returns a numpy array
    
    # Convert the detected image to bytes
    _, encoded_image = cv2.imencode('.png', detected_image)
    detected_image_bytes = encoded_image.tobytes()
    
    # Get the labels and confidence scores
    labels = []
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls)
        label = model.names[class_id]
        confidence = float(box.conf)
        labels.append({"index": i, "label": label, "confidence": confidence})
    
    # Example: Reorder or filter the labels based on some criteria
    # Here, we sort the labels by confidence score in descending order
    labels_sorted = sorted(labels, key=lambda x: x['confidence'], reverse=True)
    
    # Example: Filter labels to only include those with confidence > 0.5
    labels_filtered = [label for label in labels if label['confidence'] > 0.5]
    
    # Return the detected image as a binary response
    return StreamingResponse(io.BytesIO(detected_image_bytes), media_type="image/png")

# Endpoint to return labels
@app.post('/labels')
async def get_labels(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Load image and preprocess
    image = Image.open(io.BytesIO(contents)).convert("RGB")  # Convert to RGB if grayscale
    
    # Run the model on the image (inference mode)
    results = model(image)
    
    # Get the first result (assuming single image input)
    result = results[0]
    
    # Get the labels and confidence scores
    labels = []
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls)
        label = model.names[class_id]
        confidence = float(box.conf)
        labels.append({"index": i, "label": label, "confidence": confidence})
    
    # Example: Reorder or filter the labels based on some criteria
    # Here, we sort the labels by confidence score in descending order
    labels_sorted = sorted(labels, key=lambda x: x['confidence'], reverse=True)
    
    # Example: Filter labels to only include those with confidence > 0.5
    labels_filtered = [label for label in labels if label['confidence'] > 0.5]
    
    # Return the labels as a JSON response
    return JSONResponse(content={"labels": labels_filtered})

# To run the FastAPI app, use the command: uvicorn main:app --reload