from fastapi import FastAPI, File, UploadFile
from PIL import Image
import onnxruntime as ort
import numpy as np
import io
import math


app = FastAPI()

ort_session = ort.InferenceSession('/content/drive/MyDrive/portafolio/dlops/models/binary_classifier_3.onnx')
THRESHOLD = 0.5

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@app.get("/")
async def root():
    return {"message": "Hello World"}
    

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    input = np.expand_dims(np.array(img, dtype=np.unit8), axis=0)
    input = np.expand_dims(np.array(input, dtype=np.unit8), axis=0)
    ort_inputs = {
        "input": input
    }
    ort_output = ort_session.run(['output'], ort_inputs)[0]
    output = sigmoid(ort_output)
    response = {
        'proba': output,
        'labels': '3' if output > THRESHOLD else 'Not 3'
    }
    return response
