from fastapi import FastAPI, File, UploadFile,HTTPException
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pathlib import Path
from keras.layers import TFSMLayer  # Keras 3 specific
from keras.models import Model
import matplotlib
import uvicorn

matplotlib.use('Agg')  # Set non-interactive backend

app = FastAPI()


# Model configuration
CLASS_NAMES = ["bad_yam", "good_yam", "not_yam"]
MODEL_PATH = Path("../models/1")  # Path to your saved model


# Load model as layer
try:
    # Create inference-only model
    model_layer = TFSMLayer(str(MODEL_PATH), call_endpoint='serving_default')
    input_layer = tf.keras.Input(shape=(256, 256, 3))  # Adjust shape as needed
    output = model_layer(input_layer)
    MODEL = Model(inputs=input_layer, outputs=output)
    
    print("Model loaded successfully via TFSMLayer")
    
except Exception as e:
    raise ValueError(f"Model loading failed: {str(e)}")

def preprocess_image(image_data):
    
    try:
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        #Resize to model"s expected input shape
        image = image.resize((256, 256))

        image_array = np.array(image)

        
        return image_array
    
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")


    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()  # Sync read instead of await file.read()

        image_array = preprocess_image(image_data)
        image_batch = np.expand_dims(image_array, 0)
        print(f"\nBatch input shape: {image_batch.shape}")  # Should be (1, 256, 256, 3)


        predictions = MODEL.predict(image_batch)
        print(f"Raw predictions shape: {predictions.shape if isinstance(predictions, np.ndarray) else 'dict'}")

        # Handle output
        if isinstance(predictions, dict):
            output = next(iter(predictions.values()))
            output = output.numpy() if hasattr(output, 'numpy') else output
            output = output.squeeze()
        else:
            output = predictions.squeeze()

        predicted_class = CLASS_NAMES[np.argmax(output)]
        confidence = float(np.max(output))

        return {
            'class': predicted_class,
            'confidence': confidence,
            'filename': file.filename,
            'raw_output': output.tolist()  # optional for debug
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8001)