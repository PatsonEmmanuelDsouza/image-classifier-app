"""
This script starts a celery worker, it allows to run tasks asynchronously and allows to free up the main thred.
The program uses redis to implement a job queue along with storing the necessary results
"""

from celery import Celery
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import requests
import concurrent.futures
import os
import time

# ----- Constants -----
IMG_WIDTH = 224
IMG_HEIGHT = 224
CLASS_NAMES = ['environment','studio']
MODEL_PATH = "mobilenet_v3_small_model"
MAX_WORKERS = 5

REDIS_BROKER = "redis://redis:6379/0"
REDIS_BACKEND = "redis://redis:6379/1"


# ----- starting Celery task queue -----
celery_app = Celery(
    "image_classifier_worker",
    broker=REDIS_BROKER,
    backend=REDIS_BACKEND
)


# ----- Helper functions -----
def disable_GPU():
    '''
    Method to disable GPU and allow the model to make inferences only using the CPU
    '''
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("TensorFlow: GPUs/MPS devices have been disabled. The model will now run on the CPU.")
    except Exception as e:
        print(f"TensorFlow: Could not disable GPUs/MPS. It might still use them if available. Error: {e}")
        
def process_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Reads image bytes, resizes, and prepares it for the model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image)
    return np.expand_dims(image_array, axis=0)

def download_and_classify_url(url: str) -> dict:
    """
    This is the core function that will be run by each thread.
    It handles the entire process for a single URL.
    """
    current_model = get_model()
    
    if not current_model:
        # This check is important in case the model failed to load
        return {"url": url, "status": "error", "detail": "Model is not available."}

    try:
        # Download the image from the URL in memory
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        # Process the image bytes for the model
        image_batch = process_image_bytes(response.content)

        # Get the model's prediction
        # Note: model.predict is thread-safe in TensorFlow
        prediction = current_model.predict(image_batch)
        
        # Process the prediction to get class and confidence
        score = float(prediction[0][0])
        
        # since it is a binary model, we have two classes, where sigmoid function will return a value [0,1]
        if score >= 0.5:
            class_name = CLASS_NAMES[1]
            confidence = score
        else:
            class_name = CLASS_NAMES[0]
            confidence = 1 - score
        
        # Return a successful result dictionary
        return {
            "url": url,
            "status": "success",
            "predicted_class": class_name,
            "confidence_level": f"{confidence:.2%}"
        }
    except Exception as e:
        # If anything goes wrong, return an error dictionary
        return {"url": url, "status": "error", "detail": str(e)}


# ----- loading model -----
disable_GPU()
model = None


def get_model():
    """
    Loads the Keras model into the global 'model' variable if it hasn't been loaded yet.
    This is called "lazy loading".
    """
    global model
    if model is None:
        print("Model not loaded yet. Loading now...")
        if not os.path.exists(MODEL_PATH):
            print(f"FATAL: Model folder not found at {MODEL_PATH}")
            return None
        try:    
            model = keras.models.load_model(MODEL_PATH)
            # Perform a dummy prediction to fully initialize the model
            model.predict(np.zeros((1, IMG_WIDTH, IMG_HEIGHT, 3)))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"FATAL: Could not load model. Error: {e}")
            # The model will remain None, and tasks will fail gracefully.
    return model
    
    
    
# ----- Celery Tasks -----
@celery_app.task(name="classify_images_from_urls")
def classify_images_from_urls_task(urls: list[str]):
    """
    The main background task. It uses a thread pool to process a list of URLs concurrently.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(download_and_classify_url, urls))
    return results
    