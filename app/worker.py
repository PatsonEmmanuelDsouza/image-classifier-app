"""
This script starts a celery worker, it allows to run tasks asynchronously and allows to free up the main thred.
The program uses redis to implement a job queue along with storing the necessary results
"""
from fastapi import HTTPException, UploadFile
from celery import Celery

import tensorflow as tf
from tensorflow import keras
import numpy as np

import threading
import concurrent.futures

import io
import os
from datetime import datetime
from PIL import Image

import requests

import uuid

from .database import SessionLocal, ImageRecord
from .baseModels import URLClassificationResult, FileDownloadPredictionResult


# ----- Load Environment Variables -----
from dotenv import load_dotenv
load_dotenv()

# ----- Constants ----
IMG_WIDTH = 224
IMG_HEIGHT = 224
CLASS_NAMES = ['environment','studio']

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
CELERY_BROKER_DB = os.getenv("CELERY_BROKER_DB")
CELERY_BACKEND_DB = os.getenv("CELERY_BACKEND_DB")
MODEL_PATH = os.getenv("MODEL_PATH")

MAX_WORKERS = int(os.getenv("MAX_WORKERS")) 

REDIS_BROKER = f"redis://{REDIS_HOST}:{REDIS_PORT}/{CELERY_BROKER_DB}"
REDIS_BACKEND = f"redis://{REDIS_HOST}:{REDIS_PORT}/{CELERY_BACKEND_DB}"

MODEL_VERSION = os.getenv("MODEL_VERSION")


# ----- starting Celery task queue -----
celery_app = Celery(
    "image_classifier_worker",
    broker=REDIS_BROKER,
    backend=REDIS_BACKEND,
    result_expires = 900
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
        
def process_image(image_bytes: bytes) -> np.ndarray:
    """
    Reads image bytes, resizes, and prepares it for the model.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB") 
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = np.array(image)
        image_batch = np.expand_dims(image_array, axis=0)
        return image_batch
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        
def process_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Reads image bytes, resizes, and prepares it for the model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image)
    return np.expand_dims(image_array, axis=0)    

def download_and_classify_url(url: str, save=True) -> URLClassificationResult:
    """
    This is the core function that will be run by each thread.
    It handles the entire process for a single URL.
    """
    # Create a database session that will be closed properly
    db = SessionLocal()
    try:
        # --- 1. DB CHECK ---
        print("DB CHECK")
        existing_record = db.query(ImageRecord).filter(ImageRecord.url == url).first()
        print("DB CHECK done")

        if existing_record and existing_record.status == "success":
            print(f"CACHE HIT: Found existing successful record for {url}")
            return URLClassificationResult(
                url=existing_record.url,
                status=existing_record.status,
                predicted_class=existing_record.predicted_class,
                confidence_level=str(existing_record.confidence_level)
            )
        
        if not existing_record:
            print(f"WARNING: No pre-existing record found for {url}. Creating one now.")
            existing_record = ImageRecord(url=url, status="processing")
            db.add(existing_record)
            db.commit()
            db.refresh(existing_record)

        # --- 2. DOWNLOAD & VALIDATE ---
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            error_status = "error - url is not an image"
            existing_record.status = error_status
            existing_record.predicted_class="unknown"
            db.commit()
            return URLClassificationResult(url=url, status=error_status, predicted_class="unknown", confidence_level="0")
                        
        # --- 3. PREDICT ---
        current_model = get_model()
        
        # model is not available
        if not current_model:
            error_status = "error - model not available"
            existing_record.status = error_status
            db.commit()
            return URLClassificationResult(url=url, status=error_status, predicted_class="unknown", confidence_level="0")
         
        image_batch = process_image_bytes(response.content)
        prediction = current_model.predict(image_batch)
        score = float(prediction[0][0])
        
        if score >= 0.5:
            # studio
            class_name = CLASS_NAMES[1]
            confidence = score
        else:
            # environment
            class_name = CLASS_NAMES[0]
            confidence = 1 - score
        
        # when the save flag is true, we save the image locally
        if save:
            current_day_dir = datetime.now().strftime("%Y%m%d")

            main_save_dir = os.getenv("IMAGE_DIRECTORY")
            save_directory = f"{main_save_dir}/{current_day_dir}"
            
            os.makedirs(save_directory, exist_ok=True)

            # extension
            file_extension = content_type.split('/')[-1]
            
            unique_filename = f"{uuid.uuid4()}_{class_name}_{round(confidence*100)}.{file_extension}"
            
            save_path = os.path.join(save_directory, unique_filename)
                        
            # writing the file
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            existing_record.local_filename = unique_filename
            existing_record.folder_location = current_day_dir
            
            print(f"IMAGE LOOKUP: {url} saved to {save_path}")
        
        existing_record.status = "success"
        existing_record.predicted_class = class_name
        existing_record.confidence_level = round(confidence*100,2)
        existing_record.prediction_model_version = MODEL_VERSION
        existing_record.image_type = "url"
        
        db.commit()
        
        return URLClassificationResult(
            url=url,
            status="success",
            predicted_class=class_name,
            confidence_level=f"{(confidence*100):.2f}" # Use .2f for float formatting
        )
    
    except Exception as e:
        error_status = f"error - {e.__class__.__name__}"
        print(f"ERROR processing {url}: {e}")
        
        # Update the record in the DB with the specific error
        if 'existing_record' in locals() and existing_record:
            existing_record.status = error_status
            db.commit()
            
        return URLClassificationResult(url=url, status=error_status, predicted_class="unknown", confidence_level="0")

    finally:
        db.close()
    
def download_and_classify_image_file(image_bytes: bytes, originalFileName: str, save = True)-> FileDownloadPredictionResult:
    # image_bytes = await file.read()
    
    image_batch = process_image_bytes(image_bytes)
    
    model = get_model()
    
    prediction = model.predict(image_batch)
    
    score = float(prediction[0][0])
        
    if score >= 0.5:
        # studio
        class_name = CLASS_NAMES[1]
        confidence = score
    else:
        # environment
        class_name = CLASS_NAMES[0]
        confidence = 1 - score

    confidence=round(confidence*100,2)

    file_extension = os.path.splitext(originalFileName)[1]
    current_day_dir = datetime.now().strftime("%Y%m%d")

    main_save_dir = os.getenv("IMAGE_DIRECTORY")
    save_directory = f"{main_save_dir}/{current_day_dir}"
    
    os.makedirs(save_directory, exist_ok=True)

    unique_filename = f"{uuid.uuid4()}_{class_name}_{round(confidence*100)}{file_extension}"
    
    save_path = os.path.join(save_directory, unique_filename)
                    
    try:
        with open(save_path, 'wb') as f:
            f.write(image_bytes)
    except IOError as e:
        raise IOError(f"Failed to write file to {save_path}: {e}")
    
    return FileDownloadPredictionResult(
        local_file_name=unique_filename,
        confidence_level=str(confidence),
        predicted_class=class_name,
        current_day_dir=current_day_dir
    )
# ----- loading model -----
disable_GPU()
model = None
model_lock = threading.Lock()

def get_model():
    """
    Loads the Keras model into the global 'model' variable if it hasn't been loaded yet.
    This version is thread-safe using a lock.
    """
    global model
    # No need to acquire the lock just to check, this is a quick check.
    if model is None:
        # 2. Acquire the lock. Only one thread can pass this point at a time.
        #    Other threads will wait here until the lock is released.
        with model_lock:
            # 3. Double-check if the model is still None.
            #    This is crucial because another thread might have finished loading it
            #    while the current thread was waiting for the lock.
            if model is None:
                print("MODEL INFO:  Model not loaded yet. Loading now...")
                # ... (your existing model loading logic) ...
                try:    
                    model = keras.models.load_model(MODEL_PATH)
                    # Perform a dummy prediction to fully initialize the model
                    # This also helps with the TensorFlow retracing warning.
                    model.predict(np.zeros((1, IMG_WIDTH, IMG_HEIGHT, 3)))
                    print("MODEL INFO:  Model loaded successfully.")
                except Exception as e:
                    print(f"FATAL: Could not load model. Error: {e}")
            else:
                print("MODEL INFO:  Reusing model loaded by another thread.")
    else:
        print("MODEL INFO:  Reusing already loaded model.")
    return model

    
    
    
# ----- Celery Tasks -----
@celery_app.task(name="classify_images_from_urls")
def classify_images_from_urls_task(urls: list[str]):
    """
    The main background task. It uses a thread pool to process a list of URLs concurrently.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(download_and_classify_url, urls))
        
    serializable_results = [result.model_dump() for result in results]
    
    return serializable_results
    