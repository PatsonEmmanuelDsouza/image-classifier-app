from fastapi import FastAPI, HTTPException, Body, status, Request, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from celery.result import AsyncResult


from typing import List
from sqlalchemy.orm import Session
from .baseModels import URLPayload, JobResponse, JobResultResponse, ImageUrlPayload, URLClassificationResult
from .worker import classify_images_from_urls_task, download_and_classify_url, get_model
from .database import init_db, get_db, ImageRecord

from datetime import datetime

import os
import shutil

import logging

# ------ loading env var ------
from dotenv import load_dotenv
load_dotenv()

# ----- App setup -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("--- loading model and DB on startup ---")
    init_db()
    get_model()
    # The server is now ready to accept requests
    yield
    # Code to run on shutdown (optional)
    print("--- Server shutting down ---")


app = FastAPI(
    title="Binary Image Classification API",
    description="An API that allows you to make requests and post images, for an image posted to the the API response would be the predicted class of the image ",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000"],  # The origin of your Flask frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allows PATCH, POST, GET, etc.
    allow_headers=["*"],
)


# ----- env items ------
IMAGE_DIRECTORY = os.getenv("IMAGE_DIRECTORY")
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)

RETRAIN_STUDIO_DIR = os.getenv("RETRAIN_STUDIO_DIR")
os.makedirs(RETRAIN_STUDIO_DIR, exist_ok=True)

ERROR_IMAGE_DIR = os.getenv("ERROR_IMAGE_DIR")
os.makedirs(ERROR_IMAGE_DIR, exist_ok=True)


RETRAIN_ENVIRONMENT_DIR = os.getenv("RETRAIN_ENVIRONMENT_DIR")
os.makedirs(RETRAIN_ENVIRONMENT_DIR, exist_ok=True)


LOG_DIR = os.getenv("LOG_DIR")
LOG_LEVEL = os.getenv("LOG_LEVEL")


# --- Set up Logging ---
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "app.log")

# Configure the root logger
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path), # Log to a file
        logging.StreamHandler()            # Log to the console
    ]
)

# ------- helper functions --------
def copy_image_for_retraining(file_path: str, destination_dir: str, file_name: str):
    """
    Copies an image to the appropriate retraining folder.
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"Source file not found: '{file_path}'")
            return

        # Create the date-based sub-directory in the destination
        current_day_dir = datetime.now().strftime("%Y%m%d")
        final_destination_dir = os.path.join(destination_dir, current_day_dir)
        os.makedirs(final_destination_dir, exist_ok=True)
        
        destination_path = os.path.join(final_destination_dir, file_name)
        
        shutil.copy(file_path, destination_path)
        logging.info(f"Successfully copied '{file_name}' for retraining to '{destination_path}'")
    except Exception as e:
        logging.error(f"Failed to copy file '{file_name}': {e}")

# --- API Endpoints ---
@app.get('/', tags=["Default Endpoint"])
def home():
    # you can use this to check if the backend is up or not
    return {'message': "welcome to this tool!"}


# classify a list of images, using their urls
@app.post("/classify-image-urls/",
          summary="Start a new image classification job", 
          response_model=JobResponse, 
          status_code=status.HTTP_202_ACCEPTED, 
          tags=["Classification"])
async def classify_image_urls(payload: URLPayload, db: Session = Depends(get_db)):
    """
        Starts a background job to classify images from a list of URLs.

        This endpoint is asynchronous. It accepts a list of public image URLs,
        initiates a background classification task, and responds immediately
        with a unique ID for the created job. The job status and results can be
        retrieved later using this ID.

        Args:
            payload (URLPayload): A JSON object containing a list of image URLs.

        Returns:
            JobResponse: A JSON object containing the unique ID of the background job.

        Raises:
            HTTPException: A 400 Bad Request error if the `urls` list is empty.
    """
    if not payload.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")
        
    # Convert each HttpUrl object in the list to a plain string
    urls_as_strings = [str(url) for url in payload.urls]
    # Now, pass the list of strings to the Celery task
    
    records_to_update = []

    try:
        for url_str in urls_as_strings:
            existing_record = db.query(ImageRecord).filter(ImageRecord.url == url_str).first()
            
            if existing_record:
                continue
            else:
                # If it's a new URL, create a placeholder record
                new_record = ImageRecord(
                    url=url_str,
                )
                db.add(new_record)
                records_to_update.append(new_record)
        db.commit()
    except Exception as e:
        db.rollback()
        logging.error(f"DB Error during Phase 1 (record creation): {e}")
        raise HTTPException(status_code=500, detail="Failed to prepare database records for the job.")

    task = classify_images_from_urls_task.delay(urls_as_strings)
    
    logging.info(f"post/classify-image-urls/ was accessed! {len(urls_as_strings)} task(s) with job_id: {task.id}")
    
    # --- Database Population Logic ---
    try:
        for record in records_to_update:
            # Refresh the record in case the session state is stale
            db.refresh(record)
            record.job_id = task.id
        
        # Commit the final update with the job ID
        db.commit()
        logging.info(f"Successfully updated {len(records_to_update)} records with job_id: {task.id}")

    except Exception as e:
        db.rollback()
        # This is also a critical failure, as the job is running but the DB doesn't reflect it.
        logging.error(f"DB Error during Phase 2 (job_id update): {e}. Job {task.id} is running without linked DB records.")
        # You might need a mechanism to reconcile this later.
        raise HTTPException(status_code=500, detail="Failed to associate job ID with database records.")
        
        
    return {"job_id": task.id}


# classifies a single image using the url provided in the body of the request
@app.post("/classify-image-url/",
          summary="Classify a single image url",
          status_code=status.HTTP_200_OK,
          response_model=URLClassificationResult, 
          tags=["Classification"],
          responses={
            200: {"description": "Image classified successfully."},
            400: {"description": "Bad Request: No image_url was provided in the body."},
            500: {"description": "Database exception: A database error took place."}
            
            }
        )
async def classify_image_url(
    payload: ImageUrlPayload, db: Session = Depends(get_db)
):
    """
        Endpoint that allows to immediately classify the image from an input URL.

        Args:
            image_url: A single url for an image.

        Returns:
            URLClassificationResult: A JSON object containing the url, status and class of an image.

        Raises:
            HTTPException: A 400 Bad Request error if the `image_url` is empty.
    """
    logging.info(f"post/classify-image-url/ was accessed to classify a single url.")
    
    # convert httpUrl to str
    url = str(payload.image_url)
    
    try:
        # checking if the URL already exists on the DB
        existing_record = db.query(ImageRecord).filter(ImageRecord.url == url).first()
        if not existing_record:        
            # Create a new record for the given URL
            logging.info(f"New image URL is being looked up, creating an entry in DB.")
            db_record = ImageRecord(
                url=url,
            )
            db.add(db_record)
            db.commit()
        else: 
            logging.info(f"The URL for image being looked up is already in database, did not create new record in DB")
    
    except Exception as e:
        logging.error(f"While checking if URL is already in DB, and error was faced: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="A database error occurred.")
        
    result = download_and_classify_url(url, save=True)
    logging.info(f"Image {result.url} has been classified with predicted class: {result.predicted_class}")

    return result



@app.get("/get_job/{job_id}", response_model=JobResultResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """
    Retrieves the status and result of a classification job by its ID.
    """
    logging.info(f"get/jobs/{job_id} was accessed! to look up the progress of the job: {job_id}")
    # Get the task result from the Celery backend (Redis)
    task_result = AsyncResult(id=job_id, app=classify_images_from_urls_task.app)

    # Prepare the response
    response = {
        "job_id": job_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None
    }
    
    # Handle the case where the job failed by providing the error information
    if task_result.status == 'FAILURE':
        response['result'] = str(task_result.info) # Get the exception info

    return response


@app.get("/image/{date_folder}/{filename}", 
         name = 'get_image', 
         tags=["Images Folder"],
         summary="Endpoint that allows you to get a particular image using the filename",
         status_code=status.HTTP_200_OK,)
async def get_image(filename: str, date_folder: str):
    """
    Custom get function to get the image stored on backend when you look up the file
    """
    logging.info(f"get/lookedup-images endpoint was accessed for looking up filename '{filename}'")
    file_path = os.path.join(IMAGE_DIRECTORY, date_folder, filename)
    print(f"file path: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"Unable to find filename:'{filename}'")
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(file_path)

@app.get("/review-lookedup-images/",
         summary="An endpoint that allows you to load images that need to be reviewed by an admin",
         status_code=status.HTTP_200_OK,
         response_model=List[URLClassificationResult],
         tags=["Review Images"])
async def review_lookedup_images(request: Request, db: Session = Depends(get_db)):
    """
    Retrieves all image records that have not yet been reviewed by an admin.

    If a record doesn't have a local_url, this endpoint generates one,
    saves it to the database, and returns the complete records.
    """
    images_to_review = db.query(ImageRecord).filter(
        ImageRecord.admin_reviewed == False,
        ImageRecord.status == 'success').all()
    
    # Flag to check if we need to commit any changes
    made_changes = False

    # return list
    images_to_be_reviewed = []

    for image in images_to_review:
        # defining result
        result_image = URLClassificationResult(
            url=image.url,
            status=image.status,
            predicted_class=image.predicted_class,
            confidence_level=f"{image.confidence_level:.2f}"
        )
        
        if image.local_url is None and image.folder_location and image.filename:
            # Generate the URL if local_url is missing
            try:
                image_local_url = request.url_for('get_image', date_folder=image.folder_location, filename=image.filename)
                image.local_url = str(image_local_url)
                made_changes = True

            except Exception as e:
                logging.error(f"Unable to locate the url for the image {image.folder_location}/{image.filename} exception: {e}")
                
        result_image.local_url = image.local_url if image.local_url else image.url
        images_to_be_reviewed.append(result_image)
        
    # If any URLs were generated, commit all changes to the database at once
    if made_changes:
        db.commit()
            
    return images_to_be_reviewed
    
    
@app.patch(
    "/set-image-as-reviewed/",
    summary="Mark an image record as reviewed",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Review Images"]
)
async def set_image_as_reviewed(
    payload: ImageUrlPayload = Body(None),
    db: Session = Depends(get_db)
):
    """
    Finds an image record by its URL and sets its 'admin_reviewed'
    status to True. Returns no content on success.
    """
    url = str(payload.image_url)
    
    # Find the record
    db_record = db.query(ImageRecord).filter(ImageRecord.url == url).first()

    # Handle case where the record doesn't exist
    if not db_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image record with URL '{url}' not found."
        )
    
    # Check if it's already reviewed to avoid unnecessary database writes
    if db_record.admin_reviewed:
        # Option 1: Raise a conflict error (cleanest)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Image record with URL '{url}' has already been reviewed."
        )

    # If not found or already reviewed, update the record
    db_record.admin_reviewed = True
    db.commit()
    
    return


@app.patch(
    "/relabel-image/",
    summary="Method that will relabel an image as the predicted label is not satisfactory",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Review Images"]
)
async def relabel_image(
    background_tasks: BackgroundTasks, # 1. Add BackgroundTasks dependency
    payload: ImageUrlPayload = Body(...),
    db: Session = Depends(get_db)
):
    """
    Finds an image, updates its label in the DB, and triggers a background
    task to copy the file for retraining.
    """
    url = str(payload.image_url)
    db_record = db.query(ImageRecord).filter(ImageRecord.url == url).first()

    if not db_record:
        raise HTTPException(status_code=404, detail=f"Image record with URL '{url}' not found.")
    
    if db_record.admin_reviewed:
        raise HTTPException(status_code=409, detail=f"Image record with URL '{url}' has already been reviewed.")
    
    # --- 1. Perform the quick database update ---
    db_record.re_label = True
    db_record.admin_reviewed = True
    db.commit()
    db.refresh(db_record) # Refresh to get the latest state

    # --- 2. Schedule the slow file I/O to run in the background ---
    file_path = os.path.join(IMAGE_DIRECTORY, db_record.folder_location, db_record.filename)
    
    # Determine the correct destination directory
    if db_record.predicted_class == "studio":
        destination_dir = RETRAIN_ENVIRONMENT_DIR
    elif db_record.predicted_class == "environment":
        destination_dir = RETRAIN_STUDIO_DIR
    else:
        destination_dir = ERROR_IMAGE_DIR

    background_tasks.add_task(
        copy_image_for_retraining, 
        file_path, 
        destination_dir,
        db_record.filename
    )
            
    # The API returns a response immediately, while the file copies.
    return
    


# @app.get("/lookedup-results/", tags=["Looked Up Images"])
# async def get_lookedup_results(request: Request)->List[URLClassificationResult]:
#     """
#     This method returns a list of all lookedUp images stored on the backend with a url of the image along with the class and confidence level
#     """
#     logging.info("get/lookedup-results endpoint was accessed!")
#     allFIles = os.listdir(IMAGE_DIRECTORY)
#     results = []
#     for fileName in allFIles:
#         resource_url = request.url_for('get_image', filename=fileName)

#         result = URLClassificationResult(
#             url= str(resource_url),
#             predicted_class=fileName.split('_')[-2],
#             status='success',
#             confidence_level=fileName.split('_')[-1].split('.')[0]
#         )
#         results.append(result)
    
#     return results

# @app.post("/retrain/{filename}", tags=["Retrain"])
# async def retrain_image(filename:str):
#     """
#     This is a method that allows you to input a filename of a image stored on the backend.
#     If the images is found, this image will be pushed for retraining.
#     """
#     logging.info("post/retrain endpoint was accessed!")
    
#     source_path = os.path.join(IMAGE_DIRECTORY, filename)
    
#     if not os.path.exists(source_path):
#         logging.error(f"Failed to find image file '{source_path}'")
#         raise HTTPException(status_code=404, detail="Image not found.")
    
#     predicted_class=filename.split('_')[-2]
    
#     if predicted_class == "studio":
#         os.makedirs(RETRAIN_ENVIRONMENT_DIR, exist_ok=True)
#         destination_path = os.path.join(RETRAIN_ENVIRONMENT_DIR, filename)
#     else:
#         os.makedirs(RETRAIN_STUDIO_DIR, exist_ok=True)
#         destination_path = os.path.join(RETRAIN_STUDIO_DIR, filename)
        
#     try:
#         shutil.move(source_path, destination_path)
#         logging.info(f"Successfully moved '{filename}' for retraining to '{destination_path}'")
#     except Exception as e:
#         logging.error(f"Failed to move file '{source_path}' to '{destination_path}': {e}")
#         raise HTTPException(status_code=500, detail="Failed to move file for retraining.")
    
#     return {
#         'status': 'success',
#     }
                
# test url: https://arper-cdn.thron.com/delivery/public/thumbnail/arper/d8d37df6-e39a-4285-8fed-991fd6fb7d74/jd4oic/std/280x280/0707_Arper_CATIFA46_CRO_PO00105_RX00643_0470.jpg
