from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Request
from celery.result import AsyncResult
from .worker import classify_images_from_urls_task, download_and_classify_url
from .baseModels import URLPayload, JobResponse, JobResultResponse, ImageUrlPayload, URLClassificationResult

import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
import shutil
# ----- App -----
app = FastAPI(
    title="Binary Image Classification API",
    description="An API that allows you to make requests and post images, for an image posted to the the API response would be the predicted class of the image "
)

IMAGE_DIRECTORY = "lookedup_images"
RETRAIN_STUDIO_DIR = "retraining_dataset/studio"
RETRAIN_ENVIRONMENT_DIR = "retraining_dataset/environment"

# --- API Endpoints ---
@app.get('/')
def home():
    return {'message': "welcome to this tool!"}

@app.post("/classify-urls/", response_model=JobResponse, status_code=202, tags=["Classification"])
async def start_classification(payload: URLPayload):
    """
    Accepts a list of image URLs and starts a background classification job.
    Responds immediately with a job ID.
    """
    if not payload.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")

    # Start the Celery task in the background.
    # .delay() is the shortcut to send a task to the queue.
    task = classify_images_from_urls_task.delay(payload.urls)
    
    return {"job_id": task.id}


@app.get("/jobs/{job_id}", response_model=JobResultResponse, tags=["Classification"])
async def get_job_status(job_id: str):
    """
    Retrieves the status and result of a classification job by its ID.
    """
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

@app.post("/search-similar-products-imageUrl/", tags=["Image Search"])
async def search_similar_products(
    payload: ImageUrlPayload = Body(None)
):
    """
    function that allows you to search an image and get its classification
    """
    # check if paylaod and file are given, if neither both then throw an error
    if not (payload):
        raise HTTPException(
            status_code=400,
            detail="You must provide an 'image_url."
        )
    # predicting the type of image
    url = payload.image_url
    result = download_and_classify_url(url, save=True)
    return result

@app.get("/lookedup-images/{filename}", name = 'get_image', tags=["Images Folder"])
async def get_image(filename: str):
    """
    Custom get function to get the image stored on backend when you look up the file
    """
    file_path = os.path.join(IMAGE_DIRECTORY, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(file_path)

@app.get("/lookedup-results/", tags=["Looked Up Images"])
async def get_lookedup_results(request: Request)->List[URLClassificationResult]:
    """
    This method returns a list of all lookedUp images stored on the backend with a url of the image along with the class and confidence level
    """
    allFIles = os.listdir(IMAGE_DIRECTORY)
    results = []
    for fileName in allFIles:
        resource_url = request.url_for('get_image', filename=fileName)

        result = URLClassificationResult(
            url= str(resource_url),
            predicted_class=fileName.split('_')[2],
            status='success',
            confidence_level=fileName.split('_')[3].split('.')[0]
        )
        results.append(result)
    
    return results

@app.get("/retrain/{filename}", tags=["Retrain"])
async def retrain_image(filename:str):
    """
    This is a method that allows you to input a filename of a image stored on the backend.
    If the images is found, this image will be pushed for retraining.
    """
    source_path = os.path.join(IMAGE_DIRECTORY, filename)
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    
    predicted_class=filename.split('_')[2]
    
    if predicted_class == "studio":
        os.makedirs(RETRAIN_ENVIRONMENT_DIR, exist_ok=True)
        destination_path = os.path.join(RETRAIN_ENVIRONMENT_DIR, filename)
    else:
        os.makedirs(RETRAIN_STUDIO_DIR, exist_ok=True)
        destination_path = os.path.join(RETRAIN_STUDIO_DIR, filename)
        
    try:
        shutil.move(source_path, destination_path)
        print(f"Successfully moved '{filename}' to '{destination_path}'")
    except Exception as e:
        print(f"Error: The file '{source_path}' was not able to move to {destination_path}: {e}.")
    
    return {
        'status': 'success',
    }
                


# test url: https://arper-cdn.thron.com/delivery/public/thumbnail/arper/d8d37df6-e39a-4285-8fed-991fd6fb7d74/jd4oic/std/280x280/0707_Arper_CATIFA46_CRO_PO00105_RX00643_0470.jpg