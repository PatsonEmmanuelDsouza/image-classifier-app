from fastapi import FastAPI, HTTPException
from celery.result import AsyncResult
from worker import classify_images_from_urls_task
from baseModels import URLPayload, JobResponse, JobResultResponse

# ----- App -----
app = FastAPI(
    title="Binary Image Classification API",
    description="An API that allows you to make requests and post images, for an image posted to the the API response would be the predicted class of the image "
)

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



# test url: https://arper-cdn.thron.com/delivery/public/thumbnail/arper/d8d37df6-e39a-4285-8fed-991fd6fb7d74/jd4oic/std/280x280/0707_Arper_CATIFA46_CRO_PO00105_RX00643_0470.jpg