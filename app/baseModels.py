from pydantic import BaseModel, HttpUrl
from typing import List, Optional

# For the immediate response after starting a job
class JobResponse(BaseModel):
    # when multiple urls are passed to be classified, the response is just a job_id
    job_id: str

class URLClassificationResult(BaseModel):
    # image classification result of a URL
    url: str
    status: str
    predicted_class: str
    confidence_level: str
    # local_url: Optional[str] = None
    
class FileClassificationResult(BaseModel):
    # image classification result of a URL
    fileName: str
    status: str
    confidence_level: str
    predicted_class: str

class FileDownloadPredictionResult(BaseModel):
    local_file_name: str
    current_day_dir: str
    confidence_level: str
    predicted_class: str
    
class JobResultResponse(BaseModel):
    # this is the standard response for a job 
    job_id: str
    status: str
    result: Optional[List[dict]] = None
    
class URLPayload(BaseModel):
    # this class is used for multiple image urls that need to be classified as a job
    urls: List[HttpUrl]
    
    
class ImageUrlPayload(BaseModel):
    # this class is used to specify the payload of a single image url, this could be used for classifying an image or updating the status of an image
    image_url: HttpUrl