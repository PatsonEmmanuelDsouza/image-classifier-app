from pydantic import BaseModel, HttpUrl
from typing import List, Optional

# For the immediate response after starting a job
class JobResponse(BaseModel):
    job_id: str

# For the response when checking a job's status and result
class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[List[dict]] = None
    
class URLPayload(BaseModel):
    urls: List[str]
    
    
class ImageUrlPayload(BaseModel):
    image_url: HttpUrl
    
class URLClassificationResult(BaseModel):
    url: str
    status: str
    predicted_class: str
    confidence_level: str