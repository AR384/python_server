from typing import Generic, TypeVar, Optional
from pydantic import BaseModel
T = TypeVar('T')

class ApiResponseDTO(BaseModel, Generic[T]):
    status: str
    message: str
    data: Optional[T] #null이 될수있는 Optional 선언

class ImageUploadResponseDTO(BaseModel):
    jobid: str
    username: str

class ImageProcessResultDTO(BaseModel):
    image_base64:str
    message:str
    poly:list
    names:list
    type:list
    viewSize:list
    
class ImagePermitResponseDTO(BaseModel):
    jobid: str
    selectedIdx:list
    selectedname:list
    
class ImagePermitRequestDTO(BaseModel):
    jobid: str
    selectedIdx:list
    selectedname:list

class ImagePermitResultDTO(BaseModel):
    jobid: str
    selectedIdx:list
    selectedname:list
    image_base64:str
    viewSize:list
    amount:list
