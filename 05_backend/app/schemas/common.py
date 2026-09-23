from pydantic import BaseModel
from typing import Optional, List

class ErrorResponse(BaseModel):
    error_code: str
    message: str
