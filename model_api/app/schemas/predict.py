from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                "id": 123,
                "Gender": "Female",  
                "Age": 25, 
                "Driving_License": 1,
                "Region_Code": 1.0, 
                "Previously_Insured": 0,
                "Vehicle_Age": "< 1 Year",
                "Vehicle_Damage": "No",
                "Annual_Premium": 5000.0,
                "Policy_Sales_Channel": 4.0,
                "Vintage": 260,	

                    }
                ]
            }
        }