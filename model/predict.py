import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from model import __version__ as _version
from model.config.core import config
from model.pipeline import cross_insurance_pipe
from model.processing.data_manager import load_pipeline
from model.processing.data_manager import pre_pipeline_preparation
from model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
cross_insurance_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    

    validated_data=validated_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = cross_insurance_pipe.predict(validated_data)

    #results = {"predictions": predictions,"version": _version, "errors": errors}
    results = {"predictions": int(round(predictions[0])),"version": _version, "errors": errors}        #round to nearest integer
    print(results)
    if not errors:

        predictions = cross_insurance_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'id':[123],'Gender':['Female'],'Age':[49],'Driving_License':[1],'Region_Code':[41.0],'Previously_Insured':[0],
                'Vehicle_Damage':['Yes'],'Vehicle_Age':['1-2 Year'],'Annual_Premium':[24783.0],'Policy_Sales_Channel':[4.0],
                'Vintage':[208]}
    
    make_prediction(input_data=data_in)
    