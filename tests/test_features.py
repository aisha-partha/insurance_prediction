
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from model.config.core import config
from model.processing.features import Mapper, OutlierHandler, ColumnDropperTransformer, AgeColTransformer




def test_gender_variable_mapper(sample_input_data):
    # Given
    mapper = Mapper(variables = config.model_config.gender_var, 
                    mappings = config.model_config.gender_mappings)
    assert sample_input_data[0].loc[1731, 'Gender'] == 'Female'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[1731, 'Gender'] == 1


