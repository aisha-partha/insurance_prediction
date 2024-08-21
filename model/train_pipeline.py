import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from model.config.core import config
from model.pipeline import cross_insurance_pipe
from model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name = config.app_config.training_data_file)
    
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config.features],     # predictors
        data[config.model_config.target],       # target
        test_size = config.model_config.test_size,
        random_state=config.model_config.random_state,   # set the random seed here for reproducibility
    )

    # Pipeline fitting
    cross_insurance_pipe.fit(X_train, y_train)
    y_pred = cross_insurance_pipe.predict(X_test)

    # Calculate the score/error
    print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)
    print("F1-score(in %):", f1_score(y_test, y_pred)*100)
    

    # persist trained model
    save_pipeline(pipeline_to_persist = cross_insurance_pipe)
    
if __name__ == "__main__":
    run_training()