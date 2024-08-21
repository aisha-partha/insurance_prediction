import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from model.config.core import config
from model.processing.features import AgeColTransformer, Mapper, ColumnDropperTransformer, OutlierHandler

cross_insurance_pipe = Pipeline([


    
    ######### Mapper ###########
    ('map_vehicle_age', Mapper(variables = config.model_config.vehicle_age_var, mappings = config.model_config.vehicle_age_mappings)),
    
    ('map_vehicle_damage', Mapper(variables = config.model_config.vehicle_damage_var, mappings = config.model_config.vehicle_damage_mappings)),
    
    ('map_gender', Mapper(variables = config.model_config.gender_var, mappings = config.model_config.gender_mappings)),

    
    ######## Handle outliers ########
    ('handle_outliers_annual_premium', OutlierHandler(variable = config.model_config.annual_premium_var)),


    ######## new features ########
    ('age_col_transform', AgeColTransformer(feature = config.model_config.age_var)),

    # Scale features
    ('scaler', StandardScaler()),
    
    # Regressor
    ('model_rf', RandomForestClassifier(n_estimators = config.model_config.n_estimators, 
                                       max_depth = config.model_config.max_depth,
                                      random_state = config.model_config.random_state))
    
    ])