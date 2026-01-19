"""
Utility functions for HAR Classification project.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(X_train_split, y_train_split):
    # set different parameters to test
    param_grid = {
        'n_estimators': [150,200,300],
        'max_depth': [30,40,None], # can change these to anything and add more to test more combos
        'min_samples_split': [2,3,4]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    # find best parameter set 
    grid_search.fit(X_train_split, y_train_split)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return grid_search



