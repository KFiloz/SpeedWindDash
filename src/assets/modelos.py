import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.linear_model import Lasso, SGDRegressor, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import psycopg2
from psycopg2 import Error


def reg_lineal(Xtrain,ytrain,Xtest,ytest):
    
    reg = Pipeline([
    ('scaler', StandardScaler()),  # Escalar las características
    ('regressor', LinearRegression())
            ])
    
    reg.fit(Xtrain,ytrain)
    predictReg = reg.predict(Xtest)
    mae_reg = metrics.mean_absolute_error(ytest, predictReg)
    mse_reg = metrics.mean_squared_error(ytest, predictReg)
    rmse_reg = np.sqrt(metrics.mean_squared_error(ytest, predictReg))
    
    return mae_reg, mse_reg, rmse_reg





