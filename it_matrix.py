import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt

import os

import pdb


#shift the temporal dimension of the daily_input matrix
def shift_series_(s, shift_range,t_unit):

    s_shifts = [s.shift(-t_unit * shift, freq='D').rename(f'{s.name}_{shift}') for shift in range(*shift_range)]
    return pd.concat(s_shifts, axis=1)


#define the input-target matrix for the model (necessary to chain the past year meteo+snow variables while the target is just the last 30 days discharge average.
def create_it_matrix(daily_input, t_length,t_unit):

    # This function takes as input the daily temperature, precipitation and runoff and generates the input-target matrix

    # Read the daily input and extract runoff, evaporation, temperature and precipitation dataframe
    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)
    runoff = daily_input[['Q']]
    temp = daily_input[[c for c in daily_input.columns if c[0] == 'T']]
    prec = daily_input[[c for c in daily_input.columns if c[0] == 'P']]
    evap = daily_input[[c for c in daily_input.columns if c[0] == 'E']]
    snow = daily_input[[c for c in daily_input.columns if c[0] == 'S']]
    run  = daily_input[[c for c in daily_input.columns if c[0] == 'R']]


    output = []
    # Compute the t_unit days average runoff
    runoff_t_unit = runoff.rolling(30, min_periods=30).mean()
    output.append(runoff_t_unit)
    
    
    # Compute the t_unit days average temperature for the last year.
    if not temp.empty:
        temp_t_unit = temp.rolling(t_unit, min_periods=t_unit).mean()
        temp_t_unit = pd.concat([shift_series_(temp_t_unit.loc[:, col], (-t_length + 1, 1),t_unit) for col in temp_t_unit], axis=1)
        output.append(temp_t_unit)

    # Compute the t_unit days average snow water equivalent for the last year.
    if not snow.empty:
        snow_t_unit = snow.rolling(t_unit, min_periods=t_unit).mean()
        snow_t_unit = pd.concat([shift_series_(snow_t_unit.loc[:, col], (-t_length + 1, 1),t_unit) for col in snow_t_unit], axis=1)
        output.append(snow_t_unit)

    # Compute the t_unit days sum precipitation for the last year.
    if not prec.empty:
        prec_t_unit = prec.rolling(t_unit, min_periods=t_unit).sum()
        prec_t_unit = pd.concat([shift_series_(prec_t_unit.loc[:, col], (-t_length + 1, 1),t_unit) for col in prec_t_unit], axis=1)
        output.append(prec_t_unit)
        
    # Compute the t_unit days sum evapotranspiration for the last year.
    if not evap.empty:
        evap_t_unit = evap.rolling(t_unit, min_periods=t_unit).sum()
        evap_t_unit = pd.concat([shift_series_(evap_t_unit.loc[:, col], (-t_length + 1, 1),t_unit) for col in evap_t_unit], axis=1)
        output.append(evap_t_unit)
    #pdb.set_trace()
    
    # Create the input-target matrix
    return pd.concat(output, axis=1).dropna()



#define the input matrix for the prediction phase (necessary to chain the past year meteo+snow variables)
def create_in_matrix(daily_input, t_length, t_unit):

    # Compute the t_unit days average temperature
    if not daily_input.empty:
        daily_input_t_unit = daily_input.rolling(t_unit, min_periods=t_unit).mean()
        output = pd.concat([shift_series_(daily_input_t_unit[col], (-t_length + 1, 1),t_unit) for col in daily_input_t_unit], axis=1)

    return output;


