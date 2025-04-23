import json
import pickle
import pandas as pd

with open('D:/PyCharmProjects/GRES/models/sarimax/model.pickle', mode='rb') as filestream:
    model = pickle.load(filestream)

with open('D:/PyCharmProjects/GRES/models/sarimax/info.json', 'r', encoding='utf-8') as f:
    info = json.load(f)


def calculate(data: pd.DataFrame) -> float:
    data = data[['StationTempOutdoorAir', 'TurbineTempFeedWaterQ2',
             'StationCoalHumidity', 'StationCoalAsh',
             'StationConsumpNaturalFuel', 'Year', 'Season', 'DayOfWeek']]
    pred = model.forecast(data.shape[0], exog = data)
    return float(pred)

def get_r2() -> float:
    return float(info['R2'])

def get_mae() -> float:
    return float(info['MAE'])

def get_mse() -> float:
    return float(info['MSE'])

def get_rmse() -> float:
    return float(info['RMSE'])

def get_mape() -> float:
    return float(info['MAPE'])