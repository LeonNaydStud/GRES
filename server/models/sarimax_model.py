import json
import pickle
import pandas as pd
from pathlib import Path

# Определяем базовый путь к проекту
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Пути к файлам модели SARIMAX
MODEL_PATH = BASE_DIR / 'models' / 'sarimax' / 'model.pickle'
INFO_PATH = BASE_DIR / 'models' / 'sarimax' / 'info.json'

with open(MODEL_PATH, mode='rb') as filestream:
    model = pickle.load(filestream)

with open(INFO_PATH, 'r', encoding='utf-8') as f:
    info = json.load(f)


def calculate(data: pd.DataFrame) -> float:
    data = data[['StationTempOutdoorAir', 'TurbineTempFeedSteamQ2',
             'StationCoalHumidity', 'StationCoalAsh',
             'StationConsumpNaturalFuel', 'Year', 'Season', 'DayOfWeek']]
    pred = model.forecast(data.shape[0], exog=data)
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