import json
from catboost import CatBoostRegressor
import pandas as pd
from pathlib import Path

# Определяем базовый путь к проекту
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Пути к файлам модели SARIMAX
MODEL_PATH = BASE_DIR / 'models' / 'catboost' / 'model.cbm'
INFO_PATH = BASE_DIR / 'models' / 'catboost' / 'info.json'

# Загрузка CatBoost модели
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# Загрузка метаданных
with open(INFO_PATH, 'r', encoding='utf-8') as filestream:
    info = json.load(filestream)

def calculate(data: pd.DataFrame) -> float:
    data = data[['StationTempOutdoorAir', 'TurbineTempFeedWaterQ2',
                 'StationCoalHumidity', 'StationCoalAsh',
                 'StationConsumpNaturalFuel', 'Year', 'Season', 'Month', 'DayOfYear', 'Day', 'DayOfWeek']]
    pred = model.predict(data)
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
