import os
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor
import statsmodels.api as sm
import logging
from pathlib import Path

# Определяем базовый путь к проекту
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_csv(BASE_DIR / 'data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.resample('D').mean()

def add_time_features(data):
    data = data.copy()
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['DayOfYear'] = data.index.dayofyear
    return data
df = add_time_features(df)

def calculate_metrics(y_true, y_pred):
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }
    for name, value in metrics.items():
        logger.info(f'{name} score: {value:.4f}')
    return metrics


def save_model_info(base_path, info, model, model_type):
    try:
        os.makedirs(base_path, exist_ok=True)

        with open(os.path.join(base_path, 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

        if model_type == 'catboost':
            model.save_model(os.path.join(base_path, 'model.cbm'))
        elif model_type == 'sarimax':
            with open(os.path.join(base_path, 'model.pickle'), 'wb') as f:
                pickle.dump(model, f)

        logger.info(f"Модель успешно сохранена в {base_path}!")
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели: {e}")


def create_catboost(train_date='2015-01-01', r2_threshold=0.85):
    try:
        target = ['TurbinePowerSum']
        features = ['StationTempOutdoorAir', 'TurbineTempFeedSteamQ2', 'StationCoalHumidity',
                    'StationCoalAsh', 'StationConsumpNaturalFuel', 'Year',
                    'Season', 'Month', 'DayOfYear', 'Day', 'DayOfWeek']
        col_cat = ['Year', 'Season', 'Month', 'DayOfYear', 'Day', 'DayOfWeek']

        # Преобразование категориальных признаков
        for col in col_cat:
            df[col] = df[col].astype('int')

        train = df.loc[df.index < train_date]
        test = df.loc[df.index >= train_date]

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            random_seed=42,
            early_stopping_rounds=50,
            task_type='CPU'
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            cat_features=col_cat,
            verbose=100
        )

        y_pred = pd.Series(model.predict(X_test), index=X_test.index)
        metrics = calculate_metrics(y_test, y_pred)

        if metrics['R2'] > r2_threshold:
            feature_importance = model.get_feature_importance()
            feature_importance_dict = dict(zip(features, feature_importance.tolist()))

            info = {
                'type': 'catboost',
                'features': features,
                'target': target,
                **metrics,
                'params': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                           for k, v in model.get_params().items()},
                'feature_importance': feature_importance_dict
            }

            model_path = BASE_DIR / 'models' / 'catboost'
            save_model_info(model_path, info, model, 'catboost')
        else:
            logger.info(f"R2 <= {r2_threshold} — модель не сохранена.")

    except Exception as e:
        logger.error(f"Ошибка в create_catboost: {e}")


def create_sarimax(train_date='2015-01-01', r2_threshold=0.85, order=(0, 0, 0)):
    try:
        target = ['TurbinePowerSum']
        features = ['StationTempOutdoorAir', 'TurbineTempFeedSteamQ2', 'StationCoalHumidity',
                    'StationCoalAsh', 'StationConsumpNaturalFuel', 'Year',
                    'Season', 'DayOfWeek']

        train = df.loc[df.index < train_date]
        test = df.loc[df.index >= train_date]

        mod = sm.tsa.statespace.SARIMAX(
            endog=train[target],
            exog=train[features],
            trend='c',
            order=order,
            freq='D'
        )

        res = mod.fit(disp=False)
        y_pred = res.forecast(steps=len(test), exog=test[features])
        metrics = calculate_metrics(test[target], y_pred)

        if metrics['R2'] > r2_threshold:
            info = {
                'type': 'sarimax',
                'features': features,
                'target': target,
                **metrics,
                'params': res.params.to_dict(),
                'summary': str(res.summary())
            }

            model_path = BASE_DIR / 'models' / 'sarimax'
            save_model_info(model_path, info, res, 'sarimax')
        else:
            logger.info(f"R2 <= {r2_threshold} — модель не сохранена.")

    except Exception as e:
        logger.error(f"Ошибка в create_sarimax: {e}")

def main():
    create_catboost()
    create_sarimax()

if __name__ == '__main__':
    main()