import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import models.catboost_model as cb
import models.sarimax_model as sm

try:
    icon = Image.open(Path(__file__).parent / "icon.png")
    st.set_page_config(page_icon=icon, page_title="Выходная мощность ГРЭС", layout="wide")
except FileNotFoundError:
    st.set_page_config(page_title="Выходная мощность ГРЭС", layout="wide")

with st.sidebar:
    model_type = st.radio("**Тип модели:**" ,("CatBoost", "SARIMAX", "Обе модели"), horizontal=True)
    Date = st.date_input("**Дата**")
    st.header("Температурные параметры:")
    StationTempOutdoorAir = st.number_input('Температура вне станции (°C)',
                                          min_value=-50.0, max_value=50.0, value=-7.95)
    st.write("Температура пара в турбинах (°C):")
    c1, c2 = st.columns(2)
    with c1:
        TurbineTemp1 = st.number_input('Турбина 1',
                                           min_value=0.0, max_value=500.0, value=207.65)
        TurbineTemp2 = st.number_input('Турбина 2',
                                                 min_value=0.0, max_value=500.0, value=207.65)
        TurbineTemp3 = st.number_input('Турбина 3',
                                       min_value=0.0, max_value=500.0, value=207.65)
        TurbineTemp4 = st.number_input('Турбина 4',
                                       min_value=0.0, max_value=500.0, value=207.65)
    with c2:
        TurbineTemp5 = st.number_input('Турбина 5',
                                           min_value=0.0, max_value=500.0, value=207.65)
        TurbineTemp6 = st.number_input('Турбина 6',
                                                 min_value=0.0, max_value=500.0, value=207.65)
        TurbineTemp7 = st.number_input('Турбина 7',
                                       min_value=0.0, max_value=500.0, value=207.65)
        TurbineTemp8 = st.number_input('Турбина 8',
                                       min_value=0.0, max_value=500.0, value=207.65)
    st.header("Расход:")
    StationConsumpNaturalFuel = st.number_input('Расход натурального топлива за прошлый день (т)',
                                                min_value=0.0, max_value=6000.0, value=4097.91)
    st.header("Параметры качества угля:")
    c3, c4 = st.columns(2)
    with c3:
        StationCoalHumidity = st.number_input('Влажность (%)',
                                              min_value=0.0, max_value=100.0, value=9.51)
    with c4:
        StationCoalAsh = st.number_input('Зольность (%)',
                                         min_value=0.0, max_value=100.0, value=23.36)
    button = st.button("Предсказать выходную мощность")

def month_to_season(month):
    seasons = {
        1: 1, 2: 1, 3: 1, 12: 1,  # Winter
        4: 2, 5: 2,  # Spring
        6: 3, 7: 3, 8: 3,  # Summer
        9: 4, 10: 4, 11: 4  # Autumn
    }
    return seasons.get(month, 1)

def prepare_input_data(data: pd.DataFrame) -> pd.DataFrame:
    data['Year'] = data.index.year
    data['Season'] = data.index.month.map(month_to_season)
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['DayOfYear'] = data.index.dayofyear
    data['DayOfWeek'] = data.index.dayofweek + 1
    return data

def metric_model(model):
    st.write(f"Оценка R²: {model.get_r2():.3f}")
    st.write(f"Оценка MAE: {model.get_mae():.3f}")
    st.write(f"Оценка MSE: {model.get_mse():.3f}")
    st.write(f"Оценка RMSE: {model.get_rmse():.3f}")
    st.write(f"Оценка MAPE: {model.get_mape():.3f}")

st.title("Предсказание выходной мощности ГРЭС")

placeholder = st.empty()
if "button_pressed" not in st.session_state:
    placeholder.info("Выберите параметры станции и нажмите кнопку \"Предсказать выходную мощность\" для предсказания выходной мощности")

if button:
    try:
        turb = np.array([TurbineTemp1, TurbineTemp2, TurbineTemp3, TurbineTemp4,
                         TurbineTemp5, TurbineTemp6, TurbineTemp7, TurbineTemp8])
        turb = np.quantile(turb, q=0.5)

        df = pd.DataFrame({
            'Date': pd.to_datetime([Date]),
            'StationTempOutdoorAir': [float(StationTempOutdoorAir)],
            'TurbineTempFeedSteamQ2': [float(turb)],
            'StationCoalHumidity': [float(StationCoalHumidity)],
            'StationCoalAsh': [float(StationCoalAsh)],
            'StationConsumpNaturalFuel': [float(StationConsumpNaturalFuel)]
        }).set_index('Date')

        placeholder.empty()

        st.write("Входные данные:")
        st.dataframe(df, column_config={
            'Date' : "Дата",
            'StationTempOutdoorAir': "Температура вне станции(°C)",
            'TurbineTempFeedSteamQ2': "Температура пара в турбинах(°C)",
            'StationCoalHumidity': "Влажность угля(%)",
            'StationCoalAsh': "Зольность угля(%)",
            'StationConsumpNaturalFuel': "Расход натурального топлива(т)"})

        if model_type == "CatBoost":
            df_processed = prepare_input_data(df.copy())
            power = np.round(cb.calculate(df_processed), 2)
            col1, col2 = st.columns(2)
            with col1:
                st.header("Метрики оценки модели")
                metric_model(cb)
            with col2:
                st.header(f"Результат модели {model_type}")
                st.metric(label="Выходная мощность в час", value=f"{power:.2f} МВт",
                          help="Прогнозируемая выходная мощность на основе входных параметров")
                st.metric(label="Выходная мощность в сутки", value=f"{24 * power:.2f} МВт",
                          help="Прогнозируемая выходная мощность на основе входных параметров")
        elif model_type == "SARIMAX":
            df_processed = prepare_input_data(df.copy())
            power = np.round(sm.calculate(df_processed), 2)
            col1, col2 = st.columns(2)
            with col1:
                st.header("Метрики оценки модели")
                metric_model(sm)
            with col2:
                st.header(f"Результат модели {model_type}")
                st.metric(label="Выходная мощность в час", value=f"{power:.2f} МВт",
                          help="Прогнозируемая выходная мощность на основе входных параметров")
                st.metric(label="Выходная мощность в сутки", value=f"{24 * power:.2f} МВт",
                          help="Прогнозируемая выходная мощность на основе входных параметров")
        else:
            df_processed = prepare_input_data(df.copy())
            power_cat = np.round(cb.calculate(df_processed), 2)
            power_sm = np.round(sm.calculate(df_processed), 2)
            st.header("Результаты моделей")
            col1, col2 = st.columns(2)
            with col1:
                st.write("<h4>CatBoost</h4>", unsafe_allow_html=True)
                st.metric(label="Выходная мощность в час", value=f"{power_cat:.2f} МВт",
                          help="Прогнозируемая выходная мощность на основе входных параметров")
                st.metric(label="Выходная мощность в сутки", value=f"{24 * power_cat:.2f} МВт",
                          help="Прогнозируемая выходная мощность на основе входных параметров")
            with col2:
                st.write("<h4>SARIMAX</h4>", unsafe_allow_html=True)
                st.metric(label="Выходная мощность в час", value=f"{power_sm:.2f} МВт",
                          help="Прогнозируемая выходная мощность на основе входных параметров")
                st.metric(label="Выходная мощность в сутки", value=f"{24 * power_sm:.2f} МВт",
                          help="Прогнозируемая выходная мощность на основе входных параметров")
            st.header("Метрики оценки модели")
            col3, col4 = st.columns(2)
            with col3: metric_model(cb)
            with col4: metric_model(sm)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")