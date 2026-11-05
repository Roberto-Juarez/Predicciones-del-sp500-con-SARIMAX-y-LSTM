# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:28:17 2026

@author: usuario
"""

from datetime import datetime,timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model
import joblib
import json
import streamlit as st
import plotly.graph_objects as go

gradient_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #111827 0%,
        #1f2937 100%
    );
}
</style>
"""
st.markdown(gradient_bg, unsafe_allow_html=True)

st.markdown("""
<style>

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(30, 41, 59, 0.9);
}

/* Expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: rgba(51, 65, 85, 0.8);
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.2);
}

/* Título del expander */
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #F8FAFC !important;
    font-weight: 600;
}

/* Texto interior */
[data-testid="stSidebar"] [data-testid="stExpander"] p,
[data-testid="stSidebar"] [data-testid="stExpander"] div,
[data-testid="stSidebar"] [data-testid="stExpander"] span {
    color: #E5E7EB !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Texto normal */
p, div, span, label {
    color: #E5E7EB !important;
}

/* Títulos */
h1, h2, h3 {
    color: #F8FAFC !important;
}

/* Métricas */
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: bold;
}

[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
}

/* Expander */
[data-testid="stExpander"] summary {
    color: #F8FAFC !important;
}

/* Barra superior */
header[data-testid="stHeader"] {
    background-color: #111827;
}

/* Toolbar superior */
[data-testid="stToolbar"] {
    background-color: #111827;
}

/* Botones del header */
[data-testid="stToolbar"] button {
    color: white !important;
}

/* Menú superior */
[data-testid="stToolbar"] svg {
    fill: white !important;
}

</style>
""", unsafe_allow_html=True)

try:
    with open("fecha_entrenamiento.json") as f:
        fecha_entrenamiento = json.load(f)["fecha"]
except FileNotFoundError:
    fecha_entrenamiento = "2000-01-01"

    
def necesita_reentrenamiento(fecha_entrenamiento):

    fecha_entrenamiento = datetime.strptime(
        fecha_entrenamiento,
        "%Y-%m-%d"
    ).date()

    fecha_actual = datetime.now().date()

    dias_transcurridos = (
        fecha_actual - fecha_entrenamiento
    ).days

    return dias_transcurridos >= 15

def entrenamiento_sarimax(train_precios_meses_12,test_precios_meses_12, 
                          train_retornos_meses_12, test_retornos_meses_12):
    model_precios = SARIMAX(train_precios_meses_12, order=(1,1,2), seasonal_order=(1,0,1,5))
    res_precios_guardado = model_precios.fit(disp=False)
    forecast_precios = res_precios_guardado.get_forecast(steps=len(test_precios_meses_12))
    pred_mean_precios = forecast_precios.predicted_mean
    mape_precios = np.mean( np.where(test_precios_meses_12.values != 0,  
                                     np.abs((test_precios_meses_12.values - pred_mean_precios.values) / test_precios_meses_12.values), 0)) * 100

    model_retornos = SARIMAX(train_retornos_meses_12, order=(1,0,2), seasonal_order=(1,0,1,5))
    res_retornos_guardado = model_retornos.fit(disp=False)
    forecast_retornos = res_retornos_guardado.get_forecast(steps=len(test_retornos_meses_12))
    pred_mean_retornos = forecast_retornos.predicted_mean
    rmse = np.sqrt(mean_squared_error(test_retornos_meses_12.values, pred_mean_retornos.values))
    mae = mean_absolute_error(test_retornos_meses_12.values, pred_mean_retornos.values)
    
    return res_precios_guardado, res_retornos_guardado, mape_precios, rmse, mae

class LSTMModel(nn.Module):
    def __init__ (self, input_size = 1, hidden_size = 50, num_layers = 2, horizonte = 1):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, horizonte )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        out = self.fc(out)
        return out.unsqueeze(-1)

def entrenamiento(x_train, y_train, x_test, y_test, scaler):

     model = LSTMModel()
     criterion = nn.MSELoss()
     optimizer = torch.optim.Adam(model.parameters(), lr= 0.005)
     
     epochs=100
     for epoch in range(epochs):
         model.train()
         optimizer.zero_grad()
         output= model(x_train)
         loss = criterion(output, y_train)
         loss.backward()
         optimizer.step()
         if epoch % 10 == 0:
             print(f" Ciclo: {epoch} | Pérdida: {loss.item()}")

     model.eval()
     with torch.no_grad():
         y_pred = model(x_test)
     
     y_test_inv = scaler.inverse_transform(y_test.numpy().reshape(-1,1))
     y_pred_inv = scaler.inverse_transform(y_pred.numpy().reshape(-1,1))
     
     rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
     mae = mean_absolute_error(y_test_inv, y_pred_inv)
     mape = (abs((y_test_inv- y_pred_inv) / y_test_inv).mean()) * 100
     
     return  model, scaler, rmse,  mae, mape    

if necesita_reentrenamiento(fecha_entrenamiento):
    
    df = yf.download('^GSPC', period='2y')
    daily = df['Close'].resample("B").mean()
    daily.to_csv("ultimo_dataset.csv", index=True)
    daily_drop = daily.dropna()
    daily_precios = np.log(daily_drop)
    daily_retornos_ = np.log(daily_drop).diff()
    daily_retornos = daily_retornos_.dropna()

    train_precios_meses_12 = daily_precios.iloc[-(252+42):-42]
    test_precios_meses_12 = daily_precios.iloc[-42:]

    train_retornos_meses_12 = daily_retornos.iloc[-(252+42):-42]
    test_retornos_meses_12 = daily_retornos.iloc[-42:]

    
    daily_array_precios = np.log(daily_drop.iloc[-273:]).values.reshape(-1,1)
    daily_array_retornos_ = np.log(daily_drop.iloc[-273:]).diff()
    daily_array_retornos = daily_array_retornos_.dropna().values.reshape(-1,1)

    torch.manual_seed(42)
    np.random.seed(42)

    scaler_precios = StandardScaler()
    scaler_retornos = StandardScaler()

    serie_precios = scaler_precios.fit_transform(daily_array_precios)
    serie_retornos = scaler_retornos.fit_transform(daily_array_retornos)

    secuencia = 21
    horizonte = 1
    xs_precios=[]
    ys_precios=[]
    xs_retornos=[]
    ys_retornos=[]

    for i in range(len(serie_precios)-secuencia-horizonte +1):
        x_ = serie_precios[i:(i+secuencia)]
        y_ = serie_precios[(i+secuencia):(i+secuencia+horizonte)]
        xs_precios.append(x_)
        ys_precios.append(y_)

    x = np.array(xs_precios)
    y = np.array(ys_precios)

    x_train_precios, y_train_precios = x[:-21], y[:-21] 
    x_test_precios, y_test_precios = x[-21:], y[-21:] 

    x_train_precios = torch.tensor(x_train_precios, dtype=torch.float32)
    y_train_precios = torch.tensor(y_train_precios, dtype=torch.float32)
    x_test_precios = torch.tensor(x_test_precios, dtype=torch.float32)
    y_test_precios = torch.tensor(y_test_precios, dtype=torch.float32)

    for i in range(len(serie_retornos)-secuencia-horizonte +1):
        x_ = serie_retornos[i:(i+secuencia)]
        y_ = serie_retornos[(i+secuencia):(i+secuencia+horizonte)]
        xs_retornos.append(x_)
        ys_retornos.append(y_)

    x = np.array(xs_retornos)
    y = np.array(ys_retornos)

    x_train_retornos, y_train_retornos = x[:-21], y[:-21] 
    x_test_retornos, y_test_retornos = x[-21:], y[-21:] 

    x_train_retornos = torch.tensor(x_train_retornos, dtype=torch.float32)
    y_train_retornos = torch.tensor(y_train_retornos, dtype=torch.float32)
    x_test_retornos = torch.tensor(x_test_retornos, dtype=torch.float32)
    y_test_retornos = torch.tensor(y_test_retornos, dtype=torch.float32)
    
    res_precios_guardado, res_retornos_guardado, mape_precios, rmse, mae = entrenamiento_sarimax(train_precios_meses_12,test_precios_meses_12, 
                          train_retornos_meses_12, test_retornos_meses_12)
    
    (model_precios_lstm, scaler_precios, rmse_precios, mae_precios, mape_precios_lstm) = entrenamiento(x_train_precios,
                                                                                           y_train_precios,x_test_precios,y_test_precios,scaler_precios)
    (model_retornos_lstm, scaler_retornos, rmse_retornos, mae_retornos, mape_retornos) = entrenamiento(x_train_retornos, 
                                                                                                       y_train_retornos,x_test_retornos,y_test_retornos,scaler_retornos)
   
    metricas = {"mape_precios_sarimax": float(mape_precios),
                "rmse_retornos_sarimax":float(rmse),
                "mae_retornos_sarimax": float(mae),
                "rmse_precios_lstm": float(rmse_precios),
                "mae_precios_lstm": float(mae_precios),
                "mape_precios_lstm": float(mape_precios_lstm),
                "rmse_retornos_lstm": float(rmse_retornos),
                "mae_retornos_lstm": float(mae_retornos),
                "mape_retornos_lstm": float(mape_retornos)}
    
    with open("metricas.json", "w") as f:
        json.dump(metricas, f)
    
    joblib.dump(res_precios_guardado, "sarimax_precios.pkl")
    joblib.dump(res_retornos_guardado,"sarimax_retornos.pkl")
    
    torch.save(model_precios_lstm.state_dict(), "model_precios_lstm.pth")
    torch.save(model_retornos_lstm.state_dict(),"model_retornos_lstm.pth")

    joblib.dump(scaler_precios,"scaler_precios.pkl")
    joblib.dump(scaler_retornos,"scaler_retornos.pkl")
    
    with open("fecha_entrenamiento.json","w") as f:
        json.dump({"fecha":
                datetime.now()
                .strftime("%Y-%m-%d")},f)
    fecha_entrenamiento = datetime.now().strftime("%Y-%m-%d")
    
else:
    
    res_precios_guardado = joblib.load(
    "sarimax_precios.pkl")

    res_retornos_guardado = joblib.load(
    "sarimax_retornos.pkl")

    model_precios_lstm = LSTMModel()
    model_precios_lstm.load_state_dict(
        torch.load("model_precios_lstm.pth")
    )
    model_precios_lstm.eval()
    
    model_retornos_lstm = LSTMModel()
    model_retornos_lstm.load_state_dict(
        torch.load("model_retornos_lstm.pth")
    )
    model_retornos_lstm.eval()
    
    scaler_precios = joblib.load(
        "scaler_precios.pkl"
    )

    scaler_retornos = joblib.load(
        "scaler_retornos.pkl"
    )
    
    with open("metricas.json") as f:
        metricas = json.load(f)

@st.cache_data
def cargar_datos_actualizados(fecha_entrenamiento,
                              _scaler_precios,
                              _scaler_retornos):

    try:
        daily = pd.read_csv("ultimo_dataset.csv", index_col=0, parse_dates=True)

    except FileNotFoundError:

        df = yf.download("^GSPC", period="2y")
        daily = df["Close"].resample("B").mean()
        daily.to_csv("ultimo_dataset.csv",index=True)

    fecha_actual = datetime.now().date()
    ultima_fecha_hist = pd.to_datetime(daily.index[-1]).date()

    df_nuevos_datos = yf.download("^GSPC", start=ultima_fecha_hist, end=fecha_actual, 
                                  progress=False)

    daily_nuevos = (df_nuevos_datos["Close"].resample("B").mean().dropna())
    daily_completo = pd.concat([daily, daily_nuevos])
    daily_completo = daily_completo[~daily_completo.index.duplicated(keep="last")]
    daily_completo.to_csv("ultimo_dataset.csv")

    daily_precios = np.log(daily_completo).values.reshape(-1, 1)
    daily_retornos = (np.log(daily_completo).diff().dropna().values.reshape(-1, 1))

    serie_precios = _scaler_precios.transform(daily_precios)
    serie_retornos = _scaler_retornos.transform(daily_retornos)

    precios_torch = torch.tensor(serie_precios,dtype=torch.float32)
    retornos_torch = torch.tensor(serie_retornos,dtype=torch.float32)

    return (daily_completo,precios_torch,retornos_torch)

daily_completo, precios_torch, retornos_torch = (cargar_datos_actualizados(
        fecha_entrenamiento,
        scaler_precios,
        scaler_retornos))

fecha_dt = datetime.strptime(fecha_entrenamiento, "%Y-%m-%d").date()
nueva_fecha = fecha_dt + timedelta(days=15)

st.header("Predicción del índice SP500")
st.write("Esta aplicación estima los resultados del SP500 usando SARIMAX y LSTM")

with st.sidebar.expander("Información sobre los modelos"):
    
    st.subheader("SARIMAX")
    st.caption(f"Entrenado: {fecha_entrenamiento}")


    st.metric(
        "MAPE Precios",
        f'{metricas["mape_precios_sarimax"]:.3f}%',
    )

    st.metric(
        "RMSE Retornos",
        f'{metricas["rmse_retornos_sarimax"]:.4f}'
    )

    st.metric(
        "MAE Retornos",
        f'{metricas["mae_retornos_sarimax"]:.4f}'
    )

    st.subheader("LSTM")
    st.caption(f"Entrenado: {fecha_entrenamiento}")

    st.metric(
        "MAPE Precios",
        f'{metricas["mape_precios_lstm"]:.3f}%'
    )

    st.metric(
        "RMSE Retornos",
        f'{metricas["rmse_retornos_lstm"]:.4f}'
    )

    st.metric(
        "MAE Retornos",
        f'{metricas["mae_retornos_lstm"]:.4f}'
    )

    st.subheader(f"Próxima fecha de entrenamiento: {nueva_fecha}")
    dias_restantes = (nueva_fecha - datetime.now().date()).days
    st.caption(f"Faltan {dias_restantes} días para el próximo reentrenamiento")


def predicciones(dias):
    daily_precios = np.log(daily_completo)
    daily_retornos_ = np.log(daily_completo).diff()
    daily_retornos = daily_retornos_.dropna()
    

    fit_precios = res_precios_guardado.apply(daily_precios)
    forecast_precios = fit_precios.get_forecast(steps= dias)
    pred_mean_precios = forecast_precios.predicted_mean
    pred_ci_precios = forecast_precios.conf_int(alpha=0.01)
    precio_predicho = np.exp(pred_mean_precios)
    pred_ci_precios_predicho = np.exp(pred_ci_precios)

    fit_retornos = res_retornos_guardado.apply(daily_retornos)
    forecast_retornos = fit_retornos.get_forecast(steps= dias)
    pred_mean_retornos = forecast_retornos.predicted_mean
    pred_ci_retornos = forecast_retornos.conf_int(alpha=0.01)
    retorno_acumulado = np.exp(pred_mean_retornos.cumsum().iloc[-1]) - 1

    residuals = res_retornos_guardado.resid
    residuals_garch = residuals*100
    garch = arch_model(residuals_garch, vol='GARCH', p=1, q=1)
    fit_garch = garch.fit(disp='off')

    forecast_vol = fit_garch.forecast(horizon=dias)
    pred_var = forecast_vol.variance.iloc[-1].values
    pred_vol = np.sqrt(pred_var) / 100
    vol_acumulada = np.sqrt(np.sum(pred_vol**2))

    future_dates = pd.date_range(
    start=daily_completo.index[-1] + pd.offsets.BusinessDay(1),
    periods=dias,
    freq="B"
)

    p_low_col = pred_ci_precios.columns[0]
    p_upp_col = pred_ci_precios.columns[1]

    r_low_col = pred_ci_retornos.columns[0]
    r_upp_col = pred_ci_retornos.columns[1]

    tabla_future = pd.DataFrame({
    "Fecha": future_dates,
    "Precio":  precio_predicho.values,
    "Retorno_diario": pred_mean_retornos.values,
    "Volatilidad": pred_vol,
    "Rango precios bajo": pred_ci_precios_predicho[p_low_col].values,
    "Rango precios alto": pred_ci_precios_predicho[p_upp_col].values,
    "Rango retornos bajo": pred_ci_retornos[r_low_col].values,
    "Rango retornos alto": pred_ci_retornos[r_upp_col].values        
})
    ratio_sarimax = (
    retorno_acumulado / vol_acumulada
    if vol_acumulada > 0
    else np.nan
)
    ultimo = tabla_future.iloc[-1]


    resultado = {
        "Fecha": ultimo["Fecha"],
        "Precio": ultimo["Precio"],
        "Retorno": retorno_acumulado,
        "Volatilidad_acumulada": vol_acumulada,
        "Ratio_Retorno_Riesgo": ratio_sarimax
    }
    return resultado, tabla_future, daily_completo, daily_retornos

def predicciones_por_dia(model, dias, data, scaler):

    ventana = data[-21:].clone()
    predicciones = []

    for _ in range(dias):

        with torch.no_grad():
            pred = model(ventana.unsqueeze(0))

        valor = pred.squeeze().item()

        predicciones.append(valor)

        nueva_ventana = ventana[1:]
        nuevo_valor = torch.tensor([[valor]], dtype=torch.float32)

        ventana = torch.cat([nueva_ventana, nuevo_valor], dim=0)


    predicciones_inv = np.array(predicciones).reshape(-1,1)
    predicciones_inv = scaler.inverse_transform(predicciones_inv)


    future_dates = pd.date_range(
    start=daily_completo.index[-1] + pd.offsets.BusinessDay(1),
    periods=dias,
    freq="B")

    return  predicciones_inv, future_dates

dias = st.slider(
    label="Selecciona el rango de días:",
    min_value=1,
    max_value=100,
    value=10)


resultado, tabla, daily_completo, daily_retornos =predicciones(dias) 

pre_precios = predicciones_por_dia(model_precios_lstm, dias, precios_torch, scaler_precios)
pre_retornos = predicciones_por_dia(model_retornos_lstm, dias, retornos_torch, scaler_retornos)       

predic_precios, calendario = pre_precios
predic_retornos, calendario = pre_retornos

retorno_acumulado = (np.exp(predic_retornos.sum()) - 1) 
predicciones_precio = np.exp(predic_precios)

vol_acumulada=  resultado["Volatilidad_acumulada"]

ratio_lstm = (
retorno_acumulado / vol_acumulada
if vol_acumulada > 0
else np.nan)


fecha_predicción =  tabla["Fecha"].iloc[-1].strftime("%d-%b-%Y")

st.markdown(f"Fecha objetivo: {fecha_predicción}")

col1, col2, col3 = st.columns(3)

col1.metric("Precio actual", f"{daily_completo["^GSPC"].iloc[-1]:.2f}")
col2.metric("Volatilidad",f"{resultado["Volatilidad_acumulada"]:.2f}")

   
fig = go.Figure()
        
daily_completo = daily_completo.dropna()

# Histórico
fig.add_trace(go.Scatter(x=daily_completo.tail(126).index, y=daily_completo.tail(126).iloc[:,0],
                mode="lines",
                name="precios",
                line=dict(color="blue"),showlegend=False)) 
    
# SARIMAX
fig.add_trace(go.Scatter(x=tabla["Fecha"], y=tabla["Precio"],
              mode="lines", name=f"SARIMAX:  ${resultado['Precio']:.2f}",line=dict(color="red")))

# LSTM
fig.add_trace(go.Scatter(x=calendario, y=predicciones_precio.flatten(),mode="lines",
                name=f"LSTM: ${predicciones_precio[-1,-1]:.2f}",line=dict(color="green")))

        # Banda inferior
fig.add_trace(go.Scatter(x=tabla["Fecha"],y=tabla["Rango precios bajo"],
                                mode="lines",line=dict(width=0),showlegend=False))

        # Banda superior
fig.add_trace(go.Scatter(x=tabla["Fecha"],y=tabla["Rango precios alto"],
               mode="lines",fill="tonexty", fillcolor="rgba(255,105,180,0.25)",
               line=dict(width=0),name="IC 99%"))
           
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.08)"
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.08)"
)
        
fig.update_layout(
    template="plotly_dark",
    height=300,
    margin=dict(l=20, r=20,t=20,b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    xaxis_title="Fecha",
    yaxis_title="Precio",
    legend=dict(
        orientation="h",
        y=1.02,
        x=0,
        font=dict(color="white", size=15),
        traceorder="normal"))

st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)

col1.metric("Último retorno", f"{daily_retornos['^GSPC'].iloc[-1]:.2%}")
col2.metric("Ratio SARIMAX", f"{resultado['Ratio_Retorno_Riesgo']:.2f}")
col3.metric("Ratio LSTM",f"{ratio_lstm:.2f}")
     

fig_2 = go.Figure()

# Histórico
fig_2.add_trace(go.Scatter(x=daily_retornos.tail(126).index, y=daily_retornos.tail(126).iloc[:,0],
                mode="lines",name="retornos",
                line=dict(color="blue"),showlegend=False))   
# SARIMAX
fig_2.add_trace(go.Scatter(x=tabla["Fecha"], y=tabla["Retorno_diario"],
              mode="lines", name=f"SARIMAX:  {resultado['Retorno']:.2%}",line=dict(color="red")))

# LSTM
fig_2.add_trace(go.Scatter(x=calendario, y=predic_retornos.flatten(),mode="lines",
                name=f"LSTM: {retorno_acumulado:.2%}",line=dict(color="green")))

        # Banda inferior
fig_2.add_trace(go.Scatter(x=tabla["Fecha"],y=tabla["Rango retornos bajo"],
                                mode="lines",line=dict(width=0),showlegend=False))

        # Banda superior
fig_2.add_trace(go.Scatter(x=tabla["Fecha"],y=tabla["Rango retornos alto"],
               mode="lines",fill="tonexty", fillcolor="rgba(255,105,180,0.25)",
               line=dict(width=0),name="IC 99%"))
         
        
fig_2.update_xaxes(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.08)"
)

fig_2.update_yaxes(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.08)"
)
        
fig_2.update_layout(
    template="plotly_dark",
    height=300,
    margin=dict(l=20, r=20,t=20,b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    xaxis_title="Fecha",
    yaxis_title="Precio",
    legend=dict(
        orientation="h",
        y=1.02,
        x=0,
        font=dict(color="white", size=15),
        traceorder="normal"))

st.plotly_chart(fig_2, use_container_width=True)