import warnings
warnings.filterwarnings("ignore")

import dash
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
from assets.database import DatabaseManager
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Usar Database para obtener datos
#with DatabaseManager("BDK_owner", "Qde9y0ftCPVg", "ep-rapid-recipe-a57yu1fp.us-east-2.aws.neon.tech", "5432", "BDK") as db:
    #data = db.fetch_data("SELECT * FROM wind")
data = pd.read_csv("D:/Dataviz/SpeedWind/SpeedWindDash/src/data/wind_dataset2.csv")

dash.register_page(__name__, name='3-Arima - Sarima', title='Wind | 3-Arima')


from assets.fig_layout import my_figlayout, train_linelayout, test_linelayout, pred_linelayout, my_linelayout
from assets.acf_pacf_plots import acf_pacf

numbers = list(range(10))
_opts = [{'label': str(num), 'value': num} for num in numbers]

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3(['Arima - Sarima Modelo: Fit & Prediction'])], width=12, className='row-titles')
    ]),
########## train-test split  SLIDER
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([html.P(['Porcentaje de ',html.B(['Entrenamiento'])], className='par')], width = 4),
        dbc.Col([
            html.Div([
                dcc.Slider(50, 95, 5, value=80, marks=None, tooltip={"placement": "bottom", "always_visible": True}, id='train-slider6', persistence=True, persistence_type='session')
            ], className = 'slider-div')
        ], width = 3),
        dbc.Col([], width = 3),
         ]),
    dbc.Row([
            dbc.Col([], width = 3),
        ]),
   
       ####### BOTON CALCULAR HYPERPARAMETRO #######
    dbc.Row([
        dbc.Col([], width=3),
        dbc.Col([
            html.P(['MODELO : ', html.B([], id='comb-nr6')], className='par')
        ], width=3),
        dbc.Col([
            html.Button('Start Entrenamiento', id='start-gs15', n_clicks=0, title='The grid search may take several minutes', className='my-button')
        ], width=3, style={'text-align':'left', 'margin':'5px 1px 1px 1px'}),
        dbc.Col([], width=3)
    ]),
    html.Br(),
    ########################
       # MOSTRAR RESULTADO AIC
    dbc.Row([
        dbc.Col([], width = 3),
        dbc.Col([html.P(['AIC: '], className='par')], width = 2),
        dbc.Col([
            dcc.Loading(id='p2-1-loading', type='circle', children=html.Div([], id = 'stationarity-test55'))
        ], width = 4),
        dbc.Col([], width = 3)
    ]),

    html.Br(),

    ######## GRAFICA #########

    dbc.Row([
        dbc.Col([], width = 1),
        dbc.Col([
            dcc.Loading(id='m1-loading', type='circle', children=dcc.Graph(id='fig-pg15', className='my-graph'))
        ], width = 10),
        dbc.Col([], width = 1)
    ], className='row-content'),

    ####### GRAFICA RESIDUALES ############3
        dbc.Row([
        dbc.Col([], width = 1),
        dbc.Col([
            dcc.Loading(id='m2-loading', type='circle', children=dcc.Graph(id='fig-pg16', className='my-graph'))
        ], width = 5),
        dbc.Col([
            dcc.Loading(id='m3-loading', type='circle', children=dcc.Graph(id='fig-pg17', className='my-graph'))
        ], width = 5),
        dbc.Col([], width = 1)
    ])

 
])

# Update fig
@callback(
    Output(component_id='fig-pg15', component_property='figure'),
    Output(component_id='stationarity-test55', component_property='children'),
    Output(component_id='fig-pg16', component_property='figure'),
    Output(component_id='fig-pg17', component_property='figure'),

    #Output(component_id='fig-pacf', component_property='figure'),
    Input(component_id='train-slider6', component_property='value'),
    Input(component_id='start-gs15', component_property='n_clicks'),
    )
def plot_data(_NumSlider,_StartBtn):
    _data = data.iloc[:6574]
    _dataW = _data['WIND']
    _year = 1961
    n_Wind = len(_data)
    
    nt = 0.06
    _trainD = int(round(n_Wind * (_NumSlider / 100)))

    n_test = n_Wind - _trainD
    train_size = _trainD
    train = _data.wind[:train_size]
    dates_train = _data.date[:train_size]
    test = _data.wind[train_size:train_size + n_test] 
    dates_test = _data.date[train_size:train_size + n_test] 
    ###################### AIC ####
    best_aic = np.inf
    best_bic = np.inf

    best_order = None
    best_mdl = None
    
    pq_rng = range(2)
    d_rng  = range(2)
    if not _StartBtn or _StartBtn <= 0:
        return dash.no_update

    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    # print(i, d, j)
                    tmp_mdl = ARIMA(train, order=(i,d,j)).fit()
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    
    _test_output = dbc.Alert(children=['AIC: {:.4f}'.format(best_aic),html.Br(),'Orden ',html.B(str(best_order), className='alert-bold')], color='success')

    # Fit model
    _best_model = SARIMAX(data['WIND'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)).fit(disp=-1)
    horizon = 365  # número de días en un mes
    _model_forecast = _best_model.get_forecast(steps=horizon)
    prediction_mean = _model_forecast.predicted_mean
    prediction_ci = _model_forecast.conf_int()
    ############################################
    # Crear la figura y añadir los datos históricos y las predicciones
    fig = go.Figure(layout=my_figlayout)
    fig.add_trace(go.Scatter(x=data.index, y=data['WIND'], mode='lines', name='Historical Wind Speed'))
    fig.add_trace(go.Scatter(x=prediction_mean.index, y=prediction_mean, mode='lines', name='Forecast'))

    # Añadir intervalos de confianza
    fig.add_trace(go.Scatter(
    x=prediction_ci.index.tolist() + prediction_ci.index.tolist()[::-1],
    y=prediction_ci.iloc[:, 0].tolist() + prediction_ci.iloc[:, 1].tolist()[::-1],
    fill='toself', 
    fillcolor='rgba(0,100,80,0.2)', 
    line=dict(color='rgba(255,255,255,0)'),
    name='Confidence Interval'
))

# Actualizar el layout del gráfico con límites fijos en el eje Y
    fig.update_layout(
    title='Wind Speed Forecast Using ARIMA Model',
    xaxis_title='Data',
    yaxis_title='Wind Speed',
    yaxis=dict(
        range=[-10, 50]  # Establece el rango del eje Y de -30 a 30
    )
    )
#################################################################
     # Show residuals ACF and PACF
    resid_df = pd.DataFrame(_best_model.resid, columns = ['Residuals'])
    fig1, fig2 = acf_pacf(resid_df, 'Residuals')
    fig1.update_layout(title="Model Residuals: Autocorrelation (ACF)")
    fig2.update_layout(title="Model Residuals: Partial Autocorrelation (PACF)")


  

    return fig, _test_output, fig1, fig2
