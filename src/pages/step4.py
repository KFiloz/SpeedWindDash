import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import product
from assets.database import DatabaseManager
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.var_model import VAR


_trainp = 80



# Usar Database para obtener datos
with DatabaseManager("BDK_owner", "Qde9y0ftCPVg", "ep-rapid-recipe-a57yu1fp.us-east-2.aws.neon.tech", "5432", "BDK") as db:
    data = db.fetch_data("SELECT * FROM wind")

dash.register_page(__name__, name='4-Vector Autoregression - Exponential Smoothing', title='Wind | 4-VAR')


from assets.modelos import reg_lineal
from assets.fig_layout import my_figlayout, train_linelayout, test_linelayout, pred_linelayout
from assets.acf_pacf_plots import acf_pacf


layout = dbc.Container([
          # titulo
    dbc.Row([
        dbc.Col([
            html.H3(['MODELOS ANALITICOS']),
            html.P([html.B(['SERIES DE TIEMPO'])], className='par')
                ], width=12, className='row-titles')
          ]),
     
         ####### MODELO
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.Col([], width = 1),
                    dbc.Col([html.P(['Seleccione ',html.B(['Modelo']),' a entrenar '], className='par')], width = 10),
                    dbc.Col([], width = 1),
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['Modelo']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[
                             {'label': 'Exponential Smoothing', 'value': 'Es'},
                             {'label': 'Vector Autoregression', 'value': 'Var'}],
                              value='0', placeholder='Elija Modelo', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='p-from6')], width=8),
                    dbc.Col([], width=1),
                ]),
                ], className = 'div-hyperpar')
            ], width = 8, className = 'col-hyperpar'),
        
        ], style={'margin':'20px 0px 0px 0px'}),
    ####### BOTON CALCULAR HYPERPARAMETRO #######
    dbc.Row([
        dbc.Col([], width=3),
        dbc.Col([
            html.P(['MODELO : ', html.B([], id='comb-nr6')], className='par')
        ], width=3),
        dbc.Col([
            html.Button('Start Entrenamiento', id='start-gs6', n_clicks=0, title='The grid search may take several minutes', className='my-button')
        ], width=3, style={'text-align':'left', 'margin':'5px 1px 1px 1px'}),
        dbc.Col([], width=3)
    ]),
    
     dbc.Row([
        dbc.Col([], width = 1),
        dbc.Col([
            dcc.Loading(id='m1-loading', type='circle', children=dcc.Graph(id='fig-pg6', className='my-graph'))
        ], width = 10),
        dbc.Col([], width = 1)
    ], className='row-content')

#################### CIERRA CONTENEDOR ######################

        ])

####################CALL BACK #############################

# Entrenamiento Modelo
@callback(
    Output(component_id='comb-nr6', component_property='children'),
    Output(component_id='fig-pg6', component_property='figure'),

    Input(component_id='p-from6', component_property='value'),
    Input(component_id='start-gs6', component_property='n_clicks'),
    )


def results(_Modelo,_StartBtn):
    # Calculate combinations
    _data = data.iloc[:6574]
    # Creando un DataFrame directamente con solo las columnas de interés
    _df = pd.DataFrame({
        'wind': _data['wind'],
        'rain': _data['rain'],
        't_max': _data['t_max'],
        't_min': _data['t_min'],
     't_min_g': _data['t_min_g']
    })
    _df = _data[['wind', 'rain', 't_max', 't_min', 't_min_g','date']].copy()
   
   
    _df['date'] = pd.to_datetime(_df['date'])
    _df.set_index('date', inplace=True)
    _df = _df.apply(pd.to_numeric)
   

   
    if not _StartBtn or _StartBtn <= 0:
        return dash.no_update

   
    ###################3
    #NomModelo = None

    if _Modelo:
        if _Modelo == 'Es':
           
            title_ = html.P([html.B(['Exponential Smoothing'])], className='par')
            NomModelo = [html.Hr([], className = 'hr-footer'), title_]
            model = ExponentialSmoothing(data['wind'], trend='add', seasonal='add', seasonal_periods=500).fit()
            pred = model.forecast(365)
            _title = 'Wind Speed Forecast Usando Exponential Smoothing'
            _datax = pred.index
            _datay = pred
        elif _Modelo == 'Var':
            
            title_ = html.P([html.B(['Vector Autoregression'])], className='par')
            NomModelo = [html.Hr([], className = 'hr-footer'), title_]
            # Ajustar el modelo VAR
            model = VAR(_df)
            results = model.fit(maxlags=15, ic='aic')
          
            # Realizar predicciones
            lag_order = results.k_ar
            forecasted_values = results.forecast(_df.values[-lag_order:], steps=365)

            # Obtener el DataFrame de predicciones
            forecast_index = pd.date_range(start=_df.index[-1] + pd.Timedelta(days=1), periods=365, freq='D')
            forecast_df = pd.DataFrame(forecasted_values, index=forecast_index, columns=_df.columns)

            # Obtener intervalos de confianza (95% por defecto)
            forecast_interval = results.forecast_interval(_df.values[-lag_order:], steps=5)

           
            _title = 'Wind Speed Forecast Using Vector Autoregression (VAR)'
            _datax = forecast_df.index
            _datay = forecast_df['wind']
               
     
    else:
        NomModelo = 'No model selected'
        #table = "No data"
 
    
    # Preparar el gráfico
    fig = go.Figure(layout=my_figlayout)

    # Añadir datos históricos
    fig.add_trace(go.Scatter(x=data.index, y=data['wind'], mode='lines', name='Historical Wind Speed'))

    # Añadir predicciones
    fig.add_trace(go.Scatter(x=_datax, y=_datay, mode='lines', name='Forecasted Wind Speed'))
  
   
    # Configurar el layout del gráfico
    fig.update_layout(
    title=_title,
    xaxis_title='Data',
    yaxis_title='Wind Speed',
    height=500
    )
    
    return NomModelo,fig




  #####################   
    

   
  



