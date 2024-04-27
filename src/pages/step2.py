import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pmdarima.utils import diff
from statsmodels.tsa.stattools import adfuller
from assets.database import DatabaseManager

# Usar Database para obtener datos
with DatabaseManager("BDK_owner", "Qde9y0ftCPVg", "ep-rapid-recipe-a57yu1fp.us-east-2.aws.neon.tech", "5432", "BDK") as db:
    data = db.fetch_data("SELECT * FROM wind")

dash.register_page(__name__, name='2- Estacionalidad', title='Wind | Estacionalidad')

from assets.fig_layout import my_figlayout, my_linelayout
from assets.acf_pacf_plots import acf_pacf

years = list(range(1961, 1979))


### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3(['Análisis de Estacionalidad'])], width=12, className='row-titles')
    ]),
    # seleccione año
       dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.Col([], width = 1),
                    dbc.Col([html.P(['Seleccione ',html.B(['Año ']),' a graficar'], className='par')], width = 10),
                    dbc.Col([], width = 1),
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['Año']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[{'label': str(year), 'value': year} for year in years],
                              value='', placeholder='Elija Año', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='p-from22')], width=8),
                    dbc.Col([], width=1),
                   
                ]),
                ], className = 'div-hyperpar')
            ], width = 8, className = 'col-hyperpar'),
        
        ], style={'margin':'20px 0px 0px 0px'}),
    
    # Augmented Dickey-Fuller test
    dbc.Row([
        dbc.Col([], width = 3),
        dbc.Col([html.P(['Augmented Dickey-Fuller test: '], className='par')], width = 2),
        dbc.Col([
            dcc.Loading(id='p2-1-loading', type='circle', children=html.Div([], id = 'stationarity-test12'))
        ], width = 4),
        dbc.Col([], width = 3)
    ]),
    # Graphs
     # raw data fig
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            dcc.Loading(id='p1_1-loading', type='circle', children=dcc.Graph(id='fig-pg12', className='my-graph')),
            html.Br(),
             ], width = 8),
        dbc.Col([], width = 2)
    ], className='row-content'),




    dbc.Row([ 
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-transformed12', className='my-graph'))
        ], width=6, className='multi-graph'),
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-acf12', className='my-graph'))
        ], width=6, className='multi-graph')
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-boxcox12', className='my-graph'))
        ], width = 6, className='multi-graph'),
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-pacf12', className='my-graph'))
        ], width=6, className='multi-graph')
    ])

])


### PAGE CALLBACKS ###############################################################################################################

#################
# Apply transformations to the data
@callback(
    Output(component_id='stationarity-test12', component_property='children'),
    Output(component_id='fig-transformed12', component_property='figure'),
    Output(component_id='fig-boxcox12', component_property='figure'),
    Output(component_id='fig-acf12', component_property='figure'),
    Output(component_id='fig-pacf12', component_property='figure'),
    Output(component_id='fig-pg12', component_property='figure'),
    
    
    Input(component_id='p-from22', component_property='value'),
  
)


def data_transform(_year):
    _data = data.iloc[:6574]
    _dataW = _data['wind']
    _year = _year

    years = _data['year'].unique()

    # Perform test
    stat_test = adfuller(_dataW)
    pv = stat_test[1]
    if pv <= .05: # p-value
        #Stationary
        _test_output = dbc.Alert(children=['Test p-value: {:.4f}'.format(pv),html.Br(),'The data is ',html.B(['stationary'], className='alert-bold')], color='success')
    else:
        _test_output =dbc.Alert(children=['Test p-value: {:.4f}'.format(pv),html.Br(),'The data is ',html.B(['not stationary'], className='alert-bold')], color='danger')
    # Charts
   

        # Transformed data linechart
    fig1 = go.Figure(layout=my_figlayout)
    fig1.add_trace(go.Scatter(x=_data.index, y=_dataW, line=dict()))
    fig1.update_layout(title='Transformed Data Linechart', xaxis_title='Time', yaxis_title='_yaxisTitulo')
    fig1.update_traces(overwrite=True, line=my_linelayout)


    # Box-Cox plot
    fig2 = go.Figure(layout=my_figlayout)
    fig2.add_trace(go.Scatter(x=_data.index, y=_dataW, line=dict()))
    fig2.update_layout(title='Transformed Data Linechart', xaxis_title='Time', yaxis_title='_yaxisTitulo')
    fig2.update_traces(overwrite=True, line=my_linelayout)
    # ACF, PACF
    fig3, fig4 = acf_pacf(_data, 'wind')
   

    fig5 = None
    fig5 = go.Figure(layout=my_figlayout)
    filtered_df = _data[_data['year'] == _year]
    
  
    fig5.add_trace(go.Scatter(x=filtered_df.index, y=_dataW, mode='lines', name=str(_year)))
    
    fig5.update_layout(title='Grafico de serie temporal por cada año', 
                      xaxis_title=_year, 
                      yaxis_title='Wind',
                        height = 500)
    fig5.update_traces(overwrite=True, line=my_linelayout)

    return _test_output, fig1, fig2, fig3, fig4, fig5



