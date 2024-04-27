import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from assets.database import DatabaseManager
from pmdarima.utils import diff
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Usar Database para obtener datos
with DatabaseManager("BDK_owner", "Qde9y0ftCPVg", "ep-rapid-recipe-a57yu1fp.us-east-2.aws.neon.tech", "5432", "BDK") as db:
    data = db.fetch_data("SELECT * FROM wind")


dash.register_page(__name__, name='1-EDA', title='Wind | EDA')

from assets.fig_layout import my_figlayout, my_linelayout
from assets.acf_pacf_plots import acf_pacf



### PAGE LAYOUT ###############################################################################################################

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3(['Análisis Exploratorio'])], width=12, className='row-titles')
    ]),

           ####### MODELO
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.Col([], width = 1),
                    dbc.Col([html.P(['Seleccione ',html.B(['Variable']),'a graficar'], className='par')], width = 10),
                    dbc.Col([], width = 1),
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['Variable']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[
                             {'label': 'Speed Wind', 'value': 'SpeedWind'},
                             {'label': 'Rain', 'value': 'Rain'},
                             {'label': 'Temperatura Max', 'value': 'Tmax'},
                             {'label': 'Temperatura Min', 'value': 'Tmin'},
                             {'label': 'Temperatura Min G', 'value': 'Tming'}],
                              value='0', placeholder='Elija Variable', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='p-from9')], width=8),
                    dbc.Col([], width=1),
                   
                ]),
                ], className = 'div-hyperpar')
            ], width = 8, className = 'col-hyperpar'),
        
        ], style={'margin':'20px 0px 0px 0px'}),

    # raw data fig
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            dcc.Loading(id='p1_1-loading', type='circle', children=dcc.Graph(id='fig-pg1', className='my-graph')),
            html.Br(),
             ], width = 8),
        dbc.Col([], width = 2)
    ], className='row-content'),

    dbc.Row([ 
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-transformed2', className='my-graph'))
        ], width=6, className='multi-graph'),
       dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-boxcox2', className='my-graph'))
        ], width = 6, className='multi-graph'),
    ]),


    dbc.Row([
        dbc.Col([
            dcc.Loading(id='p1_1-loading', type='circle', children=dcc.Graph(id='fig-pg2', className='my-graph')),
             ], width = 6, className='multi-graph'),
        dbc.Col([
            dcc.Loading(id='p1_1-loading', type='circle', children=dcc.Graph(id='fig-pg3', className='my-graph')),
            html.Br(),
             ], width =6, className='multi-graph')
    ]),
])

### PAGE CALLBACKS ###############################################################################################################

# Update fig
@callback(
    Output(component_id='fig-pg1', component_property='figure'),
    Output(component_id='fig-pg2', component_property='figure'),
    Output(component_id='fig-pg3', component_property='figure'),
    Output(component_id='fig-transformed2', component_property='figure'),
    Output(component_id='fig-boxcox2', component_property='figure'),

    Input(component_id='p-from9', component_property='value')
    )
def plot_data(_Variable):
    _data = data.iloc[:2500]
    _dataG = _data['wind']
    _titulo  = 'Wind Speed'
    _tituloD = 'Distribucción Wind Speed'
    _yaxisTitulo = 'Wind'
    _Value = 'wind'
    
    if _Variable == 'SpeedWind':
        _dataG = _data['wind']
        _titulo  = 'Wind Speed'
        _tituloD = 'Distribucción Wind Speed'
        _yaxisTitulo = 'Wind'
        _Value = 'wind'
    if _Variable == 'Rain':
        _dataG = _data['rain']
        _titulo  = 'Rain'
        _tituloD = 'Distribucción Rain'
        _yaxisTitulo = 'Rain'
        _Value = 'rain'
    if _Variable == 'Tmax':
        _dataG = _data['t_max']
        _titulo  = 'Temperatura Maxima'
        _tituloD = 'Distribucción Temp. Max'
        _yaxisTitulo = 'Temp. Max'
        _Value = 't_max'
    if _Variable == 'Tmin':
        _dataG = _data['t_min']
        _titulo  = 'Temperatura Minima'
        _tituloD = 'Distribucción Temp. Min'
        _yaxisTitulo = 'Temp. Min'
        _Value = 't_min'
    if _Variable == 'Tming':
        _dataG = _data['t_min_g']
        _titulo  = 'Temperatura Minima G'
        _tituloD = 'Distribucción Temp. Min G'
        _yaxisTitulo = 'Temp. Min G'
        _Value = 't_min_g'


        

###############################################################################
    fig = None
    fig = go.Figure(layout=my_figlayout)
    fig.add_trace(go.Scatter(x=_data.index, y=_dataG, line=dict()))
    fig.update_layout(title=_titulo, 
                      xaxis_title='Data', 
                      yaxis_title=_yaxisTitulo,
                        height = 500)
    fig.update_traces(overwrite=True, line=my_linelayout)

  
    fig1 = None
    fig1 = go.Figure(layout=my_figlayout)
    fig1.add_trace(go.Histogram(
            x=_dataG,
            xbins=dict(start=_dataG.min(), end=_dataG.max(), size=(_dataG.max() - _dataG.min()) / 50),  # Adjust bin size
            marker_color='#EB89B5',  # Optional: customize color
            opacity=0.75  # Optional: customize opacity
        ))
    fig1.update_layout(
            title_text=_tituloD,  # Add a title
            xaxis_title_text=_yaxisTitulo,  # x-axis label
            yaxis_title_text='Count',  # y-axis label
            bargap=0.1,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinate
        )
    
    fig2 = None
    fig2 = go.Figure(layout=my_figlayout)
    fig2 = fig2.add_trace(go.Scatter(x=_dataG, y=data['wind'], mode='markers', name='Datos'))
    fig2 = fig2.add_trace(go.Scatter(x=[_dataG.min(), _dataG.max()], y=[data['wind'].min(), data['wind'].max()], mode='lines', name='Línea y=x'))
    fig2.update_layout(
        title='Gráfica de dispersión',
        xaxis_title=_yaxisTitulo,
        yaxis_title='Wind',
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False)
        )
  

    _dataT= list(np.log(_dataG))
    # Transformed data linechart
    fig5 = go.Figure(layout=my_figlayout)
    fig5.add_trace(go.Scatter(x=_data.index, y=_dataT, line=dict()))
    fig5.update_layout(title='Transformed Data Linechart', xaxis_title='Time', yaxis_title=_yaxisTitulo)
    fig5.update_traces(overwrite=True, line=my_linelayout)

    fig6 = go.Figure(layout=my_figlayout)


    fig6.add_trace(go.Box(y=_dataG, name=_tituloD ))


    fig6.update_layout(
        title=_tituloD ,
        yaxis_title=_yaxisTitulo,
        xaxis_title=' ',
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        xaxis=dict(showline=True, linewidth=2, linecolor='black')
        )




 

   
    return fig, fig1, fig2, fig5, fig6



