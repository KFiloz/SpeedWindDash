import dash
from dash import html
import dash_bootstrap_components as dbc
import psycopg2
from psycopg2 import Error
import pandas as pd

from assets.database import DatabaseManager

# Usar Database para obtener datos
#with DatabaseManager("BDK_owner", "Qde9y0ftCPVg", "ep-rapid-recipe-a57yu1fp.us-east-2.aws.neon.tech", "5432", "BDK") as db:
    #data = db.fetch_data("SELECT * FROM wind")

data = pd.read_csv("D:\Dataviz\SpeedWind\SpeedWindDash\src\data\wind_dataset2.csv")


dash.register_page(__name__, path='/', name='Home', title='Wind | Home')

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([
            html.H3(['Modelo Análitico para la predicción de la velocidad viento']),
            html.Img(src='assets/MWind.jpg', style={'height':'90%', 'width':'80%'}),
            html.P([html.B(['Aerogenerador'])], className='par')
        ], width=12, className='row-titles')
    ]),
    # Guidelines
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            html.P([html.B('Predicción De la velocidad del viento'),html.Br(),
                """Lograr pronósticos de velocidad del viento altamente precisos y confiables representa un desafío significativo
                   para los meteorólogos, dado que los vientos intensos originados por tormentas convectivas pueden provocar
                   daños extensos, incluyendo la destrucción de bosques, cortes de electricidad y daños estructurales a edificaciones.
                   Fenómenos convectivos tales como tormentas eléctricas, tornados, granizo grande y vientos potentes constituyen
                   riesgos naturales capaces de perturbar la cotidianidad, particularmente en regiones de geografía compleja
                   propensas a la convección. Incluso los eventos convectivos menos intensos pueden generar vientos severos
                   con consecuencias devastadoras y costosas. Por ello, pronosticar la velocidad del viento es vital para emitir
                   alertas tempranas de condiciones meteorológicas adversas. Este conjunto de datos incluye lecturas de un
                   sensor meteorológico que registró diversas variables, incluyendo temperatura y precipitación."""
                    ], className='guide'),
            html.P([
                "Para el desarrollo de este trabajo, se tomaron los datos históricos del siguiente ",
                html.A("link", href="https://www.kaggle.com/code/hishaamarmghan/wind-speed-prediction-model/input", target="_blank"),
                "."
              ], className = 'guide'),


            html.P([html.B('Dataset'),html.Br(),
                    """ El conjunto de datos abarca 6574 registros diarios promediados, provenientes de un conjunto de 5 sensores
                        para variables meteorológicas, todos integrados en una estación meteorológica. Dicha estación se situó en un
                        área extensa y despejada, a una altura de 21 metros. La recolección de datos se extendió desde enero de 1961
                        hasta diciembre de 1978, abarcando un total de 17 años. Dentro de la información recopilada se incluyen las
                        precipitaciones medias diarias, así como las temperaturas máxima y mínima y la temperatura mínima sobre
                        la superficie del césped."""], className='guide'),
            html.P([html.B('El dataset contiene 6574 filas y 9 Columnas del DATASET:'),html.Br(),
                    html.Ul([
                    html.Li("DATE (YYYY-MM-DD)"),
                    html.Li("WIND: Average wind speed [knots]"),
                    html.Li("IND: First indicator value"),
                    html.Li("RAIN: Precipitation Amount (mm)"),
                    html.Li("IND.1: Second indicator value"),
                    html.Li("T.MAX: Maximum Temperature (°C)"),
                    html.Li("IND.2: Third indicator value"),
                    html.Li("T.MIN: Minimum Temperature (°C)"),
                    html.Li("T.MIN.G: 09utc Grass Minimum Temperature (°C)")
                    ]) ], className='guide'),

            html.H4(['Dataset:']),
            html.Table([
                html.Thead(
                    html.Tr([
                html.Th(col, style={'minWidth': '100px' if col == 'date' else '62px', 'textAlign': 'center'}) 
                for col in data.columns
                                ])
                                   ),
            html.Tbody([
                 html.Tr([
                html.Td(data.iloc[i][col], style={'textAlign': 'center'}) for col in data.columns
                ]) for i in range(min(len(data), 10))
                ])
             ])

        ], width = 8),
        dbc.Col([], width = 2)
    ])
])